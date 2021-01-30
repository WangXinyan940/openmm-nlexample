#define WARPS_PER_GROUP (THREAD_BLOCK_SIZE/TILE_SIZE)


typedef struct {
    real x, y, z;
    real prm;
    real fx, fy, fz;
    int idx;
} AtomData;


/**
 * Compute nonbonded interactions. The kernel is separated into two parts,
 * tiles with exclusions and tiles without exclusions. It relies heavily on 
 * implicit warp-level synchronization. A tile is defined by two atom blocks 
 * each of warpsize. Each warp computes a range of tiles.
 * 
 * Tiles with exclusions compute the entire set of interactions across
 * atom blocks, equal to warpsize*warpsize. In order to avoid access conflicts 
 * the forces are computed and accumulated diagonally in the manner shown below
 * where, suppose
 *
 * [a-h] comprise atom block 1, [i-p] comprise atom block 2
 *
 * 1 denotes the first set of calculations within the warp
 * 2 denotes the second set of calculations within the warp
 * ... etc.
 * 
 *        threads
 *     0 1 2 3 4 5 6 7
 *         atom1 
 * L    a b c d e f g h 
 * o  i 1 2 3 4 5 6 7 8
 * c  j 8 1 2 3 4 5 6 7
 * a  k 7 8 1 2 3 4 5 6
 * l  l 6 7 8 1 2 3 4 5
 * D  m 5 6 7 8 1 2 3 4 
 * a  n 4 5 6 7 8 1 2 3
 * t  o 3 4 5 6 7 8 1 2
 * a  p 2 3 4 5 6 7 8 1
 *
 * Tiles without exclusions read off directly from the neighbourlist interactingAtoms
 * and follows the same force accumulation method. If more there are more interactingTiles
 * than the size of the neighbourlist initially allocated, the neighbourlist is rebuilt
 * and the full tileset is computed. This should happen on the first step, and very rarely 
 * afterwards.
 *
 * On CUDA devices that support the shuffle intrinsic, on diagonal exclusion tiles use
 * __shfl to broadcast. For all other types of tiles __shfl is used to pass around the 
 * forces, positions, and parameters when computing the forces. 
 *
 * [out]forceBuffers    - forces on each atom to eventually be accumulated
 * [out]energyBuffer    - energyBuffer to eventually be accumulated
 * [in]posq             - x,y,z,charge 
 * [in]exclusions       - 1024-bit flags denoting atom-atom exclusions for each tile
 * [in]exclusionTiles   - x,y denotes the indices of tiles that have an exclusion
 * [in]startTileIndex   - index into first tile to be processed
 * [in]numTileIndices   - number of tiles this context is responsible for processing
 * [in]int tiles        - the atom block for each tile
 * [in]interactionCount - total number of tiles that have an interaction
 * [in]maxTiles         - stores the size of the neighbourlist in case it needs 
 *                      - to be expanded
 * [in]periodicBoxSize  - size of the Periodic Box, last dimension (w) not used
 * [in]invPeriodicBox   - inverse of the periodicBoxSize, pre-computed for speed
 * [in]blockCenter      - the center of each block in euclidean coordinates
 * [in]blockSize        - size of the each block, radiating from the center
 *                      - x is half the distance of total length
 *                      - y is half the distance of total width
 *                      - z is half the distance of total height
 *                      - w is not used
 * [in]interactingAtoms - a list of interactions within a given tile     
 *
 */
extern "C" __global__ void computeNonbonded(
        unsigned long long*     __restrict__   forceBuffers, 
        mixed*                  __restrict__   energyBuffer, 
        const real4*            __restrict__   posq, 
        int*                    __restrict__   atomIndex,
        const tileflags*        __restrict__   exclusions,
        const int2*             __restrict__   exclusionTiles, 
        unsigned int                           startTileIndex, 
        unsigned long long                     numTileIndices, 
        const int*              __restrict__   tiles, 
        const unsigned int*     __restrict__   interactionCount, 
        real4                                  periodicBoxSize, 
        real4                                  invPeriodicBoxSize, 
        real4                                  periodicBoxVecX, 
        real4                                  periodicBoxVecY, 
        real4                                  periodicBoxVecZ, 
        unsigned int                           maxTiles, 
        const real4*            __restrict__   blockCenter,
        const real4*            __restrict__   blockSize, 
        const unsigned int*     __restrict__   interactingAtoms, 
        unsigned int                           maxSinglePairs,
        const int2*             __restrict__   singlePairs,
        real*                   __restrict__   params,
        double                                 cutoff
) {
    const unsigned int totalWarps = (blockDim.x*gridDim.x)/TILE_SIZE;
    const unsigned int warp = (blockIdx.x*blockDim.x+threadIdx.x)/TILE_SIZE; // global warpIndex
    const unsigned int tgx = threadIdx.x & (TILE_SIZE-1); // index within the warp
    const unsigned int tbx = threadIdx.x - tgx;           // block warpIndex
    mixed energy = 0;
    real cutoff2 = cutoff * cutoff;
    // used shared memory if the device cannot shuffle
    __shared__ AtomData localData[THREAD_BLOCK_SIZE];

    // First loop: process tiles that contain exclusions.

    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const int2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        real3 force = make_real3(0);
        unsigned int atom1 = x*TILE_SIZE + tgx;
        real4 posq1 = posq[atom1];

        // LOAD_ATOM1_PARAMETERS
        AtomData atom1Data;
        atom1Data.x = posq1.x;
        atom1Data.y = posq1.y;
        atom1Data.z = posq1.z;
        atom1Data.fx = 0.0;
        atom1Data.fy = 0.0;
        atom1Data.fz = 0.0;
        atom1Data.prm = params[atomIndex[atom1]];
        atom1Data.idx = atomIndex[atom1];

        tileflags excl = exclusions[pos*TILE_SIZE+tgx];
        const bool hasExclusions = true;
        if (x == y) {
            // This tile is on the diagonal.
            localData[threadIdx.x].x = posq1.x;
            localData[threadIdx.x].y = posq1.y;
            localData[threadIdx.x].z = posq1.z;
            localData[threadIdx.x].fx = 0;
            localData[threadIdx.x].fy = 0;
            localData[threadIdx.x].fz = 0;
            localData[threadIdx.x].prm = params[atomIndex[atom1]];
            localData[threadIdx.x].idx = atomIndex[atom1];

            // we do not need to fetch parameters from global since this is a symmetric tile
            // instead we can broadcast the values using shuffle
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+j;
                real3 posq2 = make_real3(localData[atom2].x, localData[atom2].y, localData[atom2].z);
                real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
                APPLY_PERIODIC_TO_DELTA(delta)
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                real invR = RSQRT(r2);
                real r = r2*invR;
                // LOAD_ATOM2_PARAMETERS
                AtomData atom2Data;
                atom2Data.x = posq2.x;
                atom2Data.y = posq2.y;
                atom2Data.z = posq2.z;
                atom2Data.fx = 0;
                atom2Data.fy = 0;
                atom2Data.fz = 0;
                atom2 = y*TILE_SIZE+j;
                atom2Data.prm = params[atomIndex[atom2]];
                atom2Data.idx = atomIndex[atom2];

                real dEdR = 0.0f;
                bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS || !(excl & 0x1));
                real tempEnergy = 0.0f;
                const real interactionScale = 0.5f;
                // COMPUTE_INTERACTION
                tempEnergy += atom1Data.prm * atom2Data.prm * invR * invR;
                dEdR += 2.0 * atom1Data.prm * atom2Data.prm * invR * invR * invR * invR;
                printf("1: %i %i %f %f %f\n", atom1Data.idx, atom2Data.idx, atom1Data.prm, atom2Data.prm, r);

                energy += 0.5f*tempEnergy;
                force.x -= delta.x*dEdR;
                force.y -= delta.y*dEdR;
                force.z -= delta.z*dEdR;
                excl >>= 1;
            }
        }
        else {
            // This is an off-diagonal tile.
            unsigned int j = y*TILE_SIZE + tgx;
            real4 shflPosq = posq[j];
            localData[threadIdx.x].x = shflPosq.x;
            localData[threadIdx.x].y = shflPosq.y;
            localData[threadIdx.x].z = shflPosq.z;
            localData[threadIdx.x].fx = 0.0f;
            localData[threadIdx.x].fy = 0.0f;
            localData[threadIdx.x].fz = 0.0f;
            // LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
            localData[threadIdx.x].prm = params[atomIndex[j]];
            localData[threadIdx.x].idx = atomIndex[j];

            excl = (excl >> tgx) | (excl << (TILE_SIZE - tgx));
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+tj;
                real3 posq2 = make_real3(localData[atom2].x, localData[atom2].y, localData[atom2].z);
                real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
                APPLY_PERIODIC_TO_DELTA(delta)
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                real invR = RSQRT(r2);
                real r = r2*invR;
                // LOAD_ATOM2_PARAMETERS
                atom2 = y*TILE_SIZE+tj;
                AtomData atom2Data;
                atom2Data.x = posq2.x;
                atom2Data.y = posq2.y;
                atom2Data.z = posq2.z;
                atom2Data.fx = 0.0;
                atom2Data.fy = 0.0;
                atom2Data.fz = 0.0;
                atom2Data.prm = params[atomIndex[atom2]];
                atom2Data.idx = atomIndex[atom2];

                
                real dEdR = 0.0f;
                bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS || !(excl & 0x1));
                real tempEnergy = 0.0f;
                const real interactionScale = 1.0f;
                // COMPUTE_INTERACTION
                tempEnergy += atom1Data.prm * atom2Data.prm * invR * invR;
                dEdR += 2.0 * atom1Data.prm * atom2Data.prm * invR * invR * invR * invR;
                printf("2: %i %i %f %f %f\n", atom1Data.idx, atom2Data.idx, atom1Data.prm, atom2Data.prm, r);

                energy += tempEnergy;
                delta *= dEdR;
                force.x -= delta.x;
                force.y -= delta.y;
                force.z -= delta.z;
                localData[tbx+tj].fx += delta.x;
                localData[tbx+tj].fy += delta.y;
                localData[tbx+tj].fz += delta.z;
                excl >>= 1;
                // cycles the indices
                // 0 1 2 3 4 5 6 7 -> 1 2 3 4 5 6 7 0
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            const unsigned int offset = y*TILE_SIZE + tgx;
            // write results for off diagonal tiles
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fx*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fy*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fz*0x100000000)));
        }
        // Write results for on and off diagonal tiles
        const unsigned int offset = x*TILE_SIZE + tgx;
        atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
    }

    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).

    const unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return; // There wasn't enough memory for the neighbor list.
    int pos = (int) (warp*(long long)numTiles/totalWarps);
    int end = (int) ((warp+1)*(long long)numTiles/totalWarps);
    int skipBase = 0;
    int currentSkipIndex = tbx;
    // atomIndices can probably be shuffled as well
    // but it probably wouldn't make things any faster
    __shared__ int atomIndices[THREAD_BLOCK_SIZE];
    __shared__ volatile int skipTiles[THREAD_BLOCK_SIZE];
    skipTiles[threadIdx.x] = -1;
    
    while (pos < end) {
        const bool hasExclusions = false;
        real3 force = make_real3(0);
        bool includeTile = true;

        // Extract the coordinates of this tile.
        int x, y;
        x = tiles[pos];
        real4 blockSizeX = blockSize[x];
        if (includeTile) {
            unsigned int atom1 = x*TILE_SIZE + tgx;
            // Load atom data for this tile.
            real4 posq1 = posq[atom1];
            // LOAD_ATOM1_PARAMETERS
            AtomData atom1Data;
            atom1Data.x = posq1.x;
            atom1Data.y = posq1.y;
            atom1Data.z = posq1.z;
            atom1Data.fx = 0.0;
            atom1Data.fy = 0.0;
            atom1Data.fz = 0.0;
            atom1Data.prm = params[atomIndex[atom1]];
            atom1Data.idx = atomIndex[atom1];

            //const unsigned int localAtomIndex = threadIdx.x;

            unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];

            atomIndices[threadIdx.x] = j;

            if (j < PADDED_NUM_ATOMS) {
                // Load position of atom j from from global memory
                localData[threadIdx.x].x = posq[j].x;
                localData[threadIdx.x].y = posq[j].y;
                localData[threadIdx.x].z = posq[j].z;
                localData[threadIdx.x].fx = 0.0f;
                localData[threadIdx.x].fy = 0.0f;
                localData[threadIdx.x].fz = 0.0f;
                localData[threadIdx.x].prm = params[atomIndex[j]];
                localData[threadIdx.x].idx = atomIndex[j];
            }
            else {

                localData[threadIdx.x].x = 0;
                localData[threadIdx.x].y = 0;
                localData[threadIdx.x].z = 0;
                localData[threadIdx.x].prm = 0;
                localData[threadIdx.x].idx = 0;
            }

            // We need to apply periodic boundary conditions separately for each interaction.
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+tj;
                real3 posq2 = make_real3(localData[atom2].x, localData[atom2].y, localData[atom2].z);
                real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
                APPLY_PERIODIC_TO_DELTA(delta)

                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;

                real invR = RSQRT(r2);
                real r = r2*invR;

                atom2 = atomIndices[tbx+tj];

                bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS);
                if (!isExcluded && r2 < cutoff2) {
                    // LOAD_ATOM2_PARAMETERS
                    AtomData atom2Data;
                    atom2Data.x = posq2.x;
                    atom2Data.y = posq2.y;
                    atom2Data.z = posq2.z;
                    atom2Data.fx = 0.0;
                    atom2Data.fy = 0.0;
                    atom2Data.fz = 0.0;
                    atom2Data.prm = params[atomIndex[atom2]];
                    atom2Data.idx = atomIndex[atom2];

                    real dEdR = 0.0f;

                    real tempEnergy = 0.0f;
                    const real interactionScale = 1.0f;
                    // COMPUTE_INTERACTION
                    tempEnergy += atom1Data.prm * atom2Data.prm * invR * invR;
                    dEdR += 2.0 * atom1Data.prm * atom2Data.prm * invR * invR * invR * invR;
                    printf("3: %i %i %f %f %f\n", atom1Data.idx, atom2Data.idx, atom1Data.prm, atom2Data.prm, r);

                    energy += tempEnergy;

                    delta *= dEdR;
                    force.x -= delta.x;
                    force.y -= delta.y;
                    force.z -= delta.z;

                    localData[tbx+tj].fx += delta.x;
                    localData[tbx+tj].fy += delta.y;
                    localData[tbx+tj].fz += delta.z;
                        
                }
                    
                tj = (tj + 1) & (TILE_SIZE - 1);
            }


            // Write results.

            atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
            atomicAdd(&forceBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
            atomicAdd(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.z*0x100000000)));

            int atom2 = atomIndices[threadIdx.x];

            if (atom2 < PADDED_NUM_ATOMS) {

                atomicAdd(&forceBuffers[atom2], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fx*0x100000000)));
                atomicAdd(&forceBuffers[atom2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fy*0x100000000)));
                atomicAdd(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fz*0x100000000)));

            }
        }
        pos++;
    }
    
    // Third loop: single pairs that aren't part of a tile.
    const unsigned int numPairs = interactionCount[1];
    if (numPairs > maxSinglePairs) {
        printf("Out 3\n");
        return; // There wasn't enough memory for the neighbor list.
    }
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numPairs; i += blockDim.x*gridDim.x) {
        int2 pair = singlePairs[i];
        int atom1 = pair.x;
        int atom2 = pair.y;
        real3 force = make_real3(0);
        real4 posq1 = posq[atom1];
        real4 posq2 = posq[atom2];
        // LOAD_ATOM1_PARAMETERS
        AtomData atom1Data;
        atom1Data.x = posq1.x;
        atom1Data.y = posq1.y;
        atom1Data.z = posq1.z;
        atom1Data.fx = 0.0;
        atom1Data.fy = 0.0;
        atom1Data.fz = 0.0;
        atom1Data.prm = params[atomIndex[atom1]];
        atom1Data.idx = atomIndex[atom1];

        // int j = atom2;
        // atom2 = threadIdx.x;
        
        // LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
        // LOAD_ATOM2_PARAMETERS
        // atom2 = pair.y;
        AtomData atom2Data;
        atom2Data.x = posq2.x;
        atom2Data.y = posq2.y;
        atom2Data.z = posq2.z;
        atom2Data.fx = 0.0;
        atom2Data.fy = 0.0;
        atom2Data.fz = 0.0;
        atom2Data.prm = params[atomIndex[atom2]];
        atom2Data.idx = atomIndex[atom2];

        real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);

        APPLY_PERIODIC_TO_DELTA(delta)

        real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        real invR = RSQRT(r2);
        real r = r2*invR;

        real dEdR = 0.0f;


        bool hasExclusions = false;
        bool isExcluded = false;
        real tempEnergy = 0.0f;
        const real interactionScale = 1.0f;
        // COMPUTE_INTERACTION
        tempEnergy += atom1Data.prm * atom2Data.prm * invR * invR;
        dEdR += 2.0 * atom1Data.prm * atom2Data.prm * invR * invR * invR * invR;
        printf("4: %i %i %f %f %f\n", atom1Data.idx, atom2Data.idx, atom1Data.prm, atom2Data.prm, r);

        energy += tempEnergy;

        delta *= dEdR;
        force.x -= delta.x;
        force.y -= delta.y;
        force.z -= delta.z;

        atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long) (-force.x*0x100000000)));
        atomicAdd(&forceBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-force.y*0x100000000)));
        atomicAdd(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-force.z*0x100000000)));
        atomicAdd(&forceBuffers[atom2], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[atom2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
    }

    energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;
}
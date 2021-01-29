#define WARPS_PER_GROUP (THREAD_BLOCK_SIZE/TILE_SIZE)

typedef struct {
    real3 pos, force;
    real param;
} AtomData;

inline __device__ void loadAtomData(AtomData& data, int atom, const real4* __restrict__ posq, real prm) {
    real4 atomPosq = posq[atom];
    data.pos = make_real3(atomPosq.x, atomPosq.y, atomPosq.z);
    data.param = prm;
}

__device__ void computeOneInteraction(AtomData& atom1, AtomData& atom2, bool hasExclusions, mixed& energy, real4& periodicBoxSize, real4& invPeriodicBoxSize, real4& periodicBoxVecX, real4& periodicBoxVecY, real4& periodicBoxVecZ) {
    if (hasExclusions) {
        // Compute the displacement.
        real3 delta = make_real3(atom2.pos.x - atom1.pos.x, atom2.pos.y - atom1.pos.y, atom2.pos.z - atom1.pos.z);
        APPLY_PERIODIC_TO_DELTA(delta)
        real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        real rInv = RSQRT(r2);
        // real r = r2*rInv;
        real p1p2 = atom1.param * atom2.param;
        energy += p1p2 * rInv * rInv;
        mixed dEdRdR = - 2 * p1p2 * rInv * rInv * rInv * rInv;
        atom1.force.x += dEdRdR * delta.x;
        atom1.force.y += dEdRdR * delta.y;
        atom1.force.z += dEdRdR * delta.z;
        atom2.force.x -= dEdRdR * delta.x;
        atom2.force.y -= dEdRdR * delta.y;
        atom2.force.z -= dEdRdR * delta.z;
    }
}

extern "C" __global__ void calcTestForcePBC(
    mixed*              __restrict__     energyBuffer,
    real4*              __restrict__     posq,
    unsigned long long* __restrict__     forceBuffers,
    real*               __restrict__     params,
    int*                __restrict__     atomIndex,
    const int2*         __restrict__     exclusionTiles, 
    unsigned int                         startTileIndex, 
    unsigned int                         numTileIndices,
    const int*          __restrict__     tiles,
    const unsigned int* __restrict__     interactionCount,
    const unsigned int* __restrict__     interactingAtoms,
    unsigned int                         maxTiles, 
    real4                                periodicBoxSize, 
    real4                                invPeriodicBoxSize,
    real4                                periodicBoxVecX, 
    real4                                periodicBoxVecY, 
    real4                                periodicBoxVecZ,
    int                                  numParticles,
    int                                  paddedNumAtoms
) {
    const unsigned int totalWarps = (blockDim.x*gridDim.x)/TILE_SIZE;
    const unsigned int warp = (blockIdx.x*blockDim.x+threadIdx.x)/TILE_SIZE;
    const unsigned int tgx = threadIdx.x & (TILE_SIZE-1);
    const unsigned int tbx = threadIdx.x - tgx;
    mixed energy = 0;
    __shared__ AtomData localData[THREAD_BLOCK_SIZE];

    // First loop: process tiles that contain exclusions.
    
    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const int2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        AtomData data;
        unsigned int atom1 = x*TILE_SIZE + tgx;
        // !!!!! Here load atom data of atom1 !!!!!
        loadAtomData(data, atom1, posq, params[atomIndex[atom1]]);
        data.force = make_real3(0);
        if (x == y) {
            // This tile is on the diagonal.
            localData[threadIdx.x].pos = data.pos;

            // Compute forces.

            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int atom2 = y*TILE_SIZE+j;
                if (atom1 != atom2 && atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    // !!!! Here calc atom1-atom2 interaction !!!!
                    computeOneInteraction(data, localData[tbx+j], true, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                }
            }
            atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long) (data.force.x*0x100000000)));
            atomicAdd(&forceBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.force.y*0x100000000)));
            atomicAdd(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.force.z*0x100000000)));
        }
        else {
            // This is an off-diagonal tile.

            unsigned int j = y*TILE_SIZE + tgx;
            loadAtomData(localData[threadIdx.x], j, posq, params[atomIndex[j]]);
            localData[threadIdx.x].force = make_real3(0);
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = y*TILE_SIZE+tj;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    computeOneInteraction(data, localData[tbx+tj], true, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            unsigned int offset = x*TILE_SIZE + tgx;
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (data.force.x*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.force.y*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.force.z*0x100000000)));
            offset = y*TILE_SIZE + tgx;
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].force.x*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].force.y*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].force.z*0x100000000)));
        }
    }

    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).

    const unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return; // There wasn't enough memory for the neighbor list.
    int pos = (int) (numTiles > maxTiles ? startTileIndex+warp*(long long)numTileIndices/totalWarps : warp*(long long)numTiles/totalWarps);
    int end = (int) (numTiles > maxTiles ? startTileIndex+(warp+1)*(long long)numTileIndices/totalWarps : (warp+1)*(long long)numTiles/totalWarps);
    __shared__ int atomIndices[THREAD_BLOCK_SIZE];
    __shared__ volatile int skipTiles[THREAD_BLOCK_SIZE];
    skipTiles[threadIdx.x] = -1;
    
    while (pos < end) {
        bool includeTile = true;

        // Extract the coordinates of this tile.
        
        int x, y;
        x = tiles[pos];
        if (includeTile) {
            unsigned int atom1 = x*TILE_SIZE + tgx;

            // Load atom data for this tile.

            AtomData data;
            loadAtomData(data, atom1, posq, params[atomIndex[atom1]]);
            data.force = make_real3(0);
            unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];
            atomIndices[threadIdx.x] = j;
            loadAtomData(localData[threadIdx.x], j, posq, params[atomIndex[j]]);
            localData[threadIdx.x].force = make_real3(0);

            // Compute forces.

            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = atomIndices[tbx+tj];
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    computeOneInteraction(data, localData[tbx+tj], false, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }

            // Write results.

            unsigned int offset = x*TILE_SIZE + tgx;
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (data.force.x*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.force.y*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.force.z*0x100000000)));
            offset = atomIndices[threadIdx.x];
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].force.x*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].force.y*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].force.z*0x100000000)));
        }
        pos++;
    }
    energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;
}
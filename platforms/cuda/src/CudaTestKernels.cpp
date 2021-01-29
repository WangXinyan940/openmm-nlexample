#include "CudaTestKernels.h"
#include "CudaTestKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaNonbondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/cuda/CudaParameterSet.h"
#include "CudaKernelSources.h"
#include <map>
#include <set>
#include <iostream>
#include <utility>

using namespace TestPlugin;
using namespace OpenMM;
using namespace std;

CudaCalcTestForceKernel::~CudaCalcTestForceKernel() {
}

void CudaCalcTestForceKernel::initialize(const System& system, const TestForce& force) {
    cu.setAsCurrent();

    int numParticles = system.getNumParticles();
    int elementSize = cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float);

    ifPBC = force.usesPeriodicBoundaryConditions();
    cutoff = force.getCutoffDistance();

    // Inititalize CUDA objects.
    // if noPBC
    set<pair<int, int>> tilesWithExclusions;
    // for (int atom1 = 0; atom1 < (int) exclusions.size(); ++atom1) {
    //     int x = atom1/CudaContext::TileSize;
    //     for (int atom2 : exclusions[atom1]) {
    //         int y = atom2/CudaContext::TileSize;
    //         tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
    //     }
    // }

    if (!ifPBC){
        map<string, string> defines;
        CUmodule module = cu.createModule(CudaKernelSources::vectorOps + CudaTestKernelSources::noPBCForce, defines);
        calcTestForceNoPBCKernel = cu.getKernel(module, "calcTestForceNoPBC");
        vector<int> idx0;
        vector<int> idx1;
        idx0.resize(numParticles*(numParticles-1)/2);
        idx1.resize(numParticles*(numParticles-1)/2);
        int count = 0;
        for(int ii=0;ii<numParticles;ii++){
            for(int jj=ii+1;jj<numParticles;jj++){
                idx0[count] = ii;
                idx1[count] = jj;
                count += 1;
            }
        }
        pairidx0.initialize(cu, numParticles*(numParticles-1)/2, sizeof(int), "index0");
        pairidx1.initialize(cu, numParticles*(numParticles-1)/2, sizeof(int), "index1");
        pairidx0.upload(idx0);
        pairidx1.upload(idx1);
    } else {
        map<string, string> pbcDefines;
        pbcDefines["NUM_ATOMS"] = cu.intToString(numParticles);
        pbcDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        pbcDefines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
        defines["THREAD_BLOCK_SIZE"] = cu.intToString(cu.getNonbondedUtilities().getForceThreadBlockSize());

        pbcDefines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
        int numExclusionTiles = tilesWithExclusions.size();
        pbcDefines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
        int numContexts = cu.getPlatformData().contexts.size();
        int startExclusionIndex = cu.getContextIndex()*numExclusionTiles/numContexts;
        int endExclusionIndex = (cu.getContextIndex()+1)*numExclusionTiles/numContexts;
        pbcDefines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
        pbcDefines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);

        // macro for short-range
        CUmodule PBCModule = cu.createModule(CudaKernelSources::vectorOps + CudaTestKernelSources::PBCForce, pbcDefines);
        calcTestForcePBCKernel = cu.getKernel(PBCModule, "calcTestForcePBC");

        vector<vector<int>> exclusions;
        cu.getNonbondedUtilities().addInteraction(true, true, true, cutoff, exclusions, "", force.getForceGroup());
    }

    hasInitializedKernel = true;
}

double CudaCalcTestForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    int numParticles = cu.getNumAtoms();
    double energy = 0.0;
    if (ifPBC){
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
        int startTileIndex = nb.getStartTileIndex();
        int numTileIndices = nb.getNumTiles();
        unsigned int maxTiles = nb.getInteractingTiles().getSize();
        void* args[] = {&cu.getEnergyBuffer().getDevicePointer(), &cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), 
            &nb.getExclusionTiles().getDevicePointer(), &startTileIndex, &numTileIndices,
            &nb.getInteractingTiles().getDevicePointer(), &nb.getInteractionCount().getDevicePointer(), &nb.getInteractingAtoms().getDevicePointer(), &maxTiles,
            cu.getPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), 
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecZPointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(calcTestForcePBCKernel, args, numParticles);
    } else {
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&cu.getEnergyBuffer().getDevicePointer(), &cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), 
            &pairidx0.getDevicePointer(), &pairidx1.getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(calcTestForceNoPBCKernel, args, numParticles*(numParticles-1)/2);
    }
    return energy;
}
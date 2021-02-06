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

class CudaCalcTestForceInfo : public CudaForceInfo {
public:
	CudaCalcTestForceInfo(const TestForce& force) :
			force(force) {
	}
    bool areParticlesIdentical(int particle1, int particle2) {
        double p1, p2;
        p1 = force.getParticleParameter(particle1);
        p2 = force.getParticleParameter(particle2);
        return (p1 == p2);
    }
	int getNumParticleGroups() {
        int natom = force.getNumParticles();
		return natom;
	}
	void getParticlesInGroup(int index, vector<int>& particles) {
		particles.resize(1);
        particles[0] = index;
	}
	bool areGroupsIdentical(int group1, int group2) {
		double p1 = force.getParticleParameter(group1);
        double p2 = force.getParticleParameter(group2);
		return (p1 == p2);
	}
private:
	const TestForce& force;
};

CudaCalcTestForceKernel::~CudaCalcTestForceKernel() {
}

void CudaCalcTestForceKernel::initialize(const System& system, const TestForce& force) {
    cu.setAsCurrent();

    int numParticles = system.getNumParticles();
    int elementSize = cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float);

    ifPBC = force.usesPeriodicBoundaryConditions();
    cutoff = force.getCutoffDistance();

    // vector<vector<int>> exclusions;
    exclusions.resize(numParticles);
    for(int ii=0;ii<numParticles;ii++){
        exclusions[ii].push_back(ii);
    }
    for(int ii=0;ii<force.getNumExclusions();ii++){
        int p1, p2;
        force.getExclusionParticles(ii, p1, p2);
        exclusions[p1].push_back(p2);
        exclusions[p2].push_back(p1);
    }

    // Inititalize CUDA objects.
    // if noPBC
    if (cu.getUseDoublePrecision()){
        vector<double> parameters;
        for(int ii=0;ii<numParticles;ii++){
            double prm = force.getParticleParameter(ii);
            parameters.push_back(prm);
        }
        params.initialize(cu, numParticles, elementSize, "params");
        params.upload(parameters);
    } else {
        vector<float> parameters;
        for(int ii=0;ii<numParticles;ii++){
            float prm = force.getParticleParameter(ii);
            parameters.push_back(prm);
        }
        params.initialize(cu, numParticles, elementSize, "params");
        params.upload(parameters);
    }

    vector<int> exidx0, exidx1;
    exidx0.resize(force.getNumExclusions());
    exidx1.resize(force.getNumExclusions());
    for(int ii=0;ii<force.getNumExclusions();ii++){
        int p1, p2;
        force.getExclusionParticles(ii, p1, p2);
        exidx0[ii] = p1;
        exidx1[ii] = p2;
    }
    expairidx0.initialize(cu, exidx0.size(), sizeof(int), "exindex0");
    expairidx1.initialize(cu, exidx1.size(), sizeof(int), "exindex1");
    expairidx0.upload(exidx0);
    expairidx1.upload(exidx1);
    numexclusions = exidx0.size();

    if (!ifPBC){
        map<string, string> defines;
        CUmodule module = cu.createModule(CudaKernelSources::vectorOps + CudaTestKernelSources::noPBCForce, defines);
        calcTestForceNoPBCKernel = cu.getKernel(module, "calcTestForceNoPBC");
        calcExcludeForceNoPBCKernel = cu.getKernel(module, "calcExcludeForceNoPBC");
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

        cu.getNonbondedUtilities().addInteraction(true, true, true, cutoff, exclusions, "", force.getForceGroup());

        set<pair<int, int>> tilesWithExclusions;
        for (int atom1 = 0; atom1 < (int) exclusions.size(); ++atom1) {
            int x = atom1/CudaContext::TileSize;
            for (int atom2 : exclusions[atom1]) {
                int y = atom2/CudaContext::TileSize;
                tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
            }
        }

        vector<int> indexAtomVec;
        indexAtomVec.resize(numParticles);
        indexAtom.initialize(cu, numParticles, sizeof(int), "indexAtom");
        indexAtom.upload(indexAtomVec);

        map<string, string> pbcDefines;
        pbcDefines["NUM_ATOMS"] = cu.intToString(numParticles);
        pbcDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        pbcDefines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
        pbcDefines["THREAD_BLOCK_SIZE"] = cu.intToString(cu.getNonbondedUtilities().getForceThreadBlockSize());

        pbcDefines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
        int numExclusionTiles = tilesWithExclusions.size();
        pbcDefines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
        int numContexts = cu.getPlatformData().contexts.size();
        int startExclusionIndex = cu.getContextIndex()*numExclusionTiles/numContexts;
        int endExclusionIndex = (cu.getContextIndex()+1)*numExclusionTiles/numContexts;
        pbcDefines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
        pbcDefines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);
        pbcDefines["USE_PERIODIC"] = "1";
        pbcDefines["USE_CUTOFF"] = "1";
        pbcDefines["USE_EXCLUSIONS"] = "";
        pbcDefines["USE_SYMMETRIC"] = "1";
        pbcDefines["INCLUDE_FORCES"] = "1";
        pbcDefines["INCLUDE_ENERGY"] = "1";
        pbcDefines["CUTOFF"] = cu.doubleToString(cutoff);

        // macro for short-range
        // CUmodule PBCModule = cu.createModule(CudaKernelSources::vectorOps + CudaTestKernelSources::PBCForce, pbcDefines);
        // calcTestForcePBCKernel = cu.getKernel(PBCModule, "calcTestForcePBC");
        CUmodule PBCModule = cu.createModule(CudaKernelSources::vectorOps + CudaTestKernelSources::PBCForce, pbcDefines);
        calcTestForcePBCKernel = cu.getKernel(PBCModule, "computeNonbonded");
        calcExclusionPBCKernel = cu.getKernel(PBCModule, "computeExclusion");
        indexAtomKernel = cu.getKernel(PBCModule, "genIndexAtom");
    }
    cu.addForce(new CudaCalcTestForceInfo(force));
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
        int maxSinglePairs = nb.getSinglePairs().getSize();

        void* args[] = {
            &cu.getForce().getDevicePointer(),                      // unsigned long long*       __restrict__     forceBuffers, 
            &cu.getEnergyBuffer().getDevicePointer(),               // mixed*                    __restrict__     energyBuffer, 
            &cu.getPosq().getDevicePointer(),                       // const real4*              __restrict__     posq, 
            &params.getDevicePointer(),                             // const real*               __restrict__     params,
            &cu.getAtomIndexArray().getDevicePointer(),             // const int*                __restrict__     atomIndex,
            &nb.getExclusions().getDevicePointer(),                 // const tileflags*          __restrict__     exclusions,
            &nb.getExclusionTiles().getDevicePointer(),             // const int2*               __restrict__     exclusionTiles,
            &startTileIndex,                                        // unsigned int                               startTileIndex,
            &numTileIndices,                                        // unsigned long long                         numTileIndices,
            &nb.getInteractingTiles().getDevicePointer(),           // const int*                __restrict__     tiles, 
            &nb.getInteractionCount().getDevicePointer(),           // const unsigned int*       __restrict__     interactionCoun
            cu.getPeriodicBoxSizePointer(),                         // real4                                      periodicBoxSize
            cu.getInvPeriodicBoxSizePointer(),                      // real4                                      invPeriodicBoxS
            cu.getPeriodicBoxVecXPointer(),                         // real4                                      periodicBoxVecX
            cu.getPeriodicBoxVecYPointer(),                         // real4                                      periodicBoxVecY
            cu.getPeriodicBoxVecZPointer(),                         // real4                                      periodicBoxVecZ
            &maxTiles,                                              // unsigned int                               maxTiles, 
            &nb.getBlockCenters().getDevicePointer(),               // const real4*              __restrict__     blockCenter,
            &nb.getBlockBoundingBoxes().getDevicePointer(),         // const real4*              __restrict__     blockSize, 
            &nb.getInteractingAtoms().getDevicePointer(),           // const unsigned int*       __restrict__     interactingAtom
            &maxSinglePairs,                                        // unsigned int                               maxSinglePairs,
            &nb.getSinglePairs().getDevicePointer()                // const int2*               __restrict__     singlePairs
        };
        cout << "1" << endl;
        cu.executeKernel(calcTestForcePBCKernel, args, nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());

        void* argSwitch[] = {
            &cu.getAtomIndexArray().getDevicePointer(),
            &indexAtom.getDevicePointer()
        };
        cout << "2" << endl;
        cu.executeKernel(indexAtomKernel, argSwitch, numParticles);

        void* argsEx[] = {
            &cu.getForce().getDevicePointer(),            //   forceBuffers, 
            &cu.getEnergyBuffer().getDevicePointer(),     //   energyBuffer, 
            &cu.getPosq().getDevicePointer(),             //   posq, 
            &params.getDevicePointer(),                   //   params,
            &cu.getAtomIndexArray().getDevicePointer(),   //   atomIndex,
            &indexAtom.getDevicePointer(),                //   indexAtom,
            &expairidx0.getDevicePointer(),               //   exclusionidx1,
            &expairidx1.getDevicePointer(),               //   exclusionidx2,
            &numexclusions,                               //   numExclusions,
            cu.getPeriodicBoxSizePointer(),               //   periodicBoxSize, 
            cu.getInvPeriodicBoxSizePointer(),            //   invPeriodicBoxSize, 
            cu.getPeriodicBoxVecXPointer(),               //   periodicBoxVecX, 
            cu.getPeriodicBoxVecYPointer(),               //   periodicBoxVecY, 
            cu.getPeriodicBoxVecZPointer()                //   periodicBoxVecZ
        };
        cout << "3" << endl;
        cu.executeKernel(calcExclusionPBCKernel, argsEx, numexclusions);

    } else {
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {
            &cu.getEnergyBuffer().getDevicePointer(), 
            &cu.getPosq().getDevicePointer(), 
            &cu.getForce().getDevicePointer(), 
            &params.getDevicePointer(), 
            &cu.getAtomIndexArray().getDevicePointer(),
            &pairidx0.getDevicePointer(), 
            &pairidx1.getDevicePointer(), 
            &numParticles, &paddedNumAtoms
        };
        cu.executeKernel(calcTestForceNoPBCKernel, args, numParticles*(numParticles-1)/2);

        void* args2[] = {
            &cu.getEnergyBuffer().getDevicePointer(), 
            &cu.getPosq().getDevicePointer(), 
            &cu.getForce().getDevicePointer(), 
            &params.getDevicePointer(), 
            &cu.getAtomIndexArray().getDevicePointer(),
            &expairidx0.getDevicePointer(), 
            &expairidx1.getDevicePointer(), 
            &numexclusions, 
            &numParticles, 
            &paddedNumAtoms
        };
        cu.executeKernel(calcExcludeForceNoPBCKernel, args2, numexclusions);
    }
    return energy;
}


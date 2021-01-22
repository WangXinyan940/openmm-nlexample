#include "CudaCosAccKernels.h"
#include "CudaCosAccKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <iostream>

using namespace CosAccPlugin;
using namespace OpenMM;
using namespace std;

CudaCalcCosAccForceKernel::~CudaCalcCosAccForceKernel() {
}

void CudaCalcCosAccForceKernel::initialize(const System& system, const CosAccForce& force) {
    cu.setAsCurrent();

    int numParticles = system.getNumParticles();
    int elementSize = cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float);

    ifPBC = force.usesPeriodicBoundaryConditions();
    cutoff = force.getCutoffDistance();

    // Inititalize CUDA objects.
    //Vec3 boxVectors[3];
    //map<string, string> defines;
    //system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    //defines["TWOPIOVERLZ"] = cu.doubleToString(6.283185307179586/boxVectors[2][2]);
    //cout << "TWOPIOVERLZ: " << defines["TWOPIOVERLZ"] << endl;
    //CUmodule module = cu.createModule(CudaCosAccKernelSources::cosAccForce, defines);
    //addForcesKernel = cu.getKernel(module, "addForces");
    // if noPBC
    if (!ifPBC){
        map<string, string> defines;
        CUmoudle module = cu.createModule(CudaCoulKernelSources::noPBCForce, defines);
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
        // macro for short-range
        CUmodule PBCModule = cu.createModule(CudaCoulKernelSources::PBCForce, pbcDefines);
        calcTestForcePBCKernel = cu.getKernel(PBCModule, "calcTestForcePBC");
    }
    hasInitializedKernel = true;
}

double CudaCalcCosAccForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    
    int numParticles = cu.getNumAtoms();
    double energy = 0.0;
    if (ifPBC){
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&cu.getEnergyBuffer().getDevicePointer(), &cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &pairidx0.getDevicePointer(), &pairidx1.getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(calcTestForcePBCKernel, args, numParticles);
    } else {
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&cu.getEnergyBuffer().getDevicePointer(), &cu.getPosQ().getDevicePointer(), &cu.getForce().getDevicePointer(), cu.getPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecZPointer(), &numParticles, &paddedNumAtoms}
        cu.executeKernel(calcTestForceNoPBCKernel, args, numParticles*(numParticles-1)/2);
    }
    return energy;
}
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

    charges_cu.initialize(cu, cu.getPaddedNumAtoms(), cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "charges");
    if (cu.getUseDoublePrecision()){
        vector<double> charges;
        charges.resize(numParticles);
        for(int i=0;i<numParticles;i++){
            charges[i] = force.getParticleCharge(i);
        }
        charges_cu.upload(charges);
    } else {
        vector<float> charges;
        charges.resize(cu.getPaddedNumAtoms());
        for(int i=0;i<numParticles;i++){
            charges[i] = force.getParticleCharge(i);
        }
        charges_cu.upload(charges);
    }

    vector<int> exclusions;
    exclusions.resize(2*force.getNumExceptions());
    for(int i=0;i<force.getNumExceptions();i++){
        int p1, p2;
        force.getExceptionParameters(i, p1, p2);
        exclusions.push_back(p1);
        exclusions.push_back(p2);
    }
    exclusions_cu.initialize(cu, 2*force.getNumExceptions(), sizeof(int), "exclusions");

    ifPBC = force.usesPeriodicBoundaryConditions();
    if (ifPBC){
        cutoff = force.getCutoffDistance();
        ewaldTol = force.getEwaldErrorTolerance();
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        alpha = (1.0/cutoff)*sqrt(-log(2.0*ewaldTol));
        one_alpha2 = 1.0 / alpha / alpha;
        kmaxx = 0;
        while (getEwaldParamValue(kmaxx, boxVectors[0][0], alpha) > ewaldTol){
            kmaxx += 1;
        }
        kmaxy = 0;
        while (getEwaldParamValue(kmaxy, boxVectors[1][1], alpha) > ewaldTol){
            kmaxy += 1;
        }
        kmaxz = 0;
        while (getEwaldParamValue(kmaxz, boxVectors[2][2], alpha) > ewaldTol){
            kmaxz += 1;
        }
        if (kmaxx%2 == 0)
            kmaxx += 1;
        if (kmaxy%2 == 0)
            kmaxy += 1;
        if (kmaxz%2 == 0)
            kmaxz += 1;

        // self energy
        ewaldSelfEnergy = 0.0;
        for(int ii=0;ii<numParticles;ii++){
            ewaldSelfEnergy -= ONE_4PI_EPS0 * charges[ii] * charges[ii] * alpha / SQRT_PI;
        }
    }

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
        calcNoPBCEnForcesKernel = cu.getKernel(module, "calcNoPBCEnForces");
        calcNoPBCExclusionsKernel = cu.getKernel(module, "calcNoPBCExclusions");
    } else {
        map<string, string> ewaldDefines;
        map<string, string> shortDefines;
        // macro for ewald
        CUmoudle ewaldModule = cu.createModule(CudaCoulKernelSources::ewaldForce, ewaldDefines);
        calcEwaldRecKernel = cu.getKernel(ewaldModule, "calcEwaldRec");
        // macro for short-range
        CUmodule shortModule = cu.createModule(CudaCoulKernelSources::shortForce, shortDefines);
        calcEwaldRealKernel = cu.getKernel(shortModule, "calcEwaldReal");
        calcEwaldExclusionsKernel = cu.getKernel(shortModule, "calcEwaldExclusions");
    }
    hasInitializedKernel = true;
}

double CudaCalcCosAccForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    
    int numParticles = cu.getNumAtoms();

    if (ifPBC){
        double energy = ewaldSelfEnergy;
    } else {
        double energy = 0.0;
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&charges_cu.getDevicePointer(), &cu.getPosQ().getDevicePointer(), &cu.getForce().getDevicePointer(), &numParticles, &paddedNumAtoms}
        cu.executeKernel(calcNoPBCEnForcesKernel, args, numParticles*(numParticles-1)/2);
    }
    if (includeEnergy) {
        energy += 1.0;
    }
    if (includeForces) {
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&massvec_cu.getDevicePointer(), &cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &accelerate, &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }
    return energy;
}
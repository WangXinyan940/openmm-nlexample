#include "ReferenceTestKernels.h"
#include "TestForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include "ReferenceForce.h"
#include <cmath>

using namespace OpenMM;
using namespace std;
using namespace TestPlugin;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->positions);
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->forces);
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

ReferenceCalcTestForceKernel::~ReferenceCalcTestForceKernel() {
}

void ReferenceCalcTestForceKernel::initialize(const System& system, const TestForce& force) {
    int numParticles = system.getNumParticles();
    charges.resize(numParticles);
    ifPBC = force.usesPeriodicBoundaryConditions();
    cutoff = force.getCutoffDistance();
}

double ReferenceCalcTestForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& forces = extractForces(context);
    Vec3* box = extractBoxVectors(context);
    int numParticles = charges.size();
    double energy = 0.0;    
    double dEdR;
    vector<double> deltaR;
    deltaR.resize(5);
    computeNeighborListVoxelHash(*neighborList, numParticles, pos, vector<set<int>>, box, ifPBC, cutoff, 0.0);
    for(auto& pair : *neighborList){
        int ii = pair.first;
        int jj = pair.second;

        double deltaR[2][ReferenceForce::LastDeltaRIndex];
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
        double r         = deltaR[0][ReferenceForce::RIndex];
        double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);

        if(includeForces){
            double dEdR = - 200.0 * inverseR * inverseR * inverseR;
            for(int kk=0;kk<3;kk++){
                double fconst = dEdR*deltaR[0][kk];
                forces[ii][kk] -= fconst;
                forces[jj][kk] += fconst;
            }
        }

        energy += 100. * inverseR * inverseR;
    }
    return energy;
}
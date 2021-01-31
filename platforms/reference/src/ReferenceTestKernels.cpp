#include "ReferenceTestKernels.h"
#include "TestForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/ReferenceForce.h"
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
    ifPBC = force.usesPeriodicBoundaryConditions();
    cutoff = force.getCutoffDistance();
    exclusions.resize(numParticles);
    for(int ii=0;ii<force.getNumExclusions();ii++){
        int p1, p2;
        force.getExclusionParticles(ii, p1, p2);
        exclusions[p1].insert(p2);
        exclusions[p2].insert(p1);
    }
    if (ifPBC) {
        neighborList = new NeighborList();
    }
    for(int ii=0;ii<numParticles;ii++){
        double prm = force.getParticleParameter(ii);
        params.push_back(prm);
    }
}

double ReferenceCalcTestForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& atomCoordinates = extractPositions(context);
    vector<Vec3>& forces = extractForces(context);
    Vec3* periodicBoxVectors = extractBoxVectors(context);
    int numParticles = atomCoordinates.size();
    double energy = 0.0;    
    double dEdR;
    vector<double> deltaR;
    deltaR.resize(5);
    if (ifPBC){
        computeNeighborListVoxelHash(*neighborList, numParticles, atomCoordinates, exclusions, periodicBoxVectors, ifPBC, cutoff, 0.0);
        for(auto& pair : *neighborList){
            int ii = pair.first;
            int jj = pair.second;
            double p1p2 = params[ii] * params[jj];
            
            double deltaR[2][ReferenceForce::LastDeltaRIndex];
            ReferenceForce::getDeltaRPeriodic(atomCoordinates[ii], atomCoordinates[jj], periodicBoxVectors, deltaR[0]);
            double r         = deltaR[0][ReferenceForce::RIndex];
            double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);

            // cout << ii << " " << jj << " " << params[ii] << " " << params[jj] << " " << r << endl;
            if(includeForces){
                double dEdRdR = - p1p2 * 2 * inverseR * inverseR * inverseR * inverseR;
                for(int kk=0;kk<3;kk++){
                    double fconst = dEdRdR*deltaR[0][kk];
                    forces[ii][kk] += fconst;
                    forces[jj][kk] -= fconst;
                }
            }
            energy += p1p2 * inverseR * inverseR;
        }
    } else {
        for (int ii=0; ii<numParticles; ii++){
            for (int jj=ii+1;jj<numParticles; jj++){
                double p1p2 = params[ii] * params[jj];
                double deltaR[2][ReferenceForce::LastDeltaRIndex];
                ReferenceForce::getDeltaR(atomCoordinates[ii], atomCoordinates[jj], deltaR[0]);
                double r         = deltaR[0][ReferenceForce::RIndex];
                double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);

                if(includeForces){
                    double dEdRdR = - 2 * p1p2 * inverseR * inverseR * inverseR * inverseR;
                    for(int kk=0;kk<3;kk++){
                        double fconst = dEdRdR*deltaR[0][kk];
                        forces[ii][kk] += fconst;
                        forces[jj][kk] -= fconst;
                    }
                }
                energy += p1p2 * inverseR * inverseR;
            }
        }
        for(int p1=0;p1<numParticles;p1++){
            for(auto iter=exclusions[p1].begin(); iter != exclusions[p1].end(); iter++){
                int p2 = *iter;
                double p1p2 = params[p1] * params[p2];
                double deltaR[2][ReferenceForce::LastDeltaRIndex];
                ReferenceForce::getDeltaR(atomCoordinates[p1], atomCoordinates[p2], deltaR[0]);
                double r         = deltaR[0][ReferenceForce::RIndex];
                double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);

                if(includeForces){
                    double dEdRdR = - 2 * p1p2 * inverseR * inverseR * inverseR * inverseR;
                    for(int kk=0;kk<3;kk++){
                        double fconst = dEdRdR*deltaR[0][kk];
                        forces[p1][kk] -= fconst;
                        forces[p2][kk] += fconst;
                    }
                }
                energy -= p1p2 * inverseR * inverseR;
            }
        }
    }
    return energy;
}
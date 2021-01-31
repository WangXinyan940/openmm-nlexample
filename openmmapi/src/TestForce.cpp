#include "TestForce.h"
#include "internal/TestForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>

using namespace TestPlugin;
using namespace OpenMM;
using namespace std;

TestForce::TestForce() {
    cutoffDistance = 1.0;
    ifPBC = false;
}

void TestForce::addParticle(double factor){
    params.push_back(factor);
}

int TestForce::getNumParticles() const {
    return params.size();
}

void TestForce::setParticleParameter(int index, double factor){
    params[index] = factor;
}

double TestForce::getParticleParameter(int index) const {
    return params[index];
}

double TestForce::getCutoffDistance() const {
    return cutoffDistance;
}

void TestForce::setCutoffDistance(double cutoff){
    cutoffDistance = cutoff;
}

bool TestForce::usesPeriodicBoundaryConditions() const {
    return ifPBC;
}

void TestForce::setUsesPeriodicBoundaryConditions(bool ifPeriod){
    ifPBC = ifPeriod;
}

void TestForce::addExclusion(int particle1, int particle2){
    pair<int,int> tmp;
    tmp.first = particle1;
    tmp.second = particle2;
    exclusions.push_back(tmp);
}

void TestForce::getExclusionParticles(int index, int& particle1, int& particle2){
    particle1 = exclusions[index].first;
    particle2 = exclusions[index].second;
}

int TestForce::getNumExclusions() const {
    return exclusions.size();
}

ForceImpl* TestForce::createImpl() const {
    return new TestForceImpl(*this);
}


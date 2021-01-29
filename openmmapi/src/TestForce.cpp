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


ForceImpl* TestForce::createImpl() const {
    return new TestForceImpl(*this);
}


#include "internal/TestForceImpl.h"
#include "TestKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace TestPlugin;
using namespace OpenMM;
using namespace std;

TestForceImpl::TestForceImpl(const TestForce& owner) : owner(owner) {
}

TestForceImpl::~TestForceImpl() {
}

void TestForceImpl::initialize(ContextImpl& context) {

    // Create the kernel.
    kernel = context.getPlatform().createKernel(CalcTestForceKernel::Name(), context);
    kernel.getAs<CalcTestForceKernel>().initialize(context.getSystem(), owner);
}

double TestForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcTestForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> TestForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcTestForceKernel::Name());
    return names;
}
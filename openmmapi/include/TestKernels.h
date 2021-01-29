#ifndef TEST_KERNELS_H_
#define TEST_KERNELS_H_

#include "TestForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace TestPlugin {

/**
 * This kernel is invoked by TestForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcTestForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcTestForce";
    }
    CalcTestForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system         the System this kernel will be applied to
     * @param force          the TestForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const TestForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

} // namespace TestPlugin

#endif /*TEST_KERNELS_H_*/
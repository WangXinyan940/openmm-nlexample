#ifndef REFERENCE_TEST_KERNELS_H_
#define REFERENCE_TEST_KERNELS_H_

#include "TestKernels.h"
#include "openmm/Platform.h"
#include "ReferenceNeighborList.h"
#include <vector>
#include <pair>
#include <iostream>

namespace TestPlugin {

/**
 * This kernel is invoked by TestForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcTestForceKernel : public CalcTestForceKernel {
public:
    ReferenceCalcTestForceKernel(std::string name, const OpenMM::Platform& platform) : CalcTestForceKernel(name, platform) {
    }
    ~ReferenceCalcTestForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system         the System this kernel will be applied to
     * @param force          the TestForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const TestForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    double cutoff;
    bool ifPBC;
    NeighborList* neighborList;
};

} // namespace TestPlugin

#endif /*REFERENCE_TEST_KERNELS_H_*/
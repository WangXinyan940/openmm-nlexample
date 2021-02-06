#ifndef CUDA_TEST_KERNELS_H_
#define CUDA_TEST_KERNELS_H_

#include "TestKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>
#include <string>

namespace TestPlugin {

/**
 * This kernel is invoked by TestForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcTestForceKernel : public CalcTestForceKernel {
public:
    CudaCalcTestForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu) :
            CalcTestForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    }
    ~CudaCalcTestForceKernel();
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
    class ForceInfo;
    bool hasInitializedKernel;
    OpenMM::CudaContext& cu;
    CUfunction calcTestForcePBCKernel;
    CUfunction calcTestForceNoPBCKernel;
    CUfunction calcExcludeForceNoPBCKernel;
    OpenMM::CudaArray pairidx0, pairidx1;
    OpenMM::CudaArray expairidx0, expairidx1;
    OpenMM::CudaArray params;
    OpenMM::CudaArray indexAtom;
    std::vector<std::vector<int>> exclusions;
    int numexclusions;
    double cutoff;
    bool ifPBC;
};

} // namespace TestPlugin

#endif /*CUDA_TEST_KERNELS_H_*/
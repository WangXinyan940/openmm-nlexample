#include <exception>

#include "CudaTestKernelFactory.h"
#include "CudaTestKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <vector>

using namespace TestPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        int argc = 0;
        vector<char**> argv = {NULL};
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaTestKernelFactory* factory = new CudaTestKernelFactory();
        platform.registerKernelFactory(CalcTestForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerTestCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaTestKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcTestForceKernel::Name())
        return new CudaCalcTestForceKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
extern "C" __global__ void calcTestForcePBC(
    real4*              __restrict__     posq,
    unsigned long long* __restrict__     forceBuffers,
    real4               __restrict__     periodicBoxSize, 
    real4               __restrict__     periodicBoxVecX, 
    real4               __restrict__     periodicBoxVecY, 
    real4               __restrict__     periodicBoxVecZ
    int                                  numParticles,
    int                                  paddedNumAtoms
) {
    
}
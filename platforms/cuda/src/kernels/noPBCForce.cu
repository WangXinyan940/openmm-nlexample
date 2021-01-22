extern "C" __global__ void calcTestForceNoPBC(
    mixed*              __restrict__     energyBuffer,
    real4*              __restrict__     posq,
    unsigned long long* __restrict__     forceBuffers,
    int*                __restrict__     pairidx0,
    int*                __restrict__     pairidx1,
    int                                  numParticles,
    int                                  paddedNumAtoms
) {
    int totpair = numParticles * (numParticles - 1) / 2;
    for (int npair = blockIdx.x*blockDim.x+threadIdx.x; npair < totpair; npair += blockDim.x*gridDim.x) {
        int ii = pairidx0[npair];
        int jj = pairidx1[npair];
        real4 delta = make_real4(posq[jj].x-posq[ii].x, posq[jj].y-posq[ii].y, posq[jj].z-posq[ii].z, 0);
        real R2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        real inverseR = RSQRT(R2);
        atomicAdd(&energyBuffer[atom], 100.0 * inverseR * inverseR);
        real4 dEdRdR = - 200.0 * inverseR * inverseR * inverseR * inverseR;
        real4 force = dEdRdR * delta;
        atomicAdd(&forceBuffers[ii], static_cast<unsigned long long>((long long) (-force.x*0x100000000)));
        atomicAdd(&forceBuffers[ii+paddedNumAtoms], static_cast<unsigned long long>((long long) (-force.y*0x100000000)));
        atomicAdd(&forceBuffers[ii+2*paddedNumAtoms], static_cast<unsigned long long>((long long) (-force.z*0x100000000)));
        atomicAdd(&forceBuffers[jj], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[jj+paddedNumAtoms], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[jj+2*paddedNumAtoms], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
    }
}
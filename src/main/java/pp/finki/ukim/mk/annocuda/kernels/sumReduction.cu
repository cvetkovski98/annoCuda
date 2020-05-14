extern "C"
__global__
void sumReduction(double *v, double *v_r) {
    extern __shared__ double partial_sum[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    partial_sum[threadIdx.x] = v[tid];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * threadIdx.x;
        if (index < blockDim.x) {
            partial_sum[index] += partial_sum[index + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}
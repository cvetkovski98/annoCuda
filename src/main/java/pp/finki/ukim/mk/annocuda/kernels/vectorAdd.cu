extern "C"
__global__
void add (int n, float *a, float *b){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] += b[i];
    }
}
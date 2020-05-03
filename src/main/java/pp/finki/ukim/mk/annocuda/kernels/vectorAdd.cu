extern "C"
__global__
void add (long n, double *a, double *b){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] += b[i];
    }
}
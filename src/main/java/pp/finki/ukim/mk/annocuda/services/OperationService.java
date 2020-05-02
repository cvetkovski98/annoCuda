package pp.finki.ukim.mk.annocuda.services;

public interface OperationService {
    void vectorAddGPU(int[] arr1, int[] arr2);

    void vectorAddCPU(int[] arr1, int[] arr2);
}

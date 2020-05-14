package pp.finki.ukim.mk.annocuda.services;

public interface TestService {
    Object cpuVectorSum(double[] vector1, double[] vector2);

    Object cpuMatrixMultiplication(double[][] vector1, double[][] vector2);

    Object gpuVectorSum(double[] vector1, double[] vector2);

    Object gpuMatrixMultiplication(double[][] vector1, double[][] vector2);
}

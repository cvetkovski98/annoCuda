package pp.finki.ukim.mk.annocuda.services.impl;

import org.springframework.stereotype.Service;
import pp.finki.ukim.mk.annocuda.annotations.GPUAction;
import pp.finki.ukim.mk.annocuda.enums.OperationType;
import pp.finki.ukim.mk.annocuda.services.TestService;

import java.util.Arrays;

@Service
public class TestServiceImpl implements TestService {

    @Override
    public Object cpuVectorSum(double[] vector1, double[] vector2) {
        for (int i = 0; i < vector1.length; i++) {
            vector1[i] *= vector2[i];
        }
        return Arrays.stream(vector1).sum();
    }

    @Override
    public Object cpuMatrixMultiplication(double[][] vector1, double[][] vector2) {
        double cell = 0.0;
        double[][] result = new double[vector1.length][vector2[0].length];
        for (int row = 0; row < vector1.length; row++) {
            for (int col = 0; col < vector1.length; col++) {
                for (int i = 0; i < vector2.length; i++) {
                    cell += vector1[row][i] * vector2[i][col];
                }
                result[row][col] = cell;
            }
        }
        return result;
    }

    @Override
    @GPUAction(operationType = OperationType.DOT_PRODUCT)
    public Object gpuVectorSum(double[] vector1, double[] vector2) {
        return null;
    }

    @Override
    @GPUAction(operationType = OperationType.MATRIX_MUL)
    public Object gpuMatrixMultiplication(double[][] vector1, double[][] vector2) {
        return null;
    }
}

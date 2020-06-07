package pp.finki.ukim.mk.annocuda.services.impl;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import org.springframework.stereotype.Service;
import pp.finki.ukim.mk.annocuda.enums.OperationType;
import pp.finki.ukim.mk.annocuda.extentions.utils;
import pp.finki.ukim.mk.annocuda.services.GPUService;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.*;

@Service
public class GPUServiceImpl implements GPUService {

    private final CUcontext context;

    public GPUServiceImpl() {
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        this.context = new CUcontext();
        cuCtxCreate(this.context, 0, device);
    }

    private CUfunction getKernel(String fileName, String functionName) {
        CUmodule cUmodule = new CUmodule();
        String fullPath = this.getModulePath(fileName);
        cuModuleLoad(cUmodule, fullPath);
        CUfunction cFunction = new CUfunction();
        cuModuleGetFunction(cFunction, cUmodule, functionName);
        return cFunction;
    }

    @Override
    public Object execute(Object[] args, OperationType operationType) {
        switch (operationType) {
            case DOT_PRODUCT:
                try {
                    double[] arr1 = (double[]) args[0];
                    double[] arr2 = (double[]) args[1];
                    return vectorAdd(arr1, arr2);
                } catch (Exception e) {
                    e.printStackTrace();
                    return null;
                }
            case MATRIX_MUL:
                try {
                    double[][] arr1 = (double[][]) args[0];
                    double[][] arr2 = (double[][]) args[1];
                    return matrixMul(arr1, arr2);
                } catch (Exception e) {
                    e.printStackTrace();
                    return null;
                }
            case MAP:
                return "MAP";
            case REDUCE:
                try {
                    double[] arr1 = (double[]) args[0];
                    return sumReduce(arr1);
                } catch (Exception e) {
                    e.printStackTrace();
                    return null;
                }
            default:
                return null;
        }
    }

    private double vectorAdd(double[] vector1, double[] vector2) {
        cuCtxSetCurrent(this.context);
        CUfunction vectorAddKernel = this.getKernel("vectorAdd.ptx", "add");

        long numberOfElements = vector1.length;
        long memSize = numberOfElements * Sizeof.DOUBLE;
        CUdeviceptr d_v1 = new CUdeviceptr();
        CUdeviceptr d_v2 = new CUdeviceptr();
        JCuda.cudaMalloc(d_v1, memSize);
        JCuda.cudaMalloc(d_v2, memSize);
        cuMemcpyHtoD(d_v1, Pointer.to(vector1), memSize);
        cuMemcpyHtoD(d_v2, Pointer.to(vector2), memSize);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(new long[]{numberOfElements}),
                Pointer.to(d_v1),
                Pointer.to(d_v2)
        );

        int blockSize = Math.min(vector1.length, 1024);
        int gridSize = (int) Math.ceil((double) numberOfElements / blockSize);

        cuLaunchKernel(vectorAddKernel,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, null,
                kernelParameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(vector1), d_v1, memSize);

        cuMemFree(d_v1);
        cuMemFree(d_v2);
        return Arrays.stream(vector1).sum();
    }

    private double[][] matrixMul(double[][] A, double[][] B) {
        setExceptionsEnabled(true);

        cuCtxSetCurrent(this.context);
        CUfunction matrixMultiplicationKernel = this.getKernel("matrixMul.ptx", "matrixMultiplicationKernel");

        long numberOfElements = A.length * B[0].length;
        long memSize = numberOfElements * Sizeof.DOUBLE;

        CUdeviceptr d_v1 = new CUdeviceptr();
        CUdeviceptr d_v2 = new CUdeviceptr();
        CUdeviceptr d_r = new CUdeviceptr();

        JCuda.cudaMalloc(d_v1, memSize);
        JCuda.cudaMalloc(d_v2, memSize);
        JCuda.cudaMalloc(d_r, memSize);

        double[] vector1 = utils.toRowMajor(A);
        double[] vector2 = utils.toRowMajor(B);
        double[] result = new double[(int) numberOfElements];

        cuMemcpyHtoD(d_v1, Pointer.to(vector1), memSize);
        cuMemcpyHtoD(d_v2, Pointer.to(vector2), memSize);
        cuMemcpyHtoD(d_r, Pointer.to(result), memSize);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_v1),
                Pointer.to(d_v2),
                Pointer.to(d_r),
                Pointer.to(new long[]{A.length})
        );

        int blockSizeX = Math.min(A.length, 32);
        int blockSizeY = Math.min(B[0].length, 32);
        int gridSizeX = (int) Math.ceil((double) A.length / blockSizeX);
        int gridSizeY = (int) Math.ceil((double) B[0].length / blockSizeY);

        cuLaunchKernel(matrixMultiplicationKernel,
                gridSizeX, gridSizeY, 1,
                blockSizeX, blockSizeY, 1,
                0, null,
                kernelParameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(result), d_r, memSize);

        cuMemFree(d_v1);
        cuMemFree(d_v2);
        cuMemFree(d_r);

        return utils.toMatrix(result, A.length, B[0].length);
    }

    private double sumReduce(double[] vector){
        setExceptionsEnabled(true);


        cuCtxSetCurrent(this.context);
        CUfunction sumReductionKernel = this.getKernel("sumReduction.ptx", "sumReduction");

        long numberOfElements = vector.length;
        long memSize = numberOfElements * Sizeof.DOUBLE;

        CUdeviceptr d_in = new CUdeviceptr();
        CUdeviceptr d_r = new CUdeviceptr();

        JCuda.cudaMalloc(d_in, memSize);
        JCuda.cudaMalloc(d_r, memSize);

        cuMemcpyHtoD(d_in, Pointer.to(vector), memSize);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_in),
                Pointer.to(d_r)
        );

        int blockSize = Math.min(vector.length, 1024);
        int gridSize = (int) Math.ceil((double) numberOfElements / blockSize);
        int shMem = blockSize * Sizeof.DOUBLE;

        cuLaunchKernel(sumReductionKernel,
                gridSize, 1, 1,
                blockSize, 1, 1,
                shMem, null,
                kernelParameters, null
        );
        cuCtxSynchronize();

        kernelParameters = Pointer.to(
                Pointer.to(d_r),
                Pointer.to(d_r)
        );

        cuLaunchKernel(sumReductionKernel,
                1, 1, 1,
                gridSize, 1, 1,
                shMem, null,
                kernelParameters, null
        );
        cuCtxSynchronize();

        double[] result = new double[vector.length];

        cuMemcpyDtoH(Pointer.to(result), d_r, memSize);

        cuMemFree(d_in);
        cuMemFree(d_r);

        return result[0];
    }

    private int[] map(Object[] objects) {
        System.out.println("MAP");
        return null;
    }

    private int[] reduce(Object[] objects) {
        System.out.println("REDUCE");
        return null;
    }

    private String getModulePath(String moduleName) {
        return System.getProperty("user.dir") + "/src/main/java/pp/finki/ukim/mk/annocuda/kernels/executables/" + moduleName;
    }


}

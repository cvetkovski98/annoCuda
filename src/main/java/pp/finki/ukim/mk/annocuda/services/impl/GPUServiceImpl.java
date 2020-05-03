package pp.finki.ukim.mk.annocuda.services.impl;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import org.springframework.stereotype.Service;
import pp.finki.ukim.mk.annocuda.enums.OperationType;
import pp.finki.ukim.mk.annocuda.services.GPUService;

import java.io.IOException;
import java.util.Arrays;

import static jcuda.driver.JCudaDriver.*;

@Service
public class GPUServiceImpl implements GPUService {

    @Override
    public CUcontext init() {
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
        return context;
    }

    @Override
    public CUfunction getKernel(String fileName, String functionName) {
        setExceptionsEnabled(true);
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
                }
            case MAP:
                return "MAP";
            case REDUCE:
                return "REDUCE";
            default:
                return null;
        }
    }

    private double vectorAdd(double[] vector1, double[] vector2) throws IOException {
        CUcontext thisContext = this.init();
        System.out.println(this.getModulePath("vectorAdd.ptx"));
        CUfunction vectorAddKernel = this.getKernel("vectorAdd.ptx", "add");
        long numberOfElements = vector1.length;
        long memSize = numberOfElements * Sizeof.DOUBLE;
        CUdeviceptr d_v1 = new CUdeviceptr();
        CUdeviceptr d_v2 = new CUdeviceptr();
        JCuda.cudaMalloc(d_v1, memSize);
        JCuda.cudaMalloc(d_v2, numberOfElements * Sizeof.DOUBLE);
        cuMemcpyHtoD(d_v1, Pointer.to(vector1), memSize);
        cuMemcpyHtoD(d_v2, Pointer.to(vector2), memSize);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(new long[]{numberOfElements}),
                Pointer.to(d_v1),
                Pointer.to(d_v2)
        );

        int blockSize = 512;
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
        cuCtxDestroy(thisContext);
        return Arrays.stream(vector1).sum();
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

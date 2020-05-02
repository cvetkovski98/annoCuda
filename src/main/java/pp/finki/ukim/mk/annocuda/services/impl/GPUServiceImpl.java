package pp.finki.ukim.mk.annocuda.services.impl;

import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import org.springframework.stereotype.Service;
import pp.finki.ukim.mk.annocuda.enums.OperationType;
import pp.finki.ukim.mk.annocuda.services.GPUService;

@Service
public abstract class GPUServiceImpl implements GPUService {
    @Override
    public CUfunction getKernel(String moduleName, String kernelName) {
        CUmodule cUmodule = new CUmodule();
        JCudaDriver.cuModuleLoad(cUmodule, moduleName);
        CUfunction cFunction = new CUfunction();
        JCudaDriver.cuModuleGetFunction(cFunction, cUmodule, kernelName);
        return cFunction;
    }

    @Override
    public Object execute(Object[] args, OperationType operationType) {
        switch (operationType) {
            case VECTOR_ADD:
                return vectorAdd(args);
            case MAP:
                return "MAP";
            case REDUCE:
                return "REDUCE";
            default:
                return null;
        }
    }

    private int[] vectorAdd(Object[] objects) {
        System.out.println("VECTOR_ADD");
        return null;
    }

    private int[] map(Object[] objects) {
        System.out.println("MAP");
        return null;
    }

    private int[] reduce(Object[] objects) {
        System.out.println("REDUCE");
        return null;
    }


}

package pp.finki.ukim.mk.annocuda.services;

import jcuda.driver.CUcontext;
import jcuda.driver.CUfunction;
import pp.finki.ukim.mk.annocuda.enums.OperationType;

public interface GPUService {
    CUcontext init();

    CUfunction getKernel(String fileName, String functionName);

    Object execute(Object[] args, OperationType operationType);
}

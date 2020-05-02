package pp.finki.ukim.mk.annocuda.services;

import jcuda.driver.CUfunction;
import pp.finki.ukim.mk.annocuda.enums.OperationType;

public interface GPUService {
    CUfunction getKernel(String moduleName, String kernelName);

    Object execute(Object[] args, OperationType operationType);
}

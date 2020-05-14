package pp.finki.ukim.mk.annocuda.services;

import pp.finki.ukim.mk.annocuda.enums.OperationType;

public interface GPUService {
    Object execute(Object[] args, OperationType operationType);
}

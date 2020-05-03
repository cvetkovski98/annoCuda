package pp.finki.ukim.mk.annocuda.services.impl;

import org.springframework.stereotype.Service;
import pp.finki.ukim.mk.annocuda.annotations.GPUAction;
import pp.finki.ukim.mk.annocuda.enums.OperationType;
import pp.finki.ukim.mk.annocuda.services.TestGpuService;

@Service
public class TestGpuServiceImpl implements TestGpuService {

    @Override
    @GPUAction(operationType = OperationType.DOT_PRODUCT)
    public Object sumTwoVectors(double[] vector1, double[] vector2) {
        return null;
    }
}

package pp.finki.ukim.mk.annocuda.services.impl;

import org.springframework.stereotype.Service;
import pp.finki.ukim.mk.annocuda.services.TestCpuService;

import java.util.Arrays;

@Service
public class TestCpuServiceImpl implements TestCpuService {

    @Override
    public double vectorDotProduct(double[] vector1, double[] vector2) {
        for (int i = 0; i < vector1.length; i++) {
            vector1[i] *= vector2[i];
        }
        return Arrays.stream(vector1).sum();
    }
}

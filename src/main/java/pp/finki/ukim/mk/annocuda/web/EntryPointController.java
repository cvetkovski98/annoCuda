package pp.finki.ukim.mk.annocuda.web;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import pp.finki.ukim.mk.annocuda.annotations.GPUServiceProvider;
import pp.finki.ukim.mk.annocuda.services.TestService;

import java.io.IOException;
import java.util.Random;

@RestController
@RequestMapping(value = "/")
public class EntryPointController {

    private final TestService testService;
    double[] arr1 = null;
    double[] arr2 = null;

    public EntryPointController(GPUServiceProvider gpuServiceProvider, TestService testService) {
        this.testService = gpuServiceProvider.bindWithGpuAnnotationProcessor(testService, TestService.class);
    }

    @GetMapping("/vectorAddCpu")
    public ResponseEntity<Object[]> vectorAddCpu(@RequestParam long arraySize) throws IOException {
        generateRandomArrays(arraySize);
        long t1 = System.currentTimeMillis();
        Object p = testService.cpuVectorSum(arr1, arr2);
        long t2 = System.currentTimeMillis();
        long time = t2 - t1;
        return ResponseEntity.ok(new Object[]{p, time});
    }

    @GetMapping("/vectorAddGpu")
    public ResponseEntity<Object[]> vectorAddGpu(@RequestParam long arraySize) {
        generateRandomArrays(arraySize);
        long t1 = System.currentTimeMillis();
        Object p = testService.gpuVectorSum(arr1, arr2);
        long t2 = System.currentTimeMillis();
        long time = t2 - t1;
        return ResponseEntity.ok(new Object[]{p, time});
    }

    @GetMapping("/matMulGpu")
    public ResponseEntity<Object[]> matrixGpu() {
        long t1 = System.currentTimeMillis();
        int sz = 32;
        double[][] test1 = new double[sz][sz];
        double[][] test2 = new double[sz][sz];
        for (int i = 0; i < sz; i++) {
            for (int j = 0; j < sz; j++) {
                test1[i][j] = 1;
                test2[i][j] = 1;
            }
        }
        Object p = testService.gpuMatrixMultiplication(test1, test2);
        long t2 = System.currentTimeMillis();
        long time = t2 - t1;
        return ResponseEntity.ok(new Object[]{p, time});
    }

    @GetMapping("/matMulCpu")
    public ResponseEntity<Object[]> matrixCpu() {
        long t1 = System.currentTimeMillis();
        int sz = 32;
        double[][] test1 = new double[sz][sz];
        double[][] test2 = new double[sz][sz];
        for (int i = 0; i < sz; i++) {
            for (int j = 0; j < sz; j++) {
                test1[i][j] = 1;
                test2[i][j] = 1;
            }
        }
        Object p = testService.cpuMatrixMultiplication(test1, test2);
        long t2 = System.currentTimeMillis();
        long time = t2 - t1;
        return ResponseEntity.ok(new Object[]{p, time});
    }

    private void generateRandomArrays(long arraySize) {
        if (arr1 != null && arr2 != null && arraySize == arr1.length) {
            return;
        }
        int rangeMin = 1;
        int rangeMax = 10;
        arr1 = arr2 = new double[Math.toIntExact(arraySize)];
        for (int i = 0; i < arraySize; i++) {
            Random r = new Random();
            double randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
            arr1[i] = arr2[i] = randomValue;
        }
    }
}

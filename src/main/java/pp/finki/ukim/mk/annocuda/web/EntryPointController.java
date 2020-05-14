package pp.finki.ukim.mk.annocuda.web;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import pp.finki.ukim.mk.annocuda.annotations.GPUServiceProvider;
import pp.finki.ukim.mk.annocuda.services.TestService;

import java.util.Random;

@RestController
@RequestMapping(value = "/")
public class EntryPointController {

    private final TestService testService;
    double[] arr1 = null;
    double[] arr2 = null;
    double[][] mat1 = null;
    double[][] mat2 = null;

    public EntryPointController(GPUServiceProvider gpuServiceProvider, TestService testService) {
        this.testService = gpuServiceProvider.bindWithGpuAnnotationProcessor(testService, TestService.class);
    }

    @GetMapping("/vectorAddCpu")
    public ResponseEntity<Object[]> vectorAddCpu(@RequestParam long arraySize) {
        generateRandomArrays(arraySize);
        long t1 = System.currentTimeMillis();
        Object p = testService.cpuVectorSum(arr1, arr2);
        long time = System.currentTimeMillis() - t1;
        return ResponseEntity.ok(new Object[]{time, p});
    }

    @GetMapping("/vectorAddGpu")
    public ResponseEntity<Object[]> vectorAddGpu(@RequestParam long arraySize) {
        generateRandomArrays(arraySize);
        long t1 = System.currentTimeMillis();
        Object p = testService.gpuVectorSum(arr1, arr2);
        long time = System.currentTimeMillis() - t1;
        return ResponseEntity.ok(new Object[]{time, p});
    }

    @GetMapping("/matMulGpu")
    public ResponseEntity<Object[]> matrixGpu(@RequestParam long arraySize) {
        generateRandomMatrices(arraySize);
        long t1 = System.currentTimeMillis();
        Object p = testService.gpuMatrixMultiplication(mat1, mat2);
        long time = System.currentTimeMillis() - t1;
        return ResponseEntity.ok(new Object[]{time});
    }

    @GetMapping("/matMulCpu")
    public ResponseEntity<Object[]> matrixCpu(@RequestParam long arraySize) {
        generateRandomMatrices(arraySize);
        long t1 = System.currentTimeMillis();
        Object p = testService.cpuMatrixMultiplication(mat1, mat2);
        long time = System.currentTimeMillis() - t1;
        return ResponseEntity.ok(new Object[]{time});
    }

    private void generateRandomArrays(long arraySize) {
        if (arr1 != null && arr2 != null && arraySize == arr1.length) {
            return;
        }
        int rangeMin = 1;
        int rangeMax = 10;
        arr1 = new double[Math.toIntExact(arraySize)];
        arr2 = new double[Math.toIntExact(arraySize)];
        for (int i = 0; i < arraySize; i++) {
            Random r = new Random();
            double randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
            arr1[i] = arr2[i] = randomValue;
        }
    }

    private void generateRandomMatrices(long arraySize) {
        if (mat1 != null && mat2 != null && arraySize == mat1.length) {
            return;
        }
        int rangeMin = 1;
        int rangeMax = 10;
        mat1 = new double[Math.toIntExact(arraySize)][Math.toIntExact(arraySize)];
        mat2 = new double[Math.toIntExact(arraySize)][Math.toIntExact(arraySize)];

        for (int i = 0; i < arraySize; i++) {
            for (int j = 0; j < arraySize; j++) {
                Random r = new Random();
                double randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
                mat1[i][j] = mat2[i][j] = randomValue;
            }
        }
    }
}

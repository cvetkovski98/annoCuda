package pp.finki.ukim.mk.annocuda.web;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import pp.finki.ukim.mk.annocuda.annotations.GPUServiceProvider;
import pp.finki.ukim.mk.annocuda.services.TestCpuService;
import pp.finki.ukim.mk.annocuda.services.TestGpuService;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

@RestController
@RequestMapping(value = "/")
public class EntryPointController {

    private final TestGpuService testGpuService;
    private final TestCpuService testCpuService;
    double[] arr1 = null;
    double[] arr2 = null;

    public EntryPointController(GPUServiceProvider gpuServiceProvider,
                                TestGpuService testGpuService,
                                TestCpuService testCpuService) {
        this.testCpuService = testCpuService;
        this.testGpuService = gpuServiceProvider.bindWithGpuAnnotationProcessor(testGpuService, TestGpuService.class);
    }

    @GetMapping("/vectorAddCpu")
    public ResponseEntity<Object[]> vectorAddCpu(@RequestParam long arraySize) throws IOException {
        generateRandomArrays(arraySize);
        long t1 = System.currentTimeMillis();
        Object p = testCpuService.vectorDotProduct(arr1, arr2);
        long t2 = System.currentTimeMillis();
        long time = t2 - t1;
        return ResponseEntity.ok(new Object[]{p, time});
    }

    @GetMapping("/vectorAddGpu")
    public ResponseEntity<Object[]> vectorAddGpu(@RequestParam long arraySize) {
        generateRandomArrays(arraySize);
        long t1 = System.currentTimeMillis();
        System.out.println(testGpuService);
        Object p = testGpuService.sumTwoVectors(arr1, arr2);
        long t2 = System.currentTimeMillis();
        long time = t2 - t1;
        return ResponseEntity.ok(new Object[]{p, time});
    }

    private void generateRandomArrays(long arraySize) {
        if (arr1 != null && arr2 != null && arraySize == arr1.length) {
            return;
        }
        int rangeMin = 1;
        int rangeMax = 2048;
        arr1 = arr2 = new double[Math.toIntExact(arraySize)];
        for (int i = 0; i < arraySize; i++) {
            Random r = new Random();
            double randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
            arr1[i] = arr2[i] = randomValue;
        }
    }
}

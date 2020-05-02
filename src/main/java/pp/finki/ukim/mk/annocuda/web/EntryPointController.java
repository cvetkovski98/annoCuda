package pp.finki.ukim.mk.annocuda.web;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import pp.finki.ukim.mk.annocuda.services.GPUServiceProvider;
import pp.finki.ukim.mk.annocuda.enums.OperationType;
import pp.finki.ukim.mk.annocuda.services.GPUService;
import pp.finki.ukim.mk.annocuda.services.impl.GPUServiceImpl;

@RestController
@RequestMapping(value = "/")
public class EntryPointController {

    private final GPUService gpuService;

    public EntryPointController(GPUServiceProvider gpuServiceProvider) {
        this.gpuService = gpuServiceProvider.getGpuServiceMethodHandler();
    }

    @GetMapping("/vectorAddCpu")
    public ResponseEntity<Object> vectorAddCpu() {
        int[] arr1 = new int[1000000];
        int[] arr2 = new int[1000000];
        for (int i = 0; i < 1000000; i++) {
            arr1[i] = arr2[i] = 1;
        }
        long t1 = System.currentTimeMillis();
        Object a = gpuService.execute(new int[][]{arr1, arr2}, OperationType.VECTOR_ADD);
        long t2 = System.currentTimeMillis();
        long time = t2 - t1;
        return ResponseEntity.ok(a);
    }

    @GetMapping("/vectorAddGpu")
    public ResponseEntity<Long> vectorAddGpu() {
        int[] arr1 = new int[1000000];
        int[] arr2 = new int[1000000];
        for (int i = 0; i < 1000000; i++) {
            arr1[i] = arr2[i] = 1;
        }
        long t1 = System.currentTimeMillis();
        gpuService.execute(new int[][]{arr1, arr2}, OperationType.MAP);
        long t2 = System.currentTimeMillis();
        long time = t2 - t1;
        return ResponseEntity.ok(time);
    }
}

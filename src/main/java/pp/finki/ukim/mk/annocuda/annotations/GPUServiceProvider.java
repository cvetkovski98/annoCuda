package pp.finki.ukim.mk.annocuda.annotations;

import org.springframework.stereotype.Component;
import pp.finki.ukim.mk.annocuda.services.GPUService;

import java.lang.reflect.Proxy;

@Component
public class GPUServiceProvider {

    private final GPUService gpuService;

    public GPUServiceProvider(GPUService gpuService) {
        this.gpuService = gpuService;
    }

    @SuppressWarnings("unchecked")
    public <T> T bindWithGpuAnnotationProcessor(T _object, Class<?> _interface) {
        return (T) Proxy.newProxyInstance(
                _object.getClass().getClassLoader(),
                new Class[]{_interface},
                new AnnotationProcessor(_object, gpuService)
        );
    }
}

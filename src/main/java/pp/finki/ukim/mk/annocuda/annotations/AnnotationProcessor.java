package pp.finki.ukim.mk.annocuda.annotations;

import org.springframework.stereotype.Component;
import pp.finki.ukim.mk.annocuda.services.GPUService;
import pp.finki.ukim.mk.annocuda.services.TestGpuService;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

@Component
public class AnnotationProcessor implements InvocationHandler {
    private final TestGpuService object;
    private final GPUService gpuService;

    public AnnotationProcessor(TestGpuService object, GPUService gpuService) {
        this.object = object;
        this.gpuService = gpuService;
    }

    @Override
    public Object invoke(Object o, Method method, Object[] objects) throws Throwable {
        Method m = object.getClass().getMethod(method.getName(), method.getParameterTypes());
        if (m.isAnnotationPresent(GPUAction.class)) {
            System.out.println("Executing " + m + " on GPU");
            return gpuService.execute(objects, m.getAnnotation(GPUAction.class).operationType());
        } else {
            System.out.println("Executing " + m + " on CPU");
            return method.invoke(this.object, objects);
        }
    }
}

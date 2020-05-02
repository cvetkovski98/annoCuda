package pp.finki.ukim.mk.annocuda.annotations;

import org.springframework.stereotype.Component;
import pp.finki.ukim.mk.annocuda.services.GPUService;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

@Component
public class AnnotationProcessor implements InvocationHandler {

    private final GPUService gpuService;

    public AnnotationProcessor(GPUService gpuService) {
        this.gpuService = gpuService;
    }

    @Override
    public Object invoke(Object o, Method method, Object[] objects) throws Throwable {
        if (gpuService.getClass().getMethod(method.getName(), method.getParameterTypes()).isAnnotationPresent(GPUAction.class)) {
            return gpuService.execute(objects, method.getAnnotation(GPUAction.class).operationType());
        }
        return method.invoke(gpuService, objects);
    }
}

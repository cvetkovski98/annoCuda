package pp.finki.ukim.mk.annocuda.annotations;

import pp.finki.ukim.mk.annocuda.services.GPUService;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

public class AnnotationProcessor implements InvocationHandler {
    private final Object object;
    private final GPUService gpuService;

    public AnnotationProcessor(Object object, GPUService gpuService) {
        this.object = object;
        this.gpuService = gpuService;
    }

    @Override
    public Object invoke(Object o, Method method, Object[] objects) throws Throwable {
        Method m = object.getClass().getMethod(method.getName(), method.getParameterTypes());
        if (m.isAnnotationPresent(GPUAction.class)) {
            return gpuService.execute(objects, m.getAnnotation(GPUAction.class).operationType());
        } else {
            return method.invoke(this.object, objects);
        }
    }
}

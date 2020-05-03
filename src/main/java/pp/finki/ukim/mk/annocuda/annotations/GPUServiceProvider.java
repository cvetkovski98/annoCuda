package pp.finki.ukim.mk.annocuda.annotations;

import org.springframework.stereotype.Component;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;

@Component
public class GPUServiceProvider {

    private final InvocationHandler invocationHandler;

    public GPUServiceProvider(InvocationHandler invocationHandler) {
        this.invocationHandler = invocationHandler;
    }

    @SuppressWarnings("unchecked")
    public <T> T bindWithGpuAnnotationProcessor(T _object, Class<?> _interface) {
        return (T) Proxy.newProxyInstance(
                _object.getClass().getClassLoader(),
                new Class[]{_interface},
                this.invocationHandler
        );
    }
}

package pp.finki.ukim.mk.annocuda.services;

import org.springframework.stereotype.Component;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;

@Component
public class GPUServiceProvider {
    private final InvocationHandler invocationHandler;

    public GPUServiceProvider(InvocationHandler invocationHandler) {
        this.invocationHandler = invocationHandler;
    }

    public GPUService getGpuServiceMethodHandler() {
        return (GPUService) Proxy.newProxyInstance(
                GPUService.class.getClassLoader(),
                new Class[]{GPUService.class},
                this.invocationHandler
        );
    }
}

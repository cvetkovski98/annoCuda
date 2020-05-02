package pp.finki.ukim.mk.annocuda.annotations;

import pp.finki.ukim.mk.annocuda.enums.OperationType;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface GPUAction {
    OperationType operationType();
}

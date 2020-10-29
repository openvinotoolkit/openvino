package org.intel.openvino;

public class ExecutableNetwork extends IEWrapper {

    protected ExecutableNetwork(long addr) {
        super(addr);
    }

    public InferRequest CreateInferRequest() {
        return new InferRequest(CreateInferRequest(nativeObj));
    }

    public Parameter GetMetric(String name) {
        return new Parameter(GetMetric(nativeObj, name));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long CreateInferRequest(long addr);

    private static native long GetMetric(long addr, String name);

    @Override
    protected native void delete(long nativeObj);
}

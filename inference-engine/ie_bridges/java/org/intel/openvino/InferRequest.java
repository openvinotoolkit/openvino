package org.intel.openvino;

import java.util.Map;

public class InferRequest extends IEWrapper {

    protected InferRequest(long addr) {
        super(addr);
    }

    public void Infer() {
        Infer(nativeObj);
    }

    public Blob GetBlob(String name) {
        return new Blob(GetBlob(nativeObj, name));
    }

    public void SetBlob(String name, Blob blob) {
        SetBlob(nativeObj, name, blob.getNativeObjAddr());
    }

    public void StartAsync() {
        StartAsync(nativeObj);
    }

    public StatusCode Wait(WaitMode waitMode) {
        return StatusCode.valueOf(Wait(nativeObj, waitMode.getValue()));
    }

    public void SetCompletionCallback(Runnable runnable) {
        SetCompletionCallback(nativeObj, runnable);
    }

    public Map<String, InferenceEngineProfileInfo> GetPerformanceCounts() {
        return GetPerformanceCounts(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native void Infer(long addr);

    private static native void StartAsync(long addr);

    private static native int Wait(long addr, int wait_mode);

    private static native void SetCompletionCallback(long addr, Runnable runnable);

    private static native long GetBlob(long addr, String name);

    private static native void SetBlob(long addr, String name, long blobAddr);

    private static native Map<String, InferenceEngineProfileInfo> GetPerformanceCounts(long addr);

    @Override
    protected native void delete(long nativeObj);
}

package org.intel.openvino;

public class LockedMemory extends IEWrapper {

    protected LockedMemory(long addr) {
        super(addr);
    }

    public void get(float[] res) {
        asFloat(nativeObj, res);
    }

    public void get(byte[] res) {
        asByte(nativeObj, res);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native void asByte(long addr, byte[] res);

    private static native void asFloat(long addr, float[] res);

    @Override
    protected native void delete(long nativeObj);
}

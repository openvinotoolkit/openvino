package org.intel.openvino;

public class IEWrapper {
    protected final long nativeObj;

    protected IEWrapper(long addr) {
        nativeObj = addr;
    }

    protected long getNativeObjAddr() {
        return nativeObj;
    }

    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    protected native void delete(long nativeObj);
}

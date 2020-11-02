package org.intel.openvino;

public class TensorDesc extends IEWrapper {

    public TensorDesc(long addr) {
        super(addr);
    }

    public TensorDesc(Precision precision, int[] dims, Layout layout) {
        super(GetTensorDesc(precision.getValue(), dims, layout.getValue()));
    }

    public int[] getDims() {
        return GetDims(nativeObj);
    }

    public Layout getLayout() {
        return Layout.valueOf(getLayout(nativeObj));
    }

    public Precision getPrecision() {
        return Precision.valueOf(getPrecision(nativeObj));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long GetTensorDesc(int precision, int[] dims, int layout);

    private native int[] GetDims(long addr);

    private static native int getLayout(long addr);

    private static native int getPrecision(long addr);

    @Override
    protected native void delete(long nativeObj);
}

package org.intel.openvino;

public class InputInfo extends IEWrapper {

    public InputInfo(long addr) {
        super(addr);
    }

    public PreProcessInfo getPreProcess() {
        return new PreProcessInfo(getPreProcess(nativeObj));
    }

    public void setLayout(Layout layout) {
        SetLayout(nativeObj, layout.getValue());
    }

    public Layout getLayout() {
        return Layout.valueOf(getLayout(nativeObj));
    }

    public void setPrecision(Precision precision) {
        SetPrecision(nativeObj, precision.getValue());
    }

    public Precision getPrecision() {
        return Precision.valueOf(getPrecision(nativeObj));
    }

    public TensorDesc getTensorDesc() {
        return new TensorDesc(GetTensorDesc(nativeObj));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long getPreProcess(long addr);

    private static native void SetLayout(long addr, int layout);

    private static native int getLayout(long addr);

    private static native void SetPrecision(long addr, int precision);

    private static native int getPrecision(long addr);

    private native long GetTensorDesc(long addr);
}

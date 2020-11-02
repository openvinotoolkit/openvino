package org.intel.openvino;

import java.util.Map;

public class CNNNetwork extends IEWrapper {

    protected CNNNetwork(long addr) {
        super(addr);
    }

    public String getName() {
        return getName(nativeObj);
    }

    public int getBatchSize() {
        return getBatchSize(nativeObj);
    }

    public Map<String, Data> getOutputsInfo() {
        return GetOutputsInfo(nativeObj);
    }

    public Map<String, InputInfo> getInputsInfo() {
        return GetInputsInfo(nativeObj);
    }

    public void reshape(Map<String, int[]> inputShapes) {
        reshape(nativeObj, inputShapes);
    }

    public Map<String, int[]> getInputShapes() {
        return getInputShapes(nativeObj);
    }

    public void addOutput(String layerName, int outputIndex) {
        addOutput(nativeObj, layerName, outputIndex);
    }

    public void addOutput(String layerName) {
        addOutput1(nativeObj, layerName);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native String getName(long addr);

    private static native int getBatchSize(long addr);

    private static native Map<String, InputInfo> GetInputsInfo(long addr);

    private static native Map<String, Data> GetOutputsInfo(long addr);

    private static native void reshape(long addr, Map<String, int[]> inputShapes);

    private static native Map<String, int[]> getInputShapes(long addr);

    private static native void addOutput(long addr, String layerName, int outputIndex);

    private static native void addOutput1(long addr, String layerName);

    @Override
    protected native void delete(long nativeObj);
}

package org.intel.openvino;

import java.util.Map;

public class IECore extends IEWrapper {
    public static final String NATIVE_LIBRARY_NAME = "inference_engine_java_api";

    public IECore() {
        super(GetCore());
    }

    public IECore(String xmlConfigFile) {
        super(GetCore_1(xmlConfigFile));
    }

    public CNNNetwork ReadNetwork(final String modelPath, final String weightPath) {
        return new CNNNetwork(ReadNetwork1(nativeObj, modelPath, weightPath));
    }

    public CNNNetwork ReadNetwork(final String modelFileName) {
        return new CNNNetwork(ReadNetwork(nativeObj, modelFileName));
    }

    public ExecutableNetwork LoadNetwork(CNNNetwork net, final String device) {
        return new ExecutableNetwork(LoadNetwork(nativeObj, net.getNativeObjAddr(), device));
    }

    public ExecutableNetwork LoadNetwork(
            CNNNetwork net, final String device, final Map<String, String> config) {
        long network = LoadNetwork1(nativeObj, net.getNativeObjAddr(), device, config);
        return new ExecutableNetwork(network);
    }

    public void RegisterPlugin(String pluginName, String deviceName) {
        RegisterPlugin(nativeObj, pluginName, deviceName);
    }

    public void RegisterPlugin(String xmlConfigFile) {
        RegisterPlugins(nativeObj, xmlConfigFile);
    }

    public void UnregisterPlugin(String deviceName) {
        UnregisterPlugin(nativeObj, deviceName);
    }

    public void AddExtension(String extension) {
        AddExtension(nativeObj, extension);
    }

    public void AddExtension(String extension, String deviceName) {
        AddExtension1(nativeObj, extension, deviceName);
    }

    public void SetConfig(Map<String, String> config, String deviceName) {
        SetConfig(nativeObj, config, deviceName);
    }

    public void SetConfig(Map<String, String> config) {
        SetConfig1(nativeObj, config);
    }

    public Parameter GetConfig(String deviceName, String name) {
        return new Parameter(GetConfig(nativeObj, deviceName, name));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long ReadNetwork(long core, final String modelFileName);

    private static native long ReadNetwork1(
            long core, final String modelPath, final String weightPath);

    private static native long LoadNetwork(long core, long net, final String device);

    private static native long LoadNetwork1(
            long core, long net, final String device, final Map<String, String> config);

    private static native void RegisterPlugin(long core, String pluginName, String deviceName);

    private static native void RegisterPlugins(long core, String xmlConfigFile);

    private static native void UnregisterPlugin(long core, String deviceName);

    private static native void AddExtension(long core, String extension);

    private static native void AddExtension1(long core, String extension, String deviceName);

    private static native void SetConfig(long core, Map<String, String> config, String deviceName);

    private static native void SetConfig1(long core, Map<String, String> config);

    private static native long GetConfig(long core, String deviceName, String name);

    private static native long GetCore();

    private static native long GetCore_1(String xmlConfigFile);

    @Override
    protected native void delete(long nativeObj);
}

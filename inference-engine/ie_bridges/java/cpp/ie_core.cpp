#include <jni.h>   // JNI header provided by JDK
#include <inference_engine.hpp>

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_GetCore(JNIEnv *env, jobject obj)
{
    static const char method_name[] = "GetCore";
    try
    {
        Core *core = new Core();
        return (jlong)core;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_GetCore_1(JNIEnv *env, jobject obj, jstring xmlConfigFile)
{
    static const char method_name[] = "GetCore_1";
    try
    {
        std::string n_xml = jstringToString(env, xmlConfigFile);
        Core *core = new Core(n_xml);
        return (jlong)core;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_ReadNetwork1(JNIEnv *env, jobject obj, jlong coreAddr, jstring xml, jstring bin)
{
    static const char method_name[] = "ReadNetwork1";
    try
    {
        std::string n_xml = jstringToString(env, xml);
        std::string n_bin = jstringToString(env, bin);

        Core *core = (Core *)coreAddr;

        CNNNetwork *network = new CNNNetwork();
        *network = core->ReadNetwork(n_xml, n_bin);

        return (jlong)network;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_ReadNetwork(JNIEnv *env, jobject obj, jlong coreAddr, jstring xml)
{
    static const char method_name[] = "ReadNetwork";
    try
    {
        std::string n_xml = jstringToString(env, xml);

        Core *core = (Core *)coreAddr;

        CNNNetwork *network = new CNNNetwork();
        *network = core->ReadNetwork(n_xml);

        return (jlong)network;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_LoadNetwork(JNIEnv *env, jobject obj, jlong coreAddr, jlong netAddr, jstring device)
{
    static const char method_name[] = "LoadNetwork";
    try
    {
        std::string n_device = jstringToString(env, device);

        Core *core = (Core *)coreAddr;

        CNNNetwork *network = (CNNNetwork *)netAddr;

        ExecutableNetwork *executable_network = new ExecutableNetwork();
        *executable_network = core->LoadNetwork(*network, n_device);

        return (jlong)executable_network;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_LoadNetwork1(JNIEnv *env, jobject obj, jlong coreAddr, jlong netAddr, jstring device, jobject config)
{
    static const char method_name[] = "LoadNetwork1";
    try
    {
        std::string n_device = jstringToString(env, device);

        Core *core = (Core *)coreAddr;
        CNNNetwork *network = (CNNNetwork *)netAddr;

        ExecutableNetwork *executable_network = new ExecutableNetwork();
        *executable_network = core->LoadNetwork(*network, n_device, javaMapToMap(env, config));

        return (jlong)executable_network;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_RegisterPlugin(JNIEnv *env, jobject obj, jlong addr, jstring pluginName, jstring deviceName) 
{
    static const char method_name[] = "RegisterPlugin";
    try
    {
        const std::string n_plugin = jstringToString(env, pluginName);
        const std::string n_device = jstringToString(env, deviceName);

        Core *core = (Core *) addr;
        core->RegisterPlugin(n_plugin, n_device);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_UnregisterPlugin(JNIEnv *env, jobject obj, jlong addr, jstring deviceName) 
{
    static const char method_name[] = "UnregisterPlugin";
    try
    {
        const std::string n_device = jstringToString(env, deviceName);

        Core *core = (Core *) addr;
        core->UnregisterPlugin(n_device);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_RegisterPlugins(JNIEnv *env, jobject obj, jlong addr, jstring xmlConfigFile) 
{
    static const char method_name[] = "RegisterPlugins";
    try
    {
        const std::string n_xml = jstringToString(env, xmlConfigFile);

        Core *core = (Core *) addr;
        core->RegisterPlugins(n_xml);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_AddExtension(JNIEnv *env, jobject obj, jlong addr, jstring extension) 
{
    static const char method_name[] = "AddExtension";
    try
    {
        const std::string n_extension = jstringToString(env, extension);

        const InferenceEngine::IExtensionPtr extension =
                        InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(n_extension);

        Core *core = (Core *) addr;
        core->AddExtension(extension);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_AddExtension1(JNIEnv *env, jobject obj, jlong addr, jstring extension, jstring deviceName) 
{
    static const char method_name[] = "AddExtension";
    try
    {
        const std::string n_extension = jstringToString(env, extension);
        const std::string n_device = jstringToString(env, deviceName);

        InferenceEngine::IExtensionPtr extension =
                        InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(n_extension);

        Core *core = (Core *) addr;
        core->AddExtension(extension, n_device);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_SetConfig(JNIEnv *env, jobject obj, jlong addr, jobject config, jstring deviceName)
{
    static const char method_name[] = "SetConfig";
    try
    {   
        Core *core = (Core *) addr;
        core->SetConfig(javaMapToMap(env, config), jstringToString(env, deviceName));
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_SetConfig1(JNIEnv *env, jobject obj, jlong addr, jobject config)
{
    static const char method_name[] = "SetConfig";
    try
    {   
        Core *core = (Core *) addr;
        core->SetConfig(javaMapToMap(env, config));
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}
JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_GetConfig(JNIEnv *env, jobject obj, jlong addr, jstring deviceName, jstring name)
{
    static const char method_name[] = "GetConfig";
    try
    {   
        Core *core = (Core *) addr;
        Parameter *parameter = new Parameter();
        *parameter = core->GetConfig(jstringToString(env, deviceName), jstringToString(env, name));
        
        return (jlong) parameter;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_delete(JNIEnv *, jobject, jlong addr)
{
    Core *core = (Core *)addr;
    delete core;
}

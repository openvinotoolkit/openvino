#include <inference_engine.hpp>

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

JNIEXPORT jstring JNICALL Java_org_intel_openvino_CNNNetwork_getName(JNIEnv *env, jobject, jlong addr) 
{
    static const char method_name[] = "getName";
    try
    {
        CNNNetwork *network = (CNNNetwork *)addr;
        return env->NewStringUTF(network->getName().c_str()); 
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

JNIEXPORT jint JNICALL Java_org_intel_openvino_CNNNetwork_getBatchSize(JNIEnv *env, jobject, jlong addr)
{
    static const char method_name[] = "getBatchSize";
    try
    {
        CNNNetwork *network = (CNNNetwork *)addr;
        return static_cast<jint>(network->getBatchSize());
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

JNIEXPORT jobject JNICALL Java_org_intel_openvino_CNNNetwork_GetInputsInfo(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "GetInputsInfo";
    try
    {
        CNNNetwork *network = (CNNNetwork *)addr;
        const InputsDataMap &inputs_map = network->getInputsInfo();
 
        jclass hashMapClass = env->FindClass("java/util/HashMap");
        jmethodID hashMapInit = env->GetMethodID(hashMapClass, "<init>", "()V");
        jobject hashMapObj = env->NewObject(hashMapClass, hashMapInit);
        jmethodID hashMapPut = env->GetMethodID(hashMapClass, "put",
                                                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");

        jclass inputInfoClass = env->FindClass("org/intel/openvino/InputInfo");
        jmethodID inputInfoConstructor = env->GetMethodID(inputInfoClass,"<init>","(J)V");

        for (const auto &item : inputs_map) {
            jobject inputInfoObj = env->NewObject(inputInfoClass, inputInfoConstructor, (jlong)(item.second.get()));
            env->CallObjectMethod(hashMapObj, hashMapPut, env->NewStringUTF(item.first.c_str()), inputInfoObj);
        }

        env->PopLocalFrame(hashMapObj);

        return hashMapObj;
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

JNIEXPORT jobject JNICALL Java_org_intel_openvino_CNNNetwork_GetOutputsInfo(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "GetOutputsInfo";
    try
    {
        CNNNetwork *network = (CNNNetwork *)addr;
        const OutputsDataMap &outputs_map = network->getOutputsInfo();

        jclass hashMapClass = env->FindClass("java/util/HashMap");
        jmethodID hashMapInit = env->GetMethodID(hashMapClass, "<init>", "()V");
        jobject hashMapObj = env->NewObject(hashMapClass, hashMapInit);
        jmethodID hashMapPut = env->GetMethodID(hashMapClass, "put",
                                                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");

        jclass outputInfoClass = env->FindClass("org/intel/openvino/Data");
        jmethodID outputInfoConstructor = env->GetMethodID(outputInfoClass,"<init>","(J)V");

        for (const auto &item : outputs_map) {
            jobject outputInfoObj = env->NewObject(outputInfoClass, outputInfoConstructor, (jlong)(item.second.get()));
            env->CallObjectMethod(hashMapObj, hashMapPut, env->NewStringUTF(item.first.c_str()), outputInfoObj);
        }

        env->PopLocalFrame(hashMapObj);

        return hashMapObj;
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

JNIEXPORT void JNICALL Java_org_intel_openvino_CNNNetwork_reshape(JNIEnv *env, jobject obj, jlong addr, jobject input)
{
    static const char method_name[] = "reshape";
    try
    {
        CNNNetwork *network = (CNNNetwork *)addr;
        network->reshape(javaMapToMap_1(env, input));
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

JNIEXPORT jobject JNICALL Java_org_intel_openvino_CNNNetwork_getInputShapes(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "getInputShapes";
    try
    {
        CNNNetwork *network = (CNNNetwork *)addr;
        std::map<std::string, std::vector<size_t>> shapes = network->getInputShapes();

        jclass hashMapClass = env->FindClass("java/util/HashMap");
        jmethodID hashMapInit = env->GetMethodID(hashMapClass, "<init>", "()V");
        jobject hashMapObj = env->NewObject(hashMapClass, hashMapInit);
        jmethodID hashMapPut = env->GetMethodID(hashMapClass, "put",
                                                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");

        for (const auto &item : shapes) {
            jintArray result = env->NewIntArray(item.second.size());

            jint *arr = env->GetIntArrayElements(result, nullptr);
            for (int i = 0; i < item.second.size(); ++i)
                arr[i] = item.second[i];

            env->ReleaseIntArrayElements(result, arr, 0);
            env->CallObjectMethod(hashMapObj, hashMapPut, env->NewStringUTF(item.first.c_str()), result);
        }

        env->PopLocalFrame(hashMapObj);

        return hashMapObj;
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

JNIEXPORT void JNICALL Java_org_intel_openvino_CNNNetwork_addOutput(JNIEnv *env, jobject obj, jlong addr, jstring layerName, jint outputIndex){
    static const char method_name[] = "addOutput";
    try
    {
        CNNNetwork *network = (CNNNetwork *)addr;
        
        std::string c_outputName = jstringToString(env, layerName);
        size_t c_outputIndex = static_cast<size_t>(outputIndex);

        network->addOutput(c_outputName, c_outputIndex);
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

JNIEXPORT void JNICALL Java_org_intel_openvino_CNNNetwork_addOutput1(JNIEnv *env, jobject obj, jlong addr, jstring layerName){
    static const char method_name[] = "addOutput";
    try
    {
        CNNNetwork *network = (CNNNetwork *)addr;
        
        std::string c_outputName = jstringToString(env, layerName);

        network->addOutput(c_outputName);
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

JNIEXPORT void JNICALL Java_org_intel_openvino_CNNNetwork_delete(JNIEnv *env, jobject obj, jlong addr)
{
    CNNNetwork *network = (CNNNetwork *)addr;
    delete network;
}

#include <inference_engine.hpp>
#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

JNIEXPORT jstring JNICALL Java_org_intel_openvino_Parameter_asString(JNIEnv *env, jobject obj, jlong addr) 
{
    static const char method_name[] = "asString";
    try
    {
        Parameter *parameter = (Parameter *)addr;
        return env->NewStringUTF(parameter->as<std::string>().c_str());
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

JNIEXPORT jint JNICALL Java_org_intel_openvino_Parameter_asInt(JNIEnv *env, jobject obj, jlong addr) 
{
    static const char method_name[] = "asInt";
    try
    {
        Parameter *parameter = (Parameter *)addr;
        return static_cast<jint>(parameter->as<unsigned int>());
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

JNIEXPORT void JNICALL Java_org_intel_openvino_Parameter_delete(JNIEnv *, jobject, jlong addr)
{
    Parameter *parameter = (Parameter *)addr;
    delete parameter;
}

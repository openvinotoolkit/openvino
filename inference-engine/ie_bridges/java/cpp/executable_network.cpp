#include <inference_engine.hpp>

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_ExecutableNetwork_CreateInferRequest(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "CreateInferRequest";
    try
    {
        ExecutableNetwork *executable_network = (ExecutableNetwork *)addr;

        InferRequest *infer_request = new InferRequest();
        *infer_request = executable_network->CreateInferRequest();

        return (jlong)infer_request;
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

JNIEXPORT jlong JNICALL Java_org_intel_openvino_ExecutableNetwork_GetMetric(JNIEnv *env, jobject obj, jlong addr, jstring name)
{
    static const char method_name[] = "GetMetric";
    try
    {
        ExecutableNetwork *executable_network = (ExecutableNetwork *)addr;

        Parameter *parameter = new Parameter();
        *parameter = executable_network->GetMetric(jstringToString(env, name));

        return (jlong)parameter;
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

JNIEXPORT void JNICALL Java_org_intel_openvino_ExecutableNetwork_delete(JNIEnv *, jobject, jlong addr)
{
    ExecutableNetwork *executable_network = (ExecutableNetwork *)addr;
    delete executable_network;
}

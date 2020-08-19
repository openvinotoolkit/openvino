#include <inference_engine.hpp>
#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

JNIEXPORT void JNICALL Java_org_intel_openvino_LockedMemory_asFloat(JNIEnv *env, jobject obj, jlong addr, jfloatArray res)
{
    static const char method_name[] = "asFloat";
    try
    {
        LockedMemory<const void> *lmem = (LockedMemory<const void> *) addr;
        const float *buffer = lmem->as<const float *>();

        const jsize size = env->GetArrayLength(res);
        env->SetFloatArrayRegion(res, 0, size, buffer);
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

JNIEXPORT void JNICALL Java_org_intel_openvino_LockedMemory_asByte(JNIEnv *env, jobject obj, jlong addr, jbyteArray res)
{
    static const char method_name[] = "asByte";
    try
    {
        LockedMemory<const void> *lmem = (LockedMemory<const void> *) addr;
        const uint8_t *buffer = lmem->as<const uint8_t *>();

        const jsize size = env->GetArrayLength(res);
        env->SetByteArrayRegion(res, 0, size, reinterpret_cast<const jbyte*>(buffer));
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

JNIEXPORT void JNICALL Java_org_intel_openvino_LockedMemory_delete(JNIEnv *, jobject, jlong addr)
{
    LockedMemory<const void> *lmem = (LockedMemory<const void> *) addr;
    delete lmem;
}

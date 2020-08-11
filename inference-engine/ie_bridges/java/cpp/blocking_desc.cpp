#include <inference_engine.hpp>

#include "openvino_java.hpp"
#include "enum_mapping.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_BlockingDesc_GetBlockingDesc(JNIEnv *env, jobject obj)
{
    static const char method_name[] = "GetBlockinDesc";
    try
    {
        BlockingDesc *bDesc = new BlockingDesc();
        return (jlong)bDesc;

    } catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_BlockingDesc_GetBlockingDesc1(JNIEnv *env, jobject obj, jintArray dims, jint layout)
{
    static const char method_name[] = "GetBlockingDesc1";
    try
    {
        BlockingDesc *bDesc = new BlockingDesc(jintArrayToVector(env, dims), layout_map.at(layout));
        return (jlong)bDesc;

    } catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_BlockingDesc_delete(JNIEnv *, jobject, jlong addr)
{
    BlockingDesc *bDesc = (BlockingDesc *)addr;
    delete bDesc;
}

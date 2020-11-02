#include <inference_engine.hpp>

#include "openvino_java.hpp"
#include "enum_mapping.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessInfo_SetResizeAlgorithm(JNIEnv *env, jobject obj, jlong addr, jint resizeAlgorithm)
{
    static const char method_name[] = "SetResizeAlgorithm";

    try
    {
        PreProcessInfo *pre_process_info = (PreProcessInfo*)addr;
        auto it = resize_alg_map.find(resizeAlgorithm);

        if (it == resize_alg_map.end())
            throw std::runtime_error("Wrong resize algorithm number!");

        pre_process_info->setResizeAlgorithm(it->second);

    } catch (const std::exception &e){
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

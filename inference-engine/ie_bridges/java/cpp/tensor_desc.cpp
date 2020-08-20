#include <inference_engine.hpp>

#include "openvino_java.hpp"
#include "enum_mapping.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_TensorDesc_GetTensorDesc(JNIEnv *env, jobject obj, jint precision, jintArray dims, jint layout)
{
    static const char method_name[] = "GetTensorDesc";
    try
    {
        auto l = precision_map.find(precision);
        if (l == precision_map.end())
            throw std::runtime_error("No such precision value!");

        auto pr = layout_map.find(layout);
        if (pr == layout_map.end())
            throw std::runtime_error("No such layout value!");

        auto n_precision = precision_map.at(precision);
        auto n_layout = layout_map.at(layout);

        TensorDesc *tDesc = new TensorDesc(n_precision, jintArrayToVector(env, dims), n_layout);

        return (jlong)tDesc;
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

JNIEXPORT jintArray JNICALL Java_org_intel_openvino_TensorDesc_GetDims(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "GetDims";
    try
    {
        TensorDesc *tDesc = (TensorDesc *)addr;
        std::vector<size_t> size_t_dims = tDesc->getDims();

        jintArray result = env->NewIntArray(size_t_dims.size());
        jint *arr = env->GetIntArrayElements(result, nullptr);

        for (int i = 0; i < size_t_dims.size(); ++i)
            arr[i] = size_t_dims[i];

        env->ReleaseIntArrayElements(result, arr, 0);
        return result;
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

JNIEXPORT jint JNICALL Java_org_intel_openvino_TensorDesc_getLayout(JNIEnv *env, jobject, jlong addr)
{
    static const char method_name[] = "getLayout";
    try
    {
        TensorDesc *tDesk = (TensorDesc *)addr;
        Layout layout = tDesk->getLayout();

        return find_by_value(layout_map, layout);

    } catch (const std::exception &e){
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT jint JNICALL Java_org_intel_openvino_TensorDesc_getPrecision(JNIEnv *env, jobject, jlong addr)
{
    static const char method_name[] = "getPrecision";
    try
    {
        TensorDesc *tDesk = (TensorDesc *)addr;
        Precision precision = tDesk->getPrecision();

        return find_by_value(precision_map, precision);

    } catch (const std::exception &e){
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_TensorDesc_delete(JNIEnv *, jobject, jlong addr)
{
    TensorDesc *tDesk = (TensorDesc *)addr;
    delete tDesk;
}

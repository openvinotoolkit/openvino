#include <jni.h>   // JNI header provided by JDK
#include <stdio.h> // C Standard IO Header
#include <inference_engine.hpp>
#include <stdexcept>

#include "openvino_java.hpp"
#include "enum_mapping.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_InputInfo_getPreProcess(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "getPreProcess";
    try
    {
        InputInfo *input_info = (InputInfo*)addr;
        return (jlong)(&input_info->getPreProcess());

    } catch (const std::exception &e){
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_InputInfo_SetLayout(JNIEnv *env, jobject obj, jlong addr, jint layout)
{
    static const char method_name[] = "SetLayout";
    try
    {
        InputInfo *input_info = (InputInfo*)addr;
        auto it = layout_map.find(layout);

        if (it == layout_map.end())
            throw std::runtime_error("No such layout value!");

        input_info->setLayout(it->second);

    } catch (const std::exception &e){
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT jint JNICALL Java_org_intel_openvino_InputInfo_getLayout(JNIEnv *env, jobject, jlong addr)
{
    static const char method_name[] = "getLayout";
    try
    {
        InputInfo *input_info = (InputInfo*)addr;
        Layout layout = input_info->getLayout();

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

JNIEXPORT void JNICALL Java_org_intel_openvino_InputInfo_SetPrecision(JNIEnv *env, jobject obj, jlong addr, jint precision)
{
    static const char method_name[] = "SetPrecision";
    try
    {
        InputInfo *input_info = (InputInfo*)addr;
        auto it = precision_map.find(precision);

        if (it == precision_map.end())
            throw std::runtime_error("No such precision value!");

        input_info->setPrecision(it->second);
        
    } catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT jint JNICALL Java_org_intel_openvino_InputInfo_getPrecision(JNIEnv *env, jobject, jlong addr)
{
    static const char method_name[] = "getPrecision";
    try
    {
        InputInfo *input_info = (InputInfo*)addr;
        Precision precision = input_info->getPrecision();

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

JNIEXPORT jlong JNICALL Java_org_intel_openvino_InputInfo_GetTensorDesc(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "GetTensorDesc";
    try
    {
        InputInfo *input_info = (InputInfo*)addr;
        TensorDesc *tDesc = new TensorDesc(input_info->getTensorDesc());

        return (jlong)tDesc;
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

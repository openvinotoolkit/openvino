#include <inference_engine.hpp>

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Blob_GetTensorDesc(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "GetTensorDesc";
    try
    {
        Blob::Ptr *output = reinterpret_cast<Blob::Ptr *>(addr);
        TensorDesc *tDesc = new TensorDesc((*output)->getTensorDesc());

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

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Blob_GetBlob(JNIEnv *env, jobject obj, jlong tensorDescAddr)
{
    static const char method_name[] = "GetBlob";
    try
    {
        TensorDesc *tDesc = (TensorDesc *)tensorDescAddr;

        Blob::Ptr *blob = new Blob::Ptr();
        *blob = make_shared_blob<uint8_t>(*tDesc);

        return (jlong)blob;
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

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Blob_BlobByte(JNIEnv *env, jobject obj, jlong tensorDescAddr, jbyteArray data)
{
    static const char method_name[] = "BlobByte";
    try
    {
        TensorDesc *tDesc = (TensorDesc *)tensorDescAddr;

        Blob::Ptr *blob = new Blob::Ptr();

        *blob = make_shared_blob<uint8_t>((*tDesc));
        (*blob)->allocate();
        env->GetByteArrayRegion(data, 0, (*blob)->size(), (*blob)->buffer());

        return (jlong)blob;
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

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Blob_BlobFloat(JNIEnv *env, jobject obj, jlong tensorDescAddr, jfloatArray data)
{
    static const char method_name[] = "BlobFloat";
    try
    {
        TensorDesc *tDesc = (TensorDesc *)tensorDescAddr;

        Blob::Ptr *blob = new Blob::Ptr();

        *blob = make_shared_blob<float>((*tDesc));
        (*blob)->allocate();
        env->GetFloatArrayRegion(data, 0, (*blob)->size(), (*blob)->buffer());

        return (jlong)blob;
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

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Blob_BlobCArray(JNIEnv *env, jobject obj, jlong tensorDescAddr, jlong matDataAddr)
{
    static const char method_name[] = "BlobCArray";
    try
    {
        TensorDesc *tDesc = (TensorDesc *)tensorDescAddr;

        auto precision = tDesc->getPrecision();

        std::vector<size_t> dims = tDesc->getDims();
        Blob::Ptr *blob = new Blob::Ptr();

        switch (precision) {
            case Precision::FP32:
            {
                float *data = (float *) matDataAddr;
                *blob = make_shared_blob<float>((*tDesc), data);
                break;
            }
            case Precision::Q78:
            case Precision::I16:
            case Precision::FP16:
            {
                short *data = (short *) matDataAddr;
                *blob = make_shared_blob<short>((*tDesc), data);
                break;
            }
            case Precision::U8:
            {
                uint8_t *data = (uint8_t *) matDataAddr;
                *blob = make_shared_blob<uint8_t>((*tDesc), data);
                break;
            }
            case Precision::I8:
            {
                int8_t *data = (int8_t *) matDataAddr;
                *blob = make_shared_blob<int8_t>((*tDesc), data);
                break;
            }
            case Precision::I32:
            {
                int32_t *data = (int32_t *) matDataAddr;
                *blob = make_shared_blob<int32_t>((*tDesc), data);
                break;
            }
            case Precision::BF16:
            {
                short *data = (short *) matDataAddr;
                *blob = make_shared_blob<short>((*tDesc), data);
                break;
            }
            default:
                throw std::runtime_error("Unsupported precision value!");
        }

        return (jlong)blob;
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

JNIEXPORT jint JNICALL Java_org_intel_openvino_Blob_size(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "size";
    try
    {
        Blob::Ptr *output = reinterpret_cast<Blob::Ptr *>(addr);
        int size = (*output)->size();

        return size;
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

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Blob_rmap(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "rmap";
    try
    {
        Blob::Ptr *output = reinterpret_cast<Blob::Ptr *>(addr);

        if ((*output)->is<MemoryBlob>()) {
            LockedMemory<const void> *lmem = new LockedMemory<const void> (as<MemoryBlob>(*output)->rmap());
            return (jlong)lmem;
        } else {
            throw std::runtime_error("Target Blob cannot be cast to the MemoryBlob!");
        }
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

JNIEXPORT void JNICALL Java_org_intel_openvino_Blob_delete(JNIEnv *, jobject, jlong addr)
{
    Blob::Ptr *output = reinterpret_cast<Blob::Ptr *>(addr);
    delete output;
}

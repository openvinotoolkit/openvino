#include <inference_engine.hpp>
#include <mutex>  

#include "openvino_java.hpp"
#include "enum_mapping.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_Infer(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "Infer";
    try
    {
        InferRequest *infer_request = (InferRequest *)addr;
        infer_request->Infer();
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

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_StartAsync(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "StartAsync";
    try
    {
        InferRequest *infer_request = (InferRequest *)addr;
        infer_request->StartAsync();
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

JNIEXPORT jint JNICALL Java_org_intel_openvino_InferRequest_Wait(JNIEnv *env, jobject obj, jlong addr, jint wait_mode)
{
    static const char method_name[] = "Wait";
    try
    {
        auto it = wait_mode_map.find(wait_mode);
        InferRequest *infer_request = (InferRequest *)addr;

        if (it == wait_mode_map.end())
            throw std::runtime_error("No such WaitMode value!");

        InferenceEngine::StatusCode status_code = infer_request->Wait(it->second);

        auto code = status_code_map.find(status_code);
        if (code == status_code_map.end())
            throw std::runtime_error("No such StatusCode value!");

        return (jint)(code->second);
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

static std::mutex map_mutex;
static std::map<jlong, jobject> runnable_glob_map;

JNIEXPORT jint JNICALL Java_org_intel_openvino_InferRequest_SetCompletionCallback(JNIEnv *env, jobject, jlong addr, jobject runnable){
    static const char method_name[] = "SetCompletionCallback";
    try
    {
        const std::lock_guard<std::mutex> lock(map_mutex);

        InferRequest *infer_request = (InferRequest *)addr;
        jobject runnable_glob = env->NewGlobalRef(runnable);

        runnable_glob_map.insert( std::make_pair(addr, runnable_glob) );

        JavaVM* jvm;
        env->GetJavaVM(&jvm);
        int version = env->GetVersion();

        infer_request->SetCompletionCallback(
                [jvm, version, runnable_glob] {
                    JNIEnv* myNewEnv;
                    JavaVMAttachArgs args;
                    args.version = version;
                    args.name = NULL; 
                    args.group = NULL;
                    jvm->AttachCurrentThread((void**)&myNewEnv, &args);

                    jclass runnable_class = myNewEnv->GetObjectClass(runnable_glob);
                    jmethodID run_method_id = myNewEnv->GetMethodID(runnable_class, "run","()V");
                    myNewEnv->CallNonvirtualVoidMethod(runnable_glob, runnable_class, run_method_id);

                    jvm->DetachCurrentThread();     
        });
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

JNIEXPORT long JNICALL Java_org_intel_openvino_InferRequest_GetBlob(JNIEnv *env, jobject obj, jlong addr, jstring output_name)
{
    static const char method_name[] = "GetBlob";
    try
    {
        InferRequest *infer_request = (InferRequest *)addr;

        std::string n_output_name =  jstringToString(env, output_name);

        Blob::Ptr *output = new Blob::Ptr();
        *output = infer_request->GetBlob(n_output_name);

        return (jlong)(output);
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

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_SetBlob(JNIEnv *env, jobject obj, jlong addr, jstring input_name, jlong blobAddr)
{
    static const char method_name[] = "SetBlob";
    try
    {
        InferRequest *infer_request = (InferRequest *)addr;
        
        Blob::Ptr *blob = reinterpret_cast<Blob::Ptr *>(blobAddr);

        std::string n_input_name = jstringToString(env, input_name);

        infer_request->SetBlob(n_input_name, (*blob));
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

JNIEXPORT jobject JNICALL Java_org_intel_openvino_InferRequest_GetPerformanceCounts(JNIEnv *env, jobject, jlong addr)
{
    static const char method_name[] = "GetPerformanceCounts";
    try
    {
        InferRequest *infer_request = (InferRequest *)addr;
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfomance;
        perfomance = infer_request->GetPerformanceCounts();

        jclass hashMap_class = env->FindClass("java/util/LinkedHashMap");
        jmethodID init_method_id = env->GetMethodID(hashMap_class, "<init>", "()V");
        jobject hashMap_object = env->NewObject(hashMap_class, init_method_id);
        jmethodID put_method_id = env->GetMethodID(hashMap_class, "put",
                                                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");

        jclass IEProfileInfo_class = env->FindClass("org/intel/openvino/InferenceEngineProfileInfo");
        jmethodID IEProfileInfo_init_id = env->GetMethodID(IEProfileInfo_class,"<init>",
                                                           "(Lorg/intel/openvino/InferenceEngineProfileInfo$LayerStatus;JJLjava/lang/String;Ljava/lang/String;I)V");

        jclass layerStatus_enum = env->FindClass("org/intel/openvino/InferenceEngineProfileInfo$LayerStatus");
        jmethodID valueOf_method_id = env->GetStaticMethodID(layerStatus_enum,"valueOf","(I)Lorg/intel/openvino/InferenceEngineProfileInfo$LayerStatus;");

        for (const auto& itr : perfomance) { 
            InferenceEngine::InferenceEngineProfileInfo ie_prof_info = itr.second;
            auto it = layer_status_map.find(ie_prof_info.status);

            if (it == layer_status_map.end())
                throw std::runtime_error("No such LayerStatus value!");

            jobject layer_status = env->CallStaticObjectMethod(layerStatus_enum, valueOf_method_id, it->second);

            std::string layer_type(ie_prof_info.layer_type);
            std::string exec_type(ie_prof_info.exec_type);
            
            jobject IEProfileInfo = env->NewObject(IEProfileInfo_class, IEProfileInfo_init_id, layer_status, 
                                                    ie_prof_info.realTime_uSec, ie_prof_info.cpu_uSec, env->NewStringUTF(exec_type.c_str()), 
                                                    env->NewStringUTF(layer_type.c_str()), static_cast<jint>(ie_prof_info.execution_index));
                                
            env->CallObjectMethod(hashMap_object, put_method_id, env->NewStringUTF(itr.first.c_str()), IEProfileInfo);
        }

        env->PopLocalFrame(hashMap_object);
        return hashMap_object;
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

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_delete(JNIEnv *env, jobject, jlong addr)
{   
    auto it = runnable_glob_map.find(addr);
    if (it != runnable_glob_map.end()) {
        env->DeleteGlobalRef(it->second);
    }
    
    InferRequest *infer_request = (InferRequest *)addr;
    delete infer_request;
}

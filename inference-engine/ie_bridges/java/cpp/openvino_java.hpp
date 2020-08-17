#include <jni.h>

#ifdef __cplusplus
extern "C"
{
#endif

//
// IECore
//
JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_ReadNetwork1(JNIEnv *, jobject, jlong, jstring, jstring);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_ReadNetwork(JNIEnv *, jobject, jlong, jstring);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_LoadNetwork(JNIEnv *, jobject, jlong, jlong, jstring);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_LoadNetwork1(JNIEnv *, jobject, jlong, jlong, jstring, jobject);
JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_RegisterPlugin(JNIEnv *, jobject, jlong, jstring, jstring);
JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_UnregisterPlugin(JNIEnv *, jobject, jlong, jstring);
JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_AddExtension(JNIEnv *, jobject, jlong, jstring);
JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_AddExtension1(JNIEnv *, jobject, jlong, jstring, jstring);
JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_RegisterPlugins(JNIEnv *, jobject, jlong, jstring);
JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_SetConfig(JNIEnv *, jobject, jlong, jobject, jstring);
JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_SetConfig1(JNIEnv *, jobject, jlong, jobject);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_GetConfig(JNIEnv *, jobject, jlong, jstring, jstring);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_GetCore(JNIEnv *, jobject);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_IECore_GetCore_1(JNIEnv *, jobject, jstring);
JNIEXPORT void JNICALL Java_org_intel_openvino_IECore_delete(JNIEnv *, jobject, jlong);

//
// CNNNetwork
//
JNIEXPORT jstring JNICALL Java_org_intel_openvino_CNNNetwork_getName(JNIEnv *, jobject, jlong);
JNIEXPORT jint JNICALL Java_org_intel_openvino_CNNNetwork_getBatchSize(JNIEnv *, jobject, jlong);
JNIEXPORT jobject JNICALL Java_org_intel_openvino_CNNNetwork_GetInputsInfo(JNIEnv *, jobject, jlong);
JNIEXPORT jobject JNICALL Java_org_intel_openvino_CNNNetwork_GetOutputsInfo(JNIEnv *, jobject, jlong);
JNIEXPORT void JNICALL Java_org_intel_openvino_CNNNetwork_reshape(JNIEnv *, jobject, jlong, jobject);
JNIEXPORT jobject JNICALL Java_org_intel_openvino_CNNNetwork_getInputShapes(JNIEnv *, jobject, jlong);
JNIEXPORT void JNICALL Java_org_intel_openvino_CNNNetwork_addOutput(JNIEnv *, jobject, jlong, jstring, jint);
JNIEXPORT void JNICALL Java_org_intel_openvino_CNNNetwork_addOutput1(JNIEnv *, jobject, jlong, jstring);
JNIEXPORT void JNICALL Java_org_intel_openvino_CNNNetwork_delete(JNIEnv *, jobject, jlong);

//
// InferRequest
//
JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_Infer(JNIEnv *, jobject, jlong);
JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_StartAsync(JNIEnv *, jobject, jlong);
JNIEXPORT jint JNICALL Java_org_intel_openvino_InferRequest_Wait(JNIEnv *, jobject, jlong, jint);
JNIEXPORT jint JNICALL Java_org_intel_openvino_InferRequest_SetCompletionCallback(JNIEnv *, jobject, jlong, jobject);
JNIEXPORT long JNICALL Java_org_intel_openvino_InferRequest_GetBlob(JNIEnv *, jobject, jlong, jstring);
JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_SetBlob(JNIEnv *, jobject, jlong, jstring, jlong);
JNIEXPORT jobject JNICALL Java_org_intel_openvino_InferRequest_GetPerformanceCounts(JNIEnv *, jobject, jlong);
JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_delete(JNIEnv *, jobject, jlong);

//
// ExecutableNetwork
//
JNIEXPORT jlong JNICALL Java_org_intel_openvino_ExecutableNetwork_CreateInferRequest(JNIEnv *, jobject, jlong);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_ExecutableNetwork_GetMetric(JNIEnv *, jobject, jlong, jstring);
JNIEXPORT void JNICALL Java_org_intel_openvino_ExecutableNetwork_delete(JNIEnv *, jobject, jlong);

//
// Blob
//
JNIEXPORT jlong JNICALL Java_org_intel_openvino_Blob_GetTensorDesc(JNIEnv *, jobject, jlong);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_Blob_GetBlob(JNIEnv *, jobject, jlong);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_Blob_BlobByte(JNIEnv *, jobject, jlong, jbyteArray);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_Blob_BlobFloat(JNIEnv *, jobject, jlong, jfloatArray);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_Blob_BlobCArray(JNIEnv *, jobject, jlong, jlong);
JNIEXPORT jint JNICALL Java_org_intel_openvino_Blob_size(JNIEnv *, jobject ,jlong);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_Blob_rmap(JNIEnv *, jobject, jlong);
JNIEXPORT void JNICALL Java_org_intel_openvino_Blob_delete(JNIEnv *, jobject, jlong);

//
// LockedMemory
//
JNIEXPORT void JNICALL Java_org_intel_openvino_LockedMemory_asByte(JNIEnv *, jobject, jlong, jbyteArray);
JNIEXPORT void JNICALL Java_org_intel_openvino_LockedMemory_asFloat(JNIEnv *, jobject, jlong, jfloatArray);
JNIEXPORT void JNICALL Java_org_intel_openvino_LockedMemory_delete(JNIEnv *, jobject, jlong);

//
// InputInfo
//
JNIEXPORT jlong JNICALL Java_org_intel_openvino_InputInfo_getPreProcess(JNIEnv *, jobject, jlong);
JNIEXPORT void JNICALL Java_org_intel_openvino_InputInfo_SetLayout(JNIEnv *, jobject, jlong, jint);
JNIEXPORT jint JNICALL Java_org_intel_openvino_InputInfo_getLayout(JNIEnv *, jobject, jlong);
JNIEXPORT void JNICALL Java_org_intel_openvino_InputInfo_SetPrecision(JNIEnv *, jobject, jlong, jint);
JNIEXPORT jint JNICALL Java_org_intel_openvino_InputInfo_getPrecision(JNIEnv *, jobject, jlong);
JNIEXPORT jlong JNICALL Java_org_intel_openvino_InputInfo_GetTensorDesc(JNIEnv *, jobject, jlong);

//
// PreProcessInfo
//
JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessInfo_SetResizeAlgorithm(JNIEnv *, jobject, jlong, jint);

//
// TensorDesc
//
JNIEXPORT jlong JNICALL Java_org_intel_openvino_TensorDesc_GetTensorDesc(JNIEnv *, jobject, jint, jintArray, jint);
JNIEXPORT jintArray JNICALL Java_org_intel_openvino_TensorDesc_GetDims(JNIEnv *, jobject, jlong);
JNIEXPORT jint JNICALL Java_org_intel_openvino_TensorDesc_getLayout(JNIEnv *, jobject, jlong);
JNIEXPORT jint JNICALL Java_org_intel_openvino_TensorDesc_getPrecision(JNIEnv *, jobject, jlong);
JNIEXPORT void JNICALL Java_org_intel_openvino_TensorDesc_delete(JNIEnv *, jobject, jlong);

//
// Parameter
//
JNIEXPORT jstring JNICALL Java_org_intel_openvino_Parameter_asString(JNIEnv *, jobject, jlong);
JNIEXPORT jint JNICALL Java_org_intel_openvino_Parameter_asInt(JNIEnv *, jobject, jlong);
JNIEXPORT void JNICALL Java_org_intel_openvino_Parameter_delete(JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif

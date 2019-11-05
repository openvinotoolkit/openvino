#include "infer_request_wrapper.h"
#include "helpers.h"

void InferenceEngineBridge::InferRequestWrap::getBlobPtr(const std::string &blob_name,
                                                         InferenceEngine::Blob::Ptr &blob_ptr) {
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(request_ptr->GetBlob(blob_name.c_str(), blob_ptr, &response));
}


void InferenceEngineBridge::InferRequestWrap::setBatch(int size) {
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(request_ptr->SetBatch(size, &response));
}

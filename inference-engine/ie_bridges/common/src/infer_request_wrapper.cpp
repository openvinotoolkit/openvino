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

void InferenceEngineBridge::InferRequestWrap::infer() {
    InferenceEngine::ResponseDesc response;
    start_time = Time::now();
    IE_CHECK_CALL(request_ptr->Infer(&response));
    auto end_time = Time::now();
    auto execTime = std::chrono::duration_cast<InferenceEngineBridge::ns>(end_time - start_time);
    exec_time = static_cast<double>(execTime.count()) * 0.000001;
}


void InferenceEngineBridge::InferRequestWrap::infer_async() {
    InferenceEngine::ResponseDesc response;
    start_time = Time::now();
    IE_CHECK_CALL(request_ptr->SetUserData(this, &response));
    request_ptr->SetCompletionCallback(latency_callback);
    IE_CHECK_CALL(request_ptr->StartAsync(&response));
}

int InferenceEngineBridge::InferRequestWrap::wait(int64_t timeout) {
    InferenceEngine::ResponseDesc responseDesc;
    InferenceEngine::StatusCode code = request_ptr->Wait(timeout, &responseDesc);
    return static_cast<int >(code);
}

void InferenceEngineBridge::InferRequestWrap::setCyCallback(InferenceEngineBridge::cy_callback callback, void *data) {
    user_callback = callback;
    user_data = data;
}

std::map<std::string, InferenceEngineBridge::ProfileInfo>
InferenceEngineBridge::InferRequestWrap::getPerformanceCounts() {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perf_counts;
    InferenceEngine::ResponseDesc response;
    request_ptr->GetPerformanceCounts(perf_counts, &response);
    std::map<std::string, InferenceEngineBridge::ProfileInfo> perf_map;

    for (auto it : perf_counts) {
        InferenceEngineBridge::ProfileInfo profile_info;
        switch (it.second.status) {
            case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
                profile_info.status = "EXECUTED";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
                profile_info.status = "NOT_RUN";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                profile_info.status = "OPTIMIZED_OUT";
                break;
            default:
                profile_info.status = "UNKNOWN";
        }
        profile_info.exec_type = it.second.exec_type;
        profile_info.layer_type = it.second.layer_type;
        profile_info.cpu_time = it.second.cpu_uSec;
        profile_info.real_time = it.second.realTime_uSec;
        profile_info.execution_index = it.second.execution_index;
        perf_map[it.first] = profile_info;
    }
    return perf_map;
}
#ifndef INFERENCEENGINE_BRIDGE_INFER_REQUEST_WRAPPER_H
#define INFERENCEENGINE_BRIDGE_INFER_REQUEST_WRAPPER_H

#include <chrono>

#include <ie_iinfer_request.hpp>

#include "profile_info.h"


namespace InferenceEngineBridge {
    using cy_callback = void (*)(void *, int);
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::nanoseconds ns;

    struct InferRequestWrap {
        void infer();

        void infer_async();

        int wait(int64_t timeout);

        void setCyCallback(cy_callback callback, void *data);

        void getBlobPtr(const std::string &blob_name, InferenceEngine::Blob::Ptr &blob_ptr);

        void setBatch(int size);

        std::map<std::string, InferenceEngineBridge::ProfileInfo> getPerformanceCounts();

        InferenceEngine::IInferRequest::Ptr request_ptr;
        Time::time_point start_time;
        double exec_time;
        cy_callback user_callback;
        void *user_data;
        int status;

    };
}
#endif //INFERENCEENGINE_BRIDGE_INFER_REQUEST_WRAPPER_H

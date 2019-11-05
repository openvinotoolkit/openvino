#include <ie_plugin_config.hpp>

#include "helpers.h"
#include "infer_request_wrapper.h"

std::map<std::string, InferenceEngine::Precision> InferenceEngineBridge::precision_map = {{"FP32", InferenceEngine::Precision::FP32},
                                                                                          {"FP16", InferenceEngine::Precision::FP16},
                                                                                          {"Q78",  InferenceEngine::Precision::Q78},
                                                                                          {"I32",  InferenceEngine::Precision::I32},
                                                                                          {"I16",  InferenceEngine::Precision::I16},
                                                                                          {"I8",   InferenceEngine::Precision::I8},
                                                                                          {"U16",  InferenceEngine::Precision::U16},
                                                                                          {"U8",   InferenceEngine::Precision::U8}};

std::map<std::string, InferenceEngine::Layout> InferenceEngineBridge::layout_map = {{"ANY",     InferenceEngine::Layout::ANY},
                                                                                    {"NCHW",    InferenceEngine::Layout::NCHW},
                                                                                    {"NHWC",    InferenceEngine::Layout::NHWC},
                                                                                    {"OIHW",    InferenceEngine::Layout::OIHW},
                                                                                    {"C",       InferenceEngine::Layout::C},
                                                                                    {"CHW",     InferenceEngine::Layout::CHW},
                                                                                    {"HW",      InferenceEngine::Layout::HW},
                                                                                    {"NC",      InferenceEngine::Layout::NC},
                                                                                    {"CN",      InferenceEngine::Layout::CN},
                                                                                    {"NCDHW",   InferenceEngine::Layout::NCDHW},
                                                                                    {"BLOCKED", InferenceEngine::Layout::BLOCKED}};

uint32_t InferenceEngineBridge::getOptimalNumberOfRequests(const InferenceEngine::IExecutableNetwork::Ptr &actual) {
    try {
        InferenceEngine::ResponseDesc response;
        InferenceEngine::Parameter parameter_value;
        IE_CHECK_CALL(actual->GetMetric(METRIC_KEY(SUPPORTED_METRICS), parameter_value, &response));
        auto supported_metrics = parameter_value.as<std::vector<std::string >>();
        std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
        if (std::find(supported_metrics.begin(), supported_metrics.end(), key) != supported_metrics.end()) {
            IE_CHECK_CALL(actual->GetMetric(key, parameter_value, &response));
            if (parameter_value.is<unsigned int>())
                return parameter_value.as<unsigned int>();
            else
                THROW_IE_EXCEPTION << "Unsupported format for " << key << "!"
                                   << " Please specify number of infer requests directly!";
        } else {
            THROW_IE_EXCEPTION << "Can't load network: " << key << " is not supported!"
                               << " Please specify number of infer requests directly!";
        }
    } catch (const std::exception &ex) {
        THROW_IE_EXCEPTION << "Can't load network: " << ex.what()
                           << " Please specify number of infer requests directly!";
    }
}


void InferenceEngineBridge::latency_callback(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) {
    if (code != InferenceEngine::StatusCode::OK) {
        THROW_IE_EXCEPTION << "Async Infer Request failed with status code " << code;
    }
    InferenceEngineBridge::InferRequestWrap *requestWrap;
    InferenceEngine::ResponseDesc dsc;
    request->GetUserData(reinterpret_cast<void **>(&requestWrap), &dsc);
    auto end_time = Time::now();
    auto execTime = std::chrono::duration_cast<ns>(end_time - requestWrap->start_time);
    requestWrap->exec_time = static_cast<double>(execTime.count()) * 0.000001;
    if (requestWrap->user_callback) {
        requestWrap->user_callback(requestWrap->user_data, code);
    }
}
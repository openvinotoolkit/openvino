#include <map>
#include "inference_engine.hpp"

//
// resize_algorithm
//
static const std::map<int, InferenceEngine::ResizeAlgorithm> resize_alg_map = {
    {0, InferenceEngine::ResizeAlgorithm::NO_RESIZE},
    {1, InferenceEngine::ResizeAlgorithm::RESIZE_AREA},
    {2, InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR}
};

//
// layout
//
static const std::map<int, InferenceEngine::Layout> layout_map = {
    {0, InferenceEngine::Layout::ANY},
    {1, InferenceEngine::Layout::NCHW},
    {2, InferenceEngine::Layout::NHWC},
    {3, InferenceEngine::Layout::NCDHW},
    {4, InferenceEngine::Layout::NDHWC},
    {64, InferenceEngine::Layout::OIHW},
    {95, InferenceEngine::Layout::SCALAR},
    {96, InferenceEngine::Layout::C},
    {128, InferenceEngine::Layout::CHW},
    {192, InferenceEngine::Layout::HW},
    {193, InferenceEngine::Layout::NC},
    {194, InferenceEngine::Layout::CN},
    {200, InferenceEngine::Layout::BLOCKED}
};

//
// precision
//
static const std::map<int, InferenceEngine::Precision> precision_map = {
    {255, InferenceEngine::Precision::UNSPECIFIED},
    {0, InferenceEngine::Precision::MIXED},
    {10, InferenceEngine::Precision::FP32},
    {11, InferenceEngine::Precision::FP16},
    {20, InferenceEngine::Precision::Q78},
    {30, InferenceEngine::Precision::I16},
    {40, InferenceEngine::Precision::U8},
    {50, InferenceEngine::Precision::I8},
    {60, InferenceEngine::Precision::U16},
    {70, InferenceEngine::Precision::I32},
    {72, InferenceEngine::Precision::I64},
    {71, InferenceEngine::Precision::BIN},
    {80, InferenceEngine::Precision::CUSTOM}
};

//
// wait_mode
//
static const std::map<int, InferenceEngine::IInferRequest::WaitMode> wait_mode_map = {
    {-1, InferenceEngine::IInferRequest::WaitMode::RESULT_READY},
    {0, InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY}
};

//
// status_code
//    
static const std::map<InferenceEngine::StatusCode, int> status_code_map = {
    {InferenceEngine::StatusCode::OK, 0},
    {InferenceEngine::StatusCode::GENERAL_ERROR, -1},
    {InferenceEngine::StatusCode::NOT_IMPLEMENTED, -2},
    {InferenceEngine::StatusCode::NETWORK_NOT_LOADED, -3},
    {InferenceEngine::StatusCode::PARAMETER_MISMATCH, -4}, 
    {InferenceEngine::StatusCode::NOT_FOUND, -5}, 
    {InferenceEngine::StatusCode::OUT_OF_BOUNDS, -6}, 
    {InferenceEngine::StatusCode::UNEXPECTED, -7}, 
    {InferenceEngine::StatusCode::REQUEST_BUSY, -8}, 
    {InferenceEngine::StatusCode::RESULT_NOT_READY, -9}, 
    {InferenceEngine::StatusCode::NOT_ALLOCATED, -10}, 
    {InferenceEngine::StatusCode::INFER_NOT_STARTED, -11}, 
    {InferenceEngine::StatusCode::NETWORK_NOT_READ, -12}, 
};

//
// layer_status
//
static const std::map<InferenceEngine::InferenceEngineProfileInfo::LayerStatus, int> layer_status_map = {
    {InferenceEngine::InferenceEngineProfileInfo::LayerStatus::NOT_RUN, 0},
    {InferenceEngine::InferenceEngineProfileInfo::LayerStatus::OPTIMIZED_OUT, 1},
    {InferenceEngine::InferenceEngineProfileInfo::LayerStatus::EXECUTED, 2},
};

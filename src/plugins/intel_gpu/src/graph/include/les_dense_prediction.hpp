#pragma once

#include <functional>

namespace cldnn {
class network;
class primitive_inst;
}

namespace ov::intel_gpu::ocl::moe {
struct DensePredictionTrigger {
    const cldnn::primitive_inst* source = nullptr;
    std::function<void()> predict;
};

DensePredictionTrigger get_dense_prediction_trigger(cldnn::network& network);
}
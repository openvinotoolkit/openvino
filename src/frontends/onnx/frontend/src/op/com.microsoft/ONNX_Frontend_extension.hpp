#ifndef PAD_HPP
#define PAD_HPP

#include <vector>

namespace onnx_opset {
    namespace op {
        void Pad(const float* input_data, const int64_t* pads_data, float* output_data, const std::vector<int64_t>& input_shape, const std::vector<int64_t>& output_shape, const std::string& mode, float constant_value = 0.0);
    }
}

#endif

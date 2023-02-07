// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocessing.hpp"


#include "openvino/core/model.hpp"
#include "openvino/core/shape.hpp"
#include "ngraph/opsets/opset9.hpp"
#include "gna_data_types.hpp"

#ifdef GNA_DEBUG
#include <ngraph/pass/manager.hpp>
#include "transformations/serialize.hpp"
#endif

using namespace ngraph::opset9;

namespace ov {
namespace intel_gna {

int16_t ConvertFloatToInt16(float src) {
    float rounding_value = (src > 0) ? 0.5f : -0.5f;
    float value = src + rounding_value;
    if (value > 32767.0) {
        return 32767;
    } else if (value < -32768.0) {
        return -32768;
    }
    return (int16_t)value;
}

int8_t ConvertFloatToInt8(float src) {
    float rounding_value = (src > 0) ? 0.5f : -0.5f;
    float value = src + rounding_value;
    if (value > 127.0) {
        return 127;
    } else if (value < -128.0) {
        return -128;
    }
    return (int8_t)value;
}

void ConvertToInt16(int16_t* ptr_dst,
                    const float* ptr_src,
                    const uint32_t num_rows,
                    const uint32_t num_columns,
                    const float scale_factor) {
    if (!ptr_dst || !ptr_src) {
        return;
    }
    for (uint32_t i = 0; i < num_rows * num_columns; i++) {
        ptr_dst[i] = ConvertFloatToInt16(ptr_src[i] * scale_factor);
    }
}

/*
Convert legacy transposition info to preprocessing model
 */
std::shared_ptr<ov::Model> to_pre_post_process_model(const InferenceEngine::SizeVector& input_dims, const size_t& c_size, const size_t& hw_size) {
    auto param = std::make_shared<Parameter>(element::f32, ov::Shape(input_dims));

    // legacy way was to revert C and HW dimentions in the reshaped tensor
    std::vector<int32_t> reshape_pattern{-1, c_size, hw_size};
    auto reshape_const = std::make_shared<Constant>(element::i32, Shape{reshape_pattern.size()}, reshape_pattern);
    auto reshape = std::make_shared<Reshape>(param, reshape_const, false);

    // NCHW -> NHWC or NHWC -> NCHW
    std::vector<int8_t> transpose_order{0, 2, 1};
    auto transpose_const = std::make_shared<Constant>(element::i8, Shape{transpose_order.size()}, transpose_order);
    auto transpose = std::make_shared<Transpose>(reshape, transpose_const);

    auto result = std::make_shared<Result>(transpose);

    std::shared_ptr<Model> model = std::make_shared<Model>(ResultVector{result}, ParameterVector{param});

    return model;
}

}  // namespace intel_gna
}  // namespace ov

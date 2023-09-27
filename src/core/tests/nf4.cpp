// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/runtime/core.hpp"

using namespace std;

template <typename T>
inline std::shared_ptr<ov::Model> get_model(const std::vector<T>& const_data, ov::element::Type_t ov_type) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov_type, ov::Shape{const_data.size()});
    auto constant = ov::op::v0::Constant::create(ov::element::nf4, ov::Shape{const_data.size()}, const_data);
    auto convert = std::make_shared<ov::op::v0::Convert>(constant, ov_type);

    auto add = std::make_shared<ov::op::v1::Add>(data, convert);
    return std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{data});
}

TEST(nf4, convert_nf4_to_string) {
    vector<uint8_t> values{186, 17};
    auto constant = make_shared<ov::op::v0::Constant>(ov::element::nf4, ov::Shape{3}, &values[0]);

    vector<string> ref{"10", "11", "1"};
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(constant->convert_value_to_string(i), ref[i]);
    }
}

TEST(nf4, tensor_or_constant_size) {
    vector<uint8_t> values{171, 16};
    auto constant = make_shared<ov::op::v0::Constant>(ov::element::nf4, ov::Shape{3}, &values[0]);
    EXPECT_EQ(2, constant->get_byte_size());

    ov::Tensor runtime_tensor(ov::element::nf4, ov::Shape{3});
    EXPECT_EQ(constant->get_byte_size(), runtime_tensor.get_byte_size());
}

template <typename T>
void test_nf4_inference(ov::element::Type_t ov_type) {
    ov::Core core;
    vector<T> const_data{-1.5,   -1.425, -1.35,  -1.275, -1.2,   -1.125, -1.05,  -0.975, -0.9,   -0.825, -0.75,
                         -0.675, -0.6,   -0.525, -0.45,  -0.375, -0.3,   -0.225, -0.15,  -0.075, 0.0,    0.075,
                         0.15,   0.225,  0.3,    0.375,  0.45,   0.525,  0.6,    0.675,  0.75,   0.825,  0.9,
                         0.975,  1.05,   1.125,  1.2,    1.275,  1.35,   1.425,  1.5};

    vector<T> target{-1.0,
                     -0.6961928009986877,
                     -0.5250730514526367,
                     -0.39491748809814453,
                     -0.28444138169288635,
                     -0.18477343022823334,
                     -0.09105003625154495,
                     0.0,
                     0.07958029955625534,
                     0.16093020141124725,
                     0.24611230194568634,
                     0.33791524171829224,
                     0.44070982933044434,
                     0.5626170039176941,
                     0.7229568362236023,
                     1.0};

    auto model = get_model<T>(const_data, ov_type);
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

    std::vector<T> model_input(const_data.size(), 0.0);
    ov::Tensor model_input_ov{ov_type, ov::Shape({model_input.size()}), &model_input[0]};

    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(0, model_input_ov);
    infer_request.infer();
    auto out = infer_request.get_output_tensor(0);
    T* out_p = static_cast<T*>(out.data(ov_type));
    auto out_val = std::vector<T>(out_p, out_p + out.get_size());

    auto it = std::unique(out_val.begin(), out_val.end());
    out_val.resize(std::distance(out_val.begin(), it));

    EXPECT_EQ(16, out_val.size());

    float max_diff = 0.0;
    for (size_t i = 0; i < 16; i++) {
        float diff = fabs(static_cast<float>(out_val[i] - target[i]));
        max_diff = std::max(max_diff, diff);
    }
    EXPECT_LE(max_diff, 0.001);
}

TEST(nf4, inference_float) {
    test_nf4_inference<float>(ov::element::f32);
    test_nf4_inference<ov::float16>(ov::element::f16);
    test_nf4_inference<ov::bfloat16>(ov::element::bf16);
}

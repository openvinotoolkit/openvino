// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"

namespace ov {
namespace test {
namespace {

/* A Result fed directly by a Parameter shares the tensor descriptor with the model input:

     Parameter
      |     \
      |     Sin
      |      |
   Result  Result

  The input and output tensors set by the user must stay independent regardless of the binding order.
*/

// Parameters: dynamic shape, bind inputs before outputs, with Sin branch.
using ParameterResultBindingParams = std::tuple<bool, bool, bool>;

class ParameterResultBindingTest : public testing::TestWithParam<ParameterResultBindingParams> {};

TEST_P(ParameterResultBindingTest, OutputWrittenInputUnchanged) {
    const auto [dynamic, inputs_first, with_sin] = GetParam();
    const ov::Shape shape{2, 3};
    const auto parameter =
        std::make_shared<ov::opset13::Parameter>(ov::element::f32,
                                                 dynamic ? ov::PartialShape{-1, 3} : ov::PartialShape(shape));
    ov::OutputVector outputs{parameter};
    if (with_sin) {
        outputs.push_back(std::make_shared<ov::opset13::Sin>(parameter));
    }
    const auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{parameter});

    ov::Core core;
    auto request =
        core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();

    for (size_t iter = 0; iter < 2; ++iter) {
        std::vector<float> input_values(ov::shape_size(shape));
        for (size_t i = 0; i < input_values.size(); ++i) {
            input_values[i] = static_cast<float>(i + 10 * iter);
        }
        const auto expected_input = input_values;
        ov::Tensor input(ov::element::f32, shape, input_values.data());
        std::vector<ov::Tensor> output_tensors;
        for (size_t i = 0; i < outputs.size(); ++i) {
            output_tensors.emplace_back(ov::element::f32, shape);
            std::fill_n(output_tensors.back().data<float>(), output_tensors.back().get_size(), 7.0f);
        }
        const auto bind_outputs = [&]() {
            for (size_t i = 0; i < output_tensors.size(); ++i) {
                request.set_output_tensor(i, output_tensors[i]);
            }
        };

        if (inputs_first) {
            request.set_input_tensor(input);
            bind_outputs();
        } else {
            bind_outputs();
            request.set_input_tensor(input);
        }
        request.infer();

        ASSERT_EQ(request.get_input_tensor().data(), input.data());
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            ASSERT_EQ(request.get_output_tensor(i).data(), output_tensors[i].data());
        }
        for (size_t i = 0; i < input_values.size(); ++i) {
            ASSERT_EQ(input_values[i], expected_input[i]) << "Input was modified, element " << i;
            ASSERT_EQ(output_tensors[0].data<float>()[i], expected_input[i]) << "Output 0, element " << i;
            if (with_sin) {
                ASSERT_FLOAT_EQ(output_tensors[1].data<float>()[i], std::sin(expected_input[i]))
                    << "Output 1, element " << i;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_ParameterResultBinding,
                         ParameterResultBindingTest,
                         testing::Combine(testing::Bool(), testing::Bool(), testing::Bool()),
                         [](const testing::TestParamInfo<ParameterResultBindingParams>& info) {
                             return std::string(std::get<0>(info.param) ? "Dynamic" : "Static") +
                                    (std::get<1>(info.param) ? "InputsFirst" : "OutputsFirst") +
                                    (std::get<2>(info.param) ? "WithSin" : "ParameterOnly");
                         });

}  // namespace
}  // namespace test
}  // namespace ov

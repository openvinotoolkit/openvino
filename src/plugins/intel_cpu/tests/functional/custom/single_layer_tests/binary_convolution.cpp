// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "openvino/op/binary_convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/reference/binary_convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace {

class BinaryConvolutionCPUTest : public SubgraphBaseStaticTest {
protected:
    const Shape input_shape {1, 256, 56, 56};
    const Shape weights_shape {5, 256, 1, 1};
    const Shape output_shape {1, 5, 56, 56};

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
        auto weights = std::make_shared<op::v0::Constant>(
            element::u1, weights_shape, packed_weights_data().data());
        auto binary_convolution = std::make_shared<op::v1::BinaryConvolution>(input,
                weights,
                Strides {1, 1},
                CoordinateDiff {0, 0},
                CoordinateDiff {0, 0},
                Strides {1, 1},
                op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT,
                -1.0f,
                op::PadType::EXPLICIT);
        auto result = std::make_shared<op::v0::Result>(binary_convolution);

        function = std::make_shared<Model>(OutputVector {result}, ParameterVector {input});
    }

    void generate_inputs(const std::vector<Shape>& target_input_static_shapes) override {
        ASSERT_EQ(target_input_static_shapes, std::vector<Shape> {input_shape});

        Tensor input_tensor(element::f32, input_shape);
        std::copy(logical_input_data().begin(), logical_input_data().end(), input_tensor.data<float>());

        inputs = {{function->input().get_node_shared_ptr(), input_tensor}};
    }

    std::vector<Tensor> calculate_refs() override {
        const auto input_it = inputs.find(function->input().get_node_shared_ptr());
        if (input_it == inputs.end()) return {};

        Tensor reference_output(element::f32, output_shape);
        const Shape reference_weights_shape {1, weights_shape[1], weights_shape[2], weights_shape[3]};
        const Shape reference_output_shape {1, 1, output_shape[2], output_shape[3]};
        const size_t weights_per_channel = shape_size(reference_weights_shape) / 8;
        const size_t output_per_channel = shape_size(reference_output_shape);
        for (size_t oc = 0; oc < weights_shape[0]; ++oc) {
            auto* output = reference_output.data<float>() + oc * output_per_channel;
            reference::binary_convolution(input_it->second.data<const float>(),
                          packed_weights_data().data() + oc * weights_per_channel,
                          output,
                          input_shape,
                          reference_weights_shape,
                          reference_output_shape,
                          Strides {1, 1},
                          Strides {1, 1},
                          CoordinateDiff {0, 0},
                          CoordinateDiff {0, 0},
                          -1.0f);
        }
        return {reference_output};
    }

    void validate() override {
        const auto actual_outputs = get_plugin_outputs();
        ASSERT_EQ(actual_outputs.size(), 1);
        EXPECT_EQ(actual_outputs[0].get_shape(), output_shape);
        EXPECT_EQ(actual_outputs[0].get_element_type(), element::f32);

        const auto expected_outputs = calculate_refs();
        compare(expected_outputs, actual_outputs);
    }

private:
    static const std::vector<float>& logical_input_data() {
        static const std::vector<float> data = [] {
            std::vector<float> values(shape_size(Shape {1, 256, 56, 56}));
            for (size_t idx = 0; idx < values.size(); ++idx) {
                values[idx] = idx % 3 == 0 ? 1.0f : 0.0f;
            }
            return values;
        }();
        return data;
    }

    static const std::vector<uint8_t>& logical_weights_data() {
        static const std::vector<uint8_t> data = [] {
            std::vector<uint8_t> values(5 * 256);
            for (size_t idx = 0; idx < values.size(); ++idx) {
                values[idx] = static_cast<uint8_t>(((idx * 37 + 13) ^ (idx >> 2)) & 1);
            }
            return values;
        }();
        return data;
    }

    static const std::vector<uint8_t>& packed_weights_data() {
        static const std::vector<uint8_t> data = [] {
            const auto& logical_weights = logical_weights_data();
            std::vector<uint8_t> packed_weights(logical_weights.size() / 8);
            for (size_t idx = 0; idx < logical_weights.size(); ++idx) {
                packed_weights[idx / 8] |= logical_weights[idx] << (7 - idx % 8);
            }
            return packed_weights;
        }();
        return data;
    }
};

TEST_F(BinaryConvolutionCPUTest, CompareOutputShapeAndValuesWithReference) {
    run();
}

}  // namespace
}  // namespace test
}  // namespace ov

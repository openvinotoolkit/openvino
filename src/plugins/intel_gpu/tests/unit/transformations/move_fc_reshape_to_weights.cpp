// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <plugin/transformations/move_fc_reshape_to_weights.hpp>

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <intel_gpu/op/fully_connected.hpp>
#include <intel_gpu/op/placeholder.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_gpu;

using MoveFCReshapeToWeightsParams = std::tuple<std::pair<ov::PartialShape, ov::Shape>,  // data_shape - weights_shape
                                                bool,                                    // add transpose
                                                bool>;                                   // add subtract

class MoveFCReshapeToWeightsTests : public TransformationTestsF, public WithParamInterface<MoveFCReshapeToWeightsParams> {
public:
    static std::string get_test_case_name(testing::TestParamInfo<MoveFCReshapeToWeightsParams> obj) {
        std::pair<ov::PartialShape, ov::Shape> input_shapes;
        bool add_transpose;
        bool add_subtract;
        std::tie(input_shapes, add_transpose, add_subtract) = obj.param;

        std::ostringstream result;
        result << "Input_shape=(" << input_shapes.first << ")_Weights_shape=(" << input_shapes.second
               << ")_add_transpose=" << add_transpose << "_add_subtract=" << add_subtract;
        return result.str();
    }

    static std::shared_ptr<ov::Model> init_model(const ov::PartialShape& data_shape,
                                                 const ov::Shape& weights_shape,
                                                 const bool add_transpose,
                                                 const bool add_subtract,
                                                 const bool add_reshape) {
        auto data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, data_shape);
        auto transposed_shape = weights_shape;
        if (add_transpose)
            std::swap(*(transposed_shape.rbegin() + 1), *transposed_shape.rbegin());
        std::shared_ptr<ov::Node> weights_path = ov::opset1::Constant::create(ov::element::u8, transposed_shape, {1});
        weights_path = std::make_shared<ov::opset1::Convert>(weights_path, ov::element::f32);

        ov::Shape decompression_shape(weights_shape.size(), 1);
        const size_t n_idx = add_transpose ? transposed_shape.size() - 1 : transposed_shape.size() - 2;
        decompression_shape[n_idx] = transposed_shape[n_idx];

        if (add_subtract) {
            auto sub_const = ov::opset1::Constant::create(ov::element::f32, decompression_shape, {1});
            weights_path = std::make_shared<ov::opset1::Subtract>(weights_path, sub_const);
        }
        auto mul_const = ov::opset1::Constant::create(ov::element::f32, decompression_shape, {1});
        weights_path = std::make_shared<ov::opset1::Multiply>(weights_path, mul_const);

        if (add_reshape) {
            auto target_shape = transposed_shape;
            target_shape.erase(target_shape.begin());
            auto reshape_const = ov::opset1::Constant::create(ov::element::i32, {2}, target_shape);
            weights_path = std::make_shared<ov::opset1::Reshape>(weights_path, reshape_const, false);
        }
        if (add_transpose) {
            auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {2}, {1, 0});
            weights_path = std::make_shared<ov::opset1::Transpose>(weights_path, transpose_const);
        }
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fully_connected = std::make_shared<op::FullyConnected>(data, weights_path, no_bias);
        return std::make_shared<ov::Model>(ov::NodeVector{fully_connected}, ov::ParameterVector{data});
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        std::pair<ov::PartialShape, ov::Shape> input_shapes;
        bool add_transpose;
        bool add_subtract;
        std::tie(input_shapes, add_transpose, add_subtract) = this->GetParam();

        ov::Shape ref_weights_shape = input_shapes.second;
        ref_weights_shape.erase(ref_weights_shape.begin());
        model = init_model(input_shapes.first, input_shapes.second, add_transpose, add_subtract, true);
        model_ref = init_model(input_shapes.first, ref_weights_shape, add_transpose, add_subtract, false);
        manager.register_pass<MoveFCReshapeToWeights>();
    }
};

TEST_P(MoveFCReshapeToWeightsTests, CompareFunctions) {}

const std::vector<std::pair<ov::PartialShape, ov::Shape>> input_shapes_wo_transpose = {
    {{-1, -1, -1}, {1, 4, 3}}
};
const std::vector<bool> add_transpose = {false, true};
const std::vector<bool> add_subtract = {false, true};

INSTANTIATE_TEST_SUITE_P(smoke_TransformationTests_wo_transpose, MoveFCReshapeToWeightsTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(input_shapes_wo_transpose),
                                ::testing::ValuesIn(add_transpose),
                                ::testing::ValuesIn(add_subtract)),
                            MoveFCReshapeToWeightsTests::get_test_case_name);

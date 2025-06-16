// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/reduce.hpp"
#include "common_test_utils/node_builders/convolution_backprop_data.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/transpose.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        std::vector<InputShape>, // input shapes
        ov::element::Type, // Network precision
        std::string // Device name
> reduceDeconvConcatDynamicGPUTestParamsSet;

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f16,
};

// Reduce should have preferred format for ouput layout
class ReduceDeconvConcatDynamicGPUTest : public testing::WithParamInterface<reduceDeconvConcatDynamicGPUTestParamsSet>,
                                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<reduceDeconvConcatDynamicGPUTestParamsSet>& obj) {
        reduceDeconvConcatDynamicGPUTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;
        std::vector<InputShape> inputShapes;
        ov::element::Type model_type;
        std::string targetDevice;

        std::tie(inputShapes, model_type, targetDevice) = basicParamsSet;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
            for (const auto& actual_shape : shape.second) {
                result << ov::test::utils::partialShape2str({actual_shape}) << "_";
            }
        }
        result << "model_type=" << model_type << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 80;
            in_data.resolution = 8;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        reduceDeconvConcatDynamicGPUTestParamsSet basicParamsSet = this->GetParam();
        std::vector<InputShape> inputShapes;
        ov::element::Type model_type;
        std::tie(inputShapes, model_type, targetDevice) = basicParamsSet;

        init_input_shapes(inputShapes);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        auto deconvOp = ov::test::utils::make_convolution_backprop_data(params[0], model_type, {2, 2, 2}, {2, 2, 2}, {0, 0, 0},
                                                                       {0, 0, 0}, {1, 1, 1}, ov::op::PadType::EXPLICIT, 16);
        deconvOp->set_friendly_name("deconv");

        std::vector<int> reduce_axes = {5};
        auto reduceAxesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({1}), reduce_axes);
        auto reduceOp = ov::test::utils::make_reduce(params[1], reduceAxesNode, false, ov::test::utils::ReductionType::Max);
        reduceOp->set_friendly_name("reduce");

        auto concatOp = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{deconvOp, reduceOp}, 1);
        concatOp->set_friendly_name("concat");

        std::vector<int> transpose_order = {0, 1, 2, 4, 3};
        auto transposeOrderNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({5}), transpose_order);
        auto transposeOp = std::make_shared<ov::op::v1::Transpose>(concatOp, transposeOrderNode);
        transposeOp->set_friendly_name("transpose");

        ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(transposeOp)};
        function = std::make_shared<ov::Model>(results, params, "transpose_out");
    }
};

TEST_P(ReduceDeconvConcatDynamicGPUTest, Inference) {
    run();
}

const std::vector<std::vector<ov::test::InputShape>> dynInputShapes = {
    {
        // Input for Deconv
        {{1, 32, 64, ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 32, 64, 64, 64}}},
        // Input for Reduce
        {{1, 8, 128, ov::Dimension::dynamic(), ov::Dimension::dynamic(), 4}, {{1, 8, 128, 128, 128, 4}}}
    }
};


const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                   ::testing::ValuesIn(netPrecisions), // netprec
                                                   ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_reduce_deconv_concat, ReduceDeconvConcatDynamicGPUTest,
                         testParams_smoke, ReduceDeconvConcatDynamicGPUTest::getTestCaseName);
} // namespace

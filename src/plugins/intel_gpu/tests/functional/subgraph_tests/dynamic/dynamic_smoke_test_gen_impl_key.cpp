// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/reduce.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        std::vector<InputShape>, // input shapes
        ov::element::Type,       // Model type
        std::string,             // Device name
        std::map<std::string, std::string> // Additional network configuration
> genImplKeyDynamicGPUTestParamsSet;

const std::vector<ov::element::Type> model_types = {
    ov::element::f16,
    ov::element::f32,
    ov::element::i32,
    ov::element::i64,
};

class GenlImplKeyDynamicGPUTest : public testing::WithParamInterface<genImplKeyDynamicGPUTestParamsSet>,
                                  virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<genImplKeyDynamicGPUTestParamsSet>& obj) {
        genImplKeyDynamicGPUTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;
        std::vector<InputShape> inputShapes;
        ov::element::Type netType;
        std::string targetDevice;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, netType, targetDevice, additionalConfig) = basicParamsSet;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
            for (const auto& actual_shape : shape.second) {
                result << ov::test::utils::partialShape2str({actual_shape}) << "_";
            }
        }
        result << "NetType=" << netType << "_";
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
        genImplKeyDynamicGPUTestParamsSet basicParamsSet = this->GetParam();
        std::vector<InputShape> inputShapes;
        ov::element::Type netType;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputShapes, netType, targetDevice, additionalConfig) = basicParamsSet;

        init_input_shapes(inputShapes);
        const auto inShapeShapeOf = inputDynamicShapes[0];
        const auto inShapeElt = inputDynamicShapes[1];
        ov::ParameterVector params;
        for (auto&& shape : {inShapeShapeOf, inShapeElt})
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netType, shape));

        auto addOp1 = ov::test::utils::make_eltwise(params[1], params[1], ov::test::utils::EltwiseTypes::ADD);
        addOp1->set_friendly_name("add1");

        auto shapeOfOp1 = std::make_shared<ov::op::v3::ShapeOf>(addOp1, ov::element::i64);
        shapeOfOp1->set_friendly_name("shapeof1");

        std::vector<int> reduce_axes = {0};
        auto reduceAxesNode1 = std::dynamic_pointer_cast<ov::Node>(
                                 std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({1}), reduce_axes));
        auto reduceOp1 = ov::test::utils::make_reduce(shapeOfOp1, reduceAxesNode1, true, ov::test::utils::ReductionType::Prod);
        reduceOp1->set_friendly_name("reduce1");

        std::vector<int64_t> shapePatternFill = {-1};
        auto reshapePatternComp1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                          ov::Shape{1}, shapePatternFill);
        auto concatOp1 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{reduceOp1, reshapePatternComp1}, 0);
        concatOp1->set_friendly_name("concat1");

        auto reshapeOp1 = std::make_shared<ov::op::v1::Reshape>(addOp1, concatOp1, false);
        reshapeOp1->set_friendly_name("reshapeOp1");

        auto addOp2 = ov::test::utils::make_eltwise(params[1], params[1], ov::test::utils::EltwiseTypes::ADD);
        addOp2->set_friendly_name("add2");

        auto shapeOfOp2 = std::make_shared<ov::op::v3::ShapeOf>(addOp2, ov::element::i64);
        shapeOfOp2->set_friendly_name("shapeof2");

        auto reduceAxesNode2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({1}), reduce_axes);
        auto reduceOp2 = ov::test::utils::make_reduce(shapeOfOp2, reduceAxesNode2, true, ov::test::utils::ReductionType::Prod);
        reduceOp2->set_friendly_name("reduce2");

        auto reshapePatternComp2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                          ov::Shape{1}, shapePatternFill);
        auto concatOp2 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{reduceOp2, reshapePatternComp2}, 0);
        concatOp2->set_friendly_name("concat2");

        auto reshapeOp2 = std::make_shared<ov::op::v1::Reshape>(addOp2, concatOp2, false);
        reshapeOp2->set_friendly_name("reshapeOp2");

        auto addOp3 = ov::test::utils::make_eltwise(reshapeOp1, reshapeOp2, ov::test::utils::EltwiseTypes::ADD);
        addOp3->set_friendly_name("add3");

        auto shapeOf3 = std::make_shared<ov::op::v3::ShapeOf>(addOp3, ov::element::i64);
        shapeOf3->set_friendly_name("shapeof3");

        ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(shapeOf3)};
        function = std::make_shared<ov::Model>(results, params, "shapeof_out");
    }
};

TEST_P(GenlImplKeyDynamicGPUTest, Inference) {
    run();
}

std::map<std::string, std::string> emptyAdditionalConfig;
const std::vector<std::vector<ov::test::InputShape>> dynInputShapes = {
    // 1D
    {
        // Input for ShapeOf
        {{ov::Dimension::dynamic()}, {{30}, {40}, {50}}},
        // Input for Add
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{3, 10}, {2, 20}, {25, 2}}}
    },
    // 2D
    {
        // Input for ShapeOf
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 10}, {2, 20}}},
        // Input for Add
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 10}, {2, 10, 2}}}
    },
    // 3D
    {
        // Input for ShapeOf
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 10, 4}, {1, 4, 12}}},
        // Input for Add
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 10, 4}, {2, 2, 12}}}
    },
    // 4D
    {
        // Input for ShapeOf
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{3, 1, 10, 4}, {2, 4, 23, 12}}},
        // Input for Add
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{30, 4}, {24, 92}}}
    }
};

const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                   ::testing::ValuesIn(model_types),
                                                   ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                   ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_impl_key, GenlImplKeyDynamicGPUTest,
                         testParams_smoke, GenlImplKeyDynamicGPUTest::getTestCaseName);
} // namespace

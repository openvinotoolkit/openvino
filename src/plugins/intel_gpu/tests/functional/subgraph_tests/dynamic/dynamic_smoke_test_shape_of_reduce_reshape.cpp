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
        ov::element::Type, // Network precision
        std::string // Device name
> shapeOfReshapeReduceDynamicGPUTestParamsSet;

const std::vector<ov::element::Type> model_types = {
    ov::element::f16,
    ov::element::f32,
    ov::element::i32,
    ov::element::i64,
};

class ShapeOfReshapeReduceDynamicGPUTest : public testing::WithParamInterface<shapeOfReshapeReduceDynamicGPUTestParamsSet>,
                                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<shapeOfReshapeReduceDynamicGPUTestParamsSet>& obj) {
        shapeOfReshapeReduceDynamicGPUTestParamsSet basicParamsSet = obj.param;
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
        shapeOfReshapeReduceDynamicGPUTestParamsSet basicParamsSet = this->GetParam();
        std::vector<InputShape> inputShapes;
        ov::element::Type model_type;
        std::tie(inputShapes, model_type, targetDevice) = basicParamsSet;

        init_input_shapes(inputShapes);
        const auto inShapeShapeOf = inputDynamicShapes[0];
        const auto inShapeElt = inputDynamicShapes[1];
        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        auto addOp = ov::test::utils::make_eltwise(params[1], params[1], ov::test::utils::EltwiseTypes::ADD);
        addOp->set_friendly_name("add");

        auto shapeOfOp1 = std::make_shared<ov::op::v3::ShapeOf>(params[0], ov::element::i64);
        shapeOfOp1->set_friendly_name("shapeof1");
        std::vector<int> reduce_axes = {0};
        auto reduceAxesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({1}), reduce_axes);
        auto reduceOp = ov::test::utils::make_reduce(shapeOfOp1, reduceAxesNode, true, ov::test::utils::ReductionType::Prod);
        reduceOp->set_friendly_name("reduce");
        std::vector<int64_t> shapePatternFill = {-1};
        auto reshapePatternComp = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, shapePatternFill);
        auto concatOp = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{reduceOp, reshapePatternComp}, 0);
        concatOp->set_friendly_name("concat");

        auto reshapeOp = std::make_shared<ov::op::v1::Reshape>(addOp, concatOp, false);

        auto shapeOf2 = std::make_shared<ov::op::v3::ShapeOf>(reshapeOp, ov::element::i64);
        shapeOf2->set_friendly_name("shapeof2");

        ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(shapeOf2)};
        function = std::make_shared<ov::Model>(results, params, "shapeof_out");
    }
};

TEST_P(ShapeOfReshapeReduceDynamicGPUTest, Inference) {
    run();
}

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
                                                   ::testing::ValuesIn(model_types), // netprec
                                                   ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_shapeof_reshape, ShapeOfReshapeReduceDynamicGPUTest,
                         testParams_smoke, ShapeOfReshapeReduceDynamicGPUTest::getTestCaseName);
} // namespace

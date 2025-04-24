// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        InputShape,
        ov::element::Type
> ShapeOfLayerGPUTestParamsSet;

class ShapeOfLayerGPUTest : public testing::WithParamInterface<ShapeOfLayerGPUTestParamsSet>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ShapeOfLayerGPUTestParamsSet> obj) {
        InputShape inputShape;
        ov::element::Type model_type;
        std::tie(inputShape, model_type) = obj.param;

        std::ostringstream result;
        result << "ShapeOfTest_";
        result << std::to_string(obj.index) << "_";
        result << "netPrec=" << model_type << "_";
        result << "IS=";
        result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << ")";
        return result.str();
    }
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        ov::element::Type model_type;
        InputShape inputShape;
        std::tie(inputShape, model_type) = this->GetParam();

        init_input_shapes({inputShape});

        outType = ov::element::i32;

        ov::ParameterVector functionParams;
        for (auto&& shape : inputDynamicShapes)
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        auto shapeOfOp = std::make_shared<ov::op::v3::ShapeOf>(functionParams[0], ov::element::i32);

        auto makeFunction = [](ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) {
            ov::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(lastNode->output(i)));

            return std::make_shared<ov::Model>(results, params, "ShapeOfLayerGPUTest");
        };

        function = makeFunction(functionParams, shapeOfOp);
    }
};

TEST_P(ShapeOfLayerGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> model_types = {
        ov::element::i32,
};

// We don't check static case, because of constant folding

// ==============================================================================
// 3D
std::vector<ov::test::InputShape> inShapesDynamic3d = {
        {
            {-1, -1, -1},
            {
                { 8, 5, 4 },
                { 8, 5, 3 },
                { 8, 5, 2 }
            }
        },
        {
            {-1, -1, -1},
            {
                { 1, 2, 4 },
                { 1, 2, 3 },
                { 1, 2, 2 }
            }
        }
};
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_3d_compareWithRefs_dynamic,
    ShapeOfLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic3d),
        ::testing::ValuesIn(model_types)),
    ShapeOfLayerGPUTest::getTestCaseName);

std::vector<ov::Shape> inShapesStatic3d = {
    { 8, 5, 4 },
    { 8, 5, 3 },
    { 8, 5, 2 },
    { 1, 2, 4 },
    { 1, 2, 3 },
    { 1, 2, 2 }
};
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_3d_compareWithRefs_static,
    ShapeOfLayerGPUTest,
    ::testing::Combine(
            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesStatic3d)),
            ::testing::Values(ov::element::i32)),
    ShapeOfLayerGPUTest::getTestCaseName);

// ==============================================================================
// 4D
std::vector<ov::test::InputShape> inShapesDynamic4d = {
        {
            {-1, -1, -1, -1},
            {
                { 8, 5, 3, 4 },
                { 8, 5, 3, 3 },
                { 8, 5, 3, 2 }
            }
        },
        {
            {-1, -1, -1, -1},
            {
                { 1, 2, 3, 4 },
                { 1, 2, 3, 3 },
                { 1, 2, 3, 2 }
            }
        }
};
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_4d_compareWithRefs_dynamic,
    ShapeOfLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic4d),
        ::testing::ValuesIn(model_types)),
    ShapeOfLayerGPUTest::getTestCaseName);

std::vector<ov::Shape> inShapesStatic4d = {
    { 8, 5, 3, 4 },
    { 8, 5, 3, 3 },
    { 8, 5, 3, 2 },
    { 1, 2, 3, 4 },
    { 1, 2, 3, 3 },
    { 1, 2, 3, 2 }
};
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_4d_compareWithRefs_static,
    ShapeOfLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesStatic4d)),
        ::testing::ValuesIn(model_types)),
    ShapeOfLayerGPUTest::getTestCaseName);

// ==============================================================================
// 5D
std::vector<ov::test::InputShape> inShapesDynamic5d = {
        {
            { -1, -1, -1, -1, -1 },
            {
                { 8, 5, 3, 2, 4 },
                { 8, 5, 3, 2, 3 },
                { 8, 5, 3, 2, 2 }
            }
        },
        {
            {-1, -1, -1, -1, -1},
            {
                { 1, 2, 3, 4, 4 },
                { 1, 2, 3, 4, 3 },
                { 1, 2, 3, 4, 2 }
            }
        }
};
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_5d_compareWithRefs_dynamic,
    ShapeOfLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic5d),
        ::testing::ValuesIn(model_types)),
    ShapeOfLayerGPUTest::getTestCaseName);

std::vector<ov::Shape> inShapesStatic5d = {
    { 8, 5, 3, 2, 4 },
    { 8, 5, 3, 2, 3 },
    { 8, 5, 3, 2, 2 },
    { 1, 2, 3, 4, 4 },
    { 1, 2, 3, 4, 3 },
    { 1, 2, 3, 4, 2 }
};
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_5d_compareWithRefs_static,
    ShapeOfLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesStatic5d)),
        ::testing::ValuesIn(model_types)),
    ShapeOfLayerGPUTest::getTestCaseName);

using ShapeOfParams = typename std::tuple<
        InputShape,            // Shape
        ov::element::Type,     // Model type
        std::string            // Device name
>;

class ShapeOfDynamicInputGPUTest : public testing::WithParamInterface<ShapeOfParams>,
                                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ShapeOfParams>& obj) {
        InputShape shapes;
        ov::element::Type model_type;
        std::string targetDevice;

        std::tie(shapes, model_type, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::partialShape2str({shapes.first}) << "_";
        for (size_t i = 0lu; i < shapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(shapes.second[i]) << "_";
            result << "}_";
        }
        result << ")_";
        result << "netPRC=" << model_type << "_";
        result << "targetDevice=" << targetDevice << "_";
        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

protected:
    void SetUp() override {
        InputShape shapes;
        ov::element::Type model_type;
        targetDevice = ov::test::utils::DEVICE_GPU;

        std::tie(shapes, model_type, targetDevice) = GetParam();

        init_input_shapes({shapes});

        auto input = std::make_shared<ov::op::v0::Parameter>(model_type, shapes.first);
        input->set_friendly_name("input_data");

        auto shape_of_01 = std::make_shared<ov::op::v3::ShapeOf>(input);
        shape_of_01->set_friendly_name("shape_of_01");

        auto shape_of_02 = std::make_shared<ov::op::v3::ShapeOf>(shape_of_01);
        shape_of_02->set_friendly_name("shape_of_02");

        auto result = std::make_shared<ov::op::v0::Result>(shape_of_02);
        result->set_friendly_name("outer_result");

        function = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{input});
        function->set_friendly_name("shape_of_test");
    }
};

TEST_P(ShapeOfDynamicInputGPUTest, Inference) {
    run();
}

const std::vector<ov::test::InputShape> dynamicshapes = {
    ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{4, 1, 1, 64, 32}, {6, 1, 1, 8, 4}, {8, 1, 1, 24, 16}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_Check, ShapeOfDynamicInputGPUTest,
                testing::Combine(
                    testing::ValuesIn(dynamicshapes),                          // input shapes
                    testing::Values(ov::element::f16),                               // network precision
                    testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),     // device type
                ShapeOfDynamicInputGPUTest::getTestCaseName);

} // namespace

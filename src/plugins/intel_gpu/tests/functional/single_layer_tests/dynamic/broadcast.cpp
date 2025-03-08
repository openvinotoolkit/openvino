// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/broadcast.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        std::vector<InputShape>,               // Shapes
        std::vector<int64_t>,                  // Target shapes
        std::vector<int64_t>,                  // Axes mapping
        ov::op::BroadcastType,                 // Broadcast mode
        ov::element::Type,                   // Network precision
        std::vector<bool>,                     // Const inputs
        std::string                            // Device name
> BroadcastLayerTestParamsSet;

class BroadcastLayerGPUTest : public testing::WithParamInterface<BroadcastLayerTestParamsSet>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BroadcastLayerTestParamsSet> obj) {
        std::vector<ov::test::InputShape> shapes;
        std::vector<int64_t> targetShapes, axesMapping;
        ov::op::BroadcastType mode;
        ov::element::Type model_type;
        std::vector<bool> isConstInputs;
        std::string deviceName;
        std::tie(shapes, targetShapes, axesMapping, mode, model_type, isConstInputs, deviceName) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : shapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "targetShape=" << ov::test::utils::vec2str(targetShapes)  << "_";
        result << "axesMapping=" << ov::test::utils::vec2str(axesMapping)  << "_";
        result << "mode=" << mode << "_";
        result << "netPrec=" << model_type << "_";
        result << "constIn=(" << (isConstInputs[0] ? "True" : "False") << "." << (isConstInputs[1] ? "True" : "False") << ")_";
        result << "trgDevice=" << deviceName;

        return result.str();
    }

protected:
    std::vector<int64_t> targetShape, axesMapping;

    void SetUp() override {
        std::vector<InputShape> shapes;
        ov::op::BroadcastType mode;
        ov::element::Type model_type;
        std::vector<bool> isConstInput;
        std::tie(shapes, targetShape, axesMapping, mode, model_type, isConstInput, targetDevice) = this->GetParam();

        bool isTargetShapeConst = isConstInput[0];
        bool isAxesMapConst = isConstInput[1];

        const auto targetShapeRank = targetShape.size();
        const auto axesMappingRank = axesMapping.size();

        if (shapes.front().first.rank() != 0) {
            inputDynamicShapes.push_back(shapes.front().first);
            if (!isTargetShapeConst) {
                inputDynamicShapes.push_back({ static_cast<int64_t>(targetShape.size()) });
            }
            if (!isAxesMapConst) {
                inputDynamicShapes.push_back({ static_cast<int64_t>(axesMapping.size()) });
            }
        }
        const size_t targetStaticShapeSize = shapes.front().second.size();
        targetStaticShapes.resize(targetStaticShapeSize);
        for (size_t i = 0lu; i < targetStaticShapeSize; ++i) {
            targetStaticShapes[i].push_back(shapes.front().second[i]);
            if (!isTargetShapeConst)
                targetStaticShapes[i].push_back({ targetShape.size() });
            if (!isAxesMapConst)
                targetStaticShapes[i].push_back({ axesMapping.size() });
        }

        ov::ParameterVector functionParams;
        if (inputDynamicShapes.empty()) {
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, targetStaticShapes.front().front()));
        } else {
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front()));
            if (!isTargetShapeConst) {
                functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[1]));
                functionParams.back()->set_friendly_name("targetShape");
            }
            if (!isAxesMapConst) {
                functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes.back()));
                functionParams.back()->set_friendly_name("axesMapping");
            }
        }
        functionParams.front()->set_friendly_name("data");

        std::shared_ptr<ov::op::v3::Broadcast> broadcastOp;
        if (mode == ov::op::BroadcastType::EXPLICIT) {
            std::shared_ptr<ov::Node> targetShapeOp;
            std::shared_ptr<ov::Node> axesMappingOp;
            if (isTargetShapeConst) {
                targetShapeOp = ov::op::v0::Constant::create(ov::element::i64, {targetShapeRank}, targetShape);
            } else {
                targetShapeOp = functionParams[1];
            }
            if (isAxesMapConst) {
                axesMappingOp = ov::op::v0::Constant::create(ov::element::i64, {axesMappingRank}, axesMapping);
            } else {
                axesMappingOp = functionParams.size() > 2 ? functionParams[2] : functionParams[1];
            }
            broadcastOp = std::make_shared<ov::op::v3::Broadcast>(functionParams[0],
                                                                targetShapeOp,
                                                                axesMappingOp,
                                                                mode);
        } else if (mode == ov::op::BroadcastType::NUMPY) {
            if (isTargetShapeConst) {
                auto targetShapeConst = ov::op::v0::Constant::create(ov::element::i64, {targetShapeRank}, targetShape);
                broadcastOp = std::make_shared<ov::op::v3::Broadcast>(functionParams[0],
                                                                      targetShapeConst,
                                                                      mode);
            } else {
                broadcastOp = std::make_shared<ov::op::v3::Broadcast>(functionParams[0],
                                                                      functionParams[1],
                                                                      mode);
            }
        }

        auto makeFunction = [](ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) {
            ov::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(lastNode->output(i)));

            return std::make_shared<ov::Model>(results, params, "BroadcastLayerGPUTest");
        };

        function = makeFunction(functionParams, broadcastOp);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0lu; i < funcInputs.size(); i++) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (funcInput.get_node()->get_friendly_name() == "targetShape") {
                tensor = ov::Tensor{ov::element::i64, targetInputStaticShapes[i]};
                auto data = tensor.data<ov::element_type_traits<ov::element::i64>::value_type>();
                for (size_t i = 0lu; i < targetShape.size(); i++) {
                    data[i] = targetShape[i];
                }
            } else if (funcInput.get_node()->get_friendly_name() == "axesMapping") {
                tensor = ov::Tensor{ov::element::i64, targetInputStaticShapes[i]};
                auto data = tensor.data<ov::element_type_traits<ov::element::i64>::value_type>();
                for (size_t i = 0lu; i < axesMapping.size(); i++) {
                    data[i] = axesMapping[i];
                }
            } else {
                if (funcInput.get_element_type().is_real()) {
                    ov::test::utils::InputGenerateData in_data;
                    in_data.start_from = 0;
                    in_data.range = 10;
                    in_data.resolution = 1000;
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
                } else {
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(BroadcastLayerGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> inputPrecisionsFloat = {
    ov::element::f32,
};

const std::vector<ov::element::Type> inputPrecisionsInt = {
    ov::element::i32,
};

const std::vector<std::vector<bool>> inputConstants = {
    {true, true},
    {false, true},
#if 0   // axes map input doesn't supported parameter input
    {true, false},
    {false, false},
#endif
};

// ==============================================================================
// 1D
const std::vector<std::vector<InputShape>> dynamicInputShapes1D_explicit = {
    {
        { {-1}, {{7}} }
    },
};
INSTANTIATE_TEST_SUITE_P(smoke_broadcast_1d_explicit_compareWithRefs_dynamic,
    BroadcastLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes1D_explicit),
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{4, 7, 3}, {2, 7, 4, 3, 6}}),
        ::testing::Values(std::vector<int64_t>{1}),
        ::testing::Values(ov::op::BroadcastType::EXPLICIT),
        ::testing::ValuesIn(inputPrecisionsFloat),
        ::testing::ValuesIn(inputConstants),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    BroadcastLayerGPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> dynamicInputShapes1D = {
    {
        { {-1}, {{1}, {7}} }
    },
};
INSTANTIATE_TEST_SUITE_P(smoke_broadcast_1d_numpy_compareWithRefs_dynamic,
    BroadcastLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes1D),
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{7}, {2, 4, 7}, {2, 3, 4, 7}}),
        ::testing::Values(std::vector<int64_t>{}),
        ::testing::Values(ov::op::BroadcastType::NUMPY),
        ::testing::ValuesIn(inputPrecisionsInt),
        ::testing::ValuesIn(inputConstants),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    BroadcastLayerGPUTest::getTestCaseName);

// ==============================================================================
// 2D
const std::vector<std::vector<InputShape>> dynamicInputShapes2D_explicit = {
    {
        { {-1, -1}, {{3, 5}} }
    }
};
INSTANTIATE_TEST_SUITE_P(smoke_broadcast_2d_explicit_compareWithRefs_dynamic,
    BroadcastLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes2D_explicit),
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{3, 4, 5}, {3, 6, 5, 7}}),
        ::testing::Values(std::vector<int64_t>{0, 2}),
        ::testing::Values(ov::op::BroadcastType::EXPLICIT),
        ::testing::ValuesIn(inputPrecisionsInt),
        ::testing::ValuesIn(inputConstants),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    BroadcastLayerGPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> dynamicInputShapes2D = {
    {
        { {-1, -1}, {{3, 1}, {3, 5}} }
    }
};
INSTANTIATE_TEST_SUITE_P(smoke_broadcast_2d_numpy_compareWithRefs_dynamic,
    BroadcastLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes2D),
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{3, 5}, {2, 3, 5}}),
        ::testing::Values(std::vector<int64_t>{}),
        ::testing::Values(ov::op::BroadcastType::NUMPY),
        ::testing::ValuesIn(inputPrecisionsFloat),
        ::testing::ValuesIn(inputConstants),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    BroadcastLayerGPUTest::getTestCaseName);

// ==============================================================================
// 3D
const std::vector<std::vector<InputShape>> dynamicInputShapes3D_explicit = {
    {
        { {-1, -1, -1}, {{4, 5, 6}} }
    },
};
INSTANTIATE_TEST_SUITE_P(smoke_broadcast_3d_explicit_compareWithRefs_dynamic,
    BroadcastLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes3D_explicit),
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{4, 5, 6}, {4, 5, 6, 2, 3}}),
        ::testing::Values(std::vector<int64_t>{0, 1, 2}),
        ::testing::Values(ov::op::BroadcastType::EXPLICIT),
        ::testing::ValuesIn(inputPrecisionsFloat),
        ::testing::ValuesIn(inputConstants),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    BroadcastLayerGPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> dynamicInputShapes3D = {
    {
        { {-1, -1, -1}, {{4, 5, 1}, {1, 5, 1}} }
    },
};
INSTANTIATE_TEST_SUITE_P(smoke_broadcast_3d_numpy_compareWithRefs_dynamic,
    BroadcastLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes3D),
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{4, 5, 6}, {2, 4, 5, 1}}),
        ::testing::Values(std::vector<int64_t>{}),
        ::testing::Values(ov::op::BroadcastType::NUMPY),
        ::testing::ValuesIn(inputPrecisionsInt),
        ::testing::ValuesIn(inputConstants),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    BroadcastLayerGPUTest::getTestCaseName);

// ==============================================================================
// 4D
const std::vector<std::vector<InputShape>> dynamicInputShapes4D_explicit = {
    {
        { {-1, -1, -1, -1}, {{1, 16, 1, 7}} }
    },
};
INSTANTIATE_TEST_SUITE_P(smoke_broadcast_4d_explicit_compareWithRefs_dynamic,
    BroadcastLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D_explicit),
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{1, 16, 2, 1, 7}, {1, 16, 2, 1, 7, 3}}),
        ::testing::Values(std::vector<int64_t>{0, 1, 3, 4}),
        ::testing::Values(ov::op::BroadcastType::EXPLICIT),
        ::testing::ValuesIn(inputPrecisionsInt),
        ::testing::ValuesIn(inputConstants),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    BroadcastLayerGPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> dynamicInputShapes4D = {
    {
        { {-1, -1, -1, -1}, {{2, 1, 1, 3}, {1, 4, 1, 3}} }
    },
};
INSTANTIATE_TEST_SUITE_P(smoke_broadcast_4d_numpy_compareWithRefs_dynamic,
    BroadcastLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D),
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{2, 4, 1, 3}, {3, 2, 2, 4, 1, 3}}),
        ::testing::Values(std::vector<int64_t>{}),
        ::testing::Values(ov::op::BroadcastType::NUMPY),
        ::testing::ValuesIn(inputPrecisionsFloat),
        ::testing::ValuesIn(inputConstants),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    BroadcastLayerGPUTest::getTestCaseName);

// ==============================================================================
// 5D
const std::vector<std::vector<InputShape>> dynamicInputShapes5D_explicit = {
    {
        { {-1, -1, -1, -1, -1}, {{2, 3, 4, 5, 6}} }
    },
};
INSTANTIATE_TEST_SUITE_P(smoke_broadcast_5d_explicit_compareWithRefs_dynamic,
    BroadcastLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes5D_explicit),
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{2, 3, 4, 5, 6}}),
        ::testing::Values(std::vector<int64_t>{0, 1, 2, 3, 4}),
        ::testing::Values(ov::op::BroadcastType::EXPLICIT),
        ::testing::ValuesIn(inputPrecisionsInt),
        ::testing::ValuesIn(inputConstants),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    BroadcastLayerGPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> dynamicInputShapes5D = {
    {
        { {-1, -1, -1, -1, -1}, {{8, 1, 1, 7, 1}, {8, 4, 1, 7, 3}} }
    },
};
INSTANTIATE_TEST_SUITE_P(smoke_broadcast_5d_numpy_compareWithRefs_dynamic,
    BroadcastLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes5D),
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{8, 4, 1, 7, 3}, {8, 4, 5, 7, 3}}),
        ::testing::Values(std::vector<int64_t>{}),
        ::testing::Values(ov::op::BroadcastType::NUMPY),
        ::testing::ValuesIn(inputPrecisionsFloat),
        ::testing::ValuesIn(inputConstants),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    BroadcastLayerGPUTest::getTestCaseName);
// ==============================================================================
// 6D
const std::vector<std::vector<InputShape>> dynamicInputShapes6D = {
    {
        { {-1, -1, -1, -1, -1, -1}, {{8, 1, 1, 7, 1, 3}, {8, 4, 1, 7, 16, 3}} }
    },
};
INSTANTIATE_TEST_SUITE_P(smoke_broadcast_6d_numpy_compareWithRefs_dynamic,
    BroadcastLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes6D),
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{8, 4, 1, 7, 16, 3}, {8, 4, 5, 7, 16, 3}}),
        ::testing::Values(std::vector<int64_t>{}),
        ::testing::Values(ov::op::BroadcastType::NUMPY),
        ::testing::ValuesIn(inputPrecisionsInt),
        ::testing::ValuesIn(inputConstants),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    BroadcastLayerGPUTest::getTestCaseName);

} // namespace

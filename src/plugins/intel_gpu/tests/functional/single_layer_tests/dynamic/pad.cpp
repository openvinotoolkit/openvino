// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/pad.hpp"

namespace {
using ov::test::InputShape;

using PadLayerGPUTestParamSet = std::tuple<
        InputShape,                                     // Input shape
        ov::element::Type,                              // Input element type
        std::vector<int64_t>,                           // padsBegin
        std::vector<int64_t>,                           // padsEnd
        float,                                          // argPadValue
        std::vector<ov::test::utils::InputLayerType>,   // for {begin, end, padValue}
        ov::op::PadMode>;                               // padMode

class PadLayerGPUTest : public testing::WithParamInterface<PadLayerGPUTestParamSet>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PadLayerGPUTestParamSet> obj) {
        const auto& [shapes, model_type, padsBegin, padsEnd, argPadValue, inputLayerTypes, padMode] = obj.param;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "Prc=" << model_type << "_";
        results << "padsBegin=" << ov::test::utils::vec2str(padsBegin) << "_";
        results << "padsEnd=" << ov::test::utils::vec2str(padsEnd) << "_";
        if (padMode == ov::op::PadMode::CONSTANT) {
            results << "Value=" << argPadValue << "_";
        }
        results << "constantInput=" << inputLayerTypes[0] << "/" << inputLayerTypes[1] << "/" << inputLayerTypes[2] << "_";
        results << "PadMode=" << padMode;

        return results.str();
    }

protected:
    std::vector<int64_t> padsBegin, padsEnd;
    float argPadValue;

    void SetUp() override {
        const auto& [shapes, _inType, _padsBegin, _padsEnd, _argPadValue, inputLayerTypes, padMode] = this->GetParam();
        inType = _inType;
        padsBegin = _padsBegin;
        padsEnd = _padsEnd;
        argPadValue = _argPadValue;

        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> inputShapes;
        inputShapes.push_back(shapes);
        if (inputLayerTypes[0] == ov::test::utils::InputLayerType::PARAMETER) {
            inputShapes.push_back(InputShape({static_cast<int64_t>(padsBegin.size())}, std::vector<ov::Shape>(shapes.second.size(), {padsBegin.size()})));
        }
        if (inputLayerTypes[1] == ov::test::utils::InputLayerType::PARAMETER) {
            inputShapes.push_back(InputShape({static_cast<int64_t>(padsEnd.size())}, std::vector<ov::Shape>(shapes.second.size(), {padsEnd.size()})));
        }

        init_input_shapes(inputShapes);

        // Add empty shape for parameter input of scalar 'pad_value'
        if (inputLayerTypes[2] == ov::test::utils::InputLayerType::PARAMETER) {
            inputDynamicShapes.push_back(ov::PartialShape({}));
            for (size_t i = 0; i < shapes.second.size(); ++i) {
                for (size_t k = 0; k < targetStaticShapes.size(); ++k) {
                    targetStaticShapes[k].push_back(ov::Shape({}));
                }
            }
        }

        ov::ParameterVector functionParams;
        functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front()));
        functionParams.front()->set_friendly_name("data");

        std::shared_ptr<ov::Node> pads_begin, pads_end, arg_pad_value;
        // padsBegin
        if (inputLayerTypes[0] == ov::test::utils::InputLayerType::PARAMETER) {
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{padsBegin.size()}));
            functionParams.back()->set_friendly_name("padsBegin");
            pads_begin = functionParams.back();
        } else {
            pads_begin = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{padsBegin.size()}, padsBegin.data());
        }

        // padsEnd
        if (inputLayerTypes[1] == ov::test::utils::InputLayerType::PARAMETER) {
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{padsEnd.size()}));
            functionParams.back()->set_friendly_name("padsEnd");
            pads_end = functionParams.back();
        } else {
            pads_end = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{padsEnd.size()}, padsEnd.data());
        }

        // argPadValue
        if (inputLayerTypes[2] == ov::test::utils::InputLayerType::PARAMETER) {
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape({})));
            functionParams.back()->set_friendly_name("padValue");
            arg_pad_value = functionParams.back();
        } else {
            arg_pad_value = std::make_shared<ov::op::v0::Constant>(inType, ov::Shape{}, &argPadValue);
        }

        auto pad = std::make_shared<ov::op::v1::Pad>(functionParams[0], pads_begin, pads_end, arg_pad_value, padMode);

        ov::ResultVector results;
        for (size_t i = 0; i < pad->get_output_size(); ++i) {
            results.push_back(std::make_shared<ov::op::v0::Result>(pad->output(i)));
        }

        function = std::make_shared<ov::Model>(results, functionParams, "PadLayerGPUTest");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0lu; i < funcInputs.size(); i++) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (funcInput.get_node()->get_friendly_name() == "padsBegin") {
                tensor = ov::Tensor{funcInput.get_element_type(), targetInputStaticShapes[i]};
                auto data = tensor.data<float>();
                for (size_t i = 0lu; i < padsBegin.size(); i++) {
                    data[i] = static_cast<float>(padsBegin[i]);
                }
            } else if (funcInput.get_node()->get_friendly_name() == "padsEnd") {
                tensor = ov::Tensor{funcInput.get_element_type(), targetInputStaticShapes[i]};
                auto data = tensor.data<float>();
                for (size_t i = 0lu; i < padsEnd.size(); i++) {
                    data[i] = static_cast<float>(padsEnd[i]);
                }
            } else if (funcInput.get_node()->get_friendly_name() == "padValue") {
                tensor = ov::Tensor{funcInput.get_element_type(), targetInputStaticShapes[i]};
                auto data = tensor.data<float>();
                data[0] = argPadValue;
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

TEST_P(PadLayerGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f32
};

const std::vector<float> argPadValue = {0.f, -1.f};

const std::vector<ov::op::PadMode> padMode = {
        ov::op::PadMode::EDGE,
        ov::op::PadMode::REFLECT,
        ov::op::PadMode::SYMMETRIC
};

const std::vector<std::vector<ov::test::utils::InputLayerType>> isConstantInput = {
    {ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::CONSTANT},
    {ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::CONSTANT},
    {ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::PARAMETER},
    {ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::PARAMETER}
};

//====================== Dynamic Shapes Tests 2D ======================
const std::vector<InputShape> inputShapesDynamic2D = {
    {{-1, -1}, {{5, 36}, {3, 16}}}
};

const std::vector<std::vector<int64_t>> padsBegin2D_Smoke = {{0, 1}, {1, 2}};
const std::vector<std::vector<int64_t>> padsEnd2D_Smoke   = {{1, 2}, {2, 1}};

INSTANTIATE_TEST_SUITE_P(
        smoke_GPUPadDynamic2DConst,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic2D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin2D_Smoke),
                ::testing::ValuesIn(padsEnd2D_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::ValuesIn(isConstantInput),
                ::testing::Values(ov::op::PadMode::CONSTANT)),
        PadLayerGPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_GPUPadDynamic2D,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic2D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin2D_Smoke),
                ::testing::ValuesIn(padsEnd2D_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(isConstantInput),
                ::testing::ValuesIn(padMode)),
        PadLayerGPUTest::getTestCaseName
);

//====================== Dynamic Shapes Tests 4D ======================
const std::vector<InputShape> inputShapesDynamic4D = {
    {{-1, -1, -1, -1}, {{5, 36, 5, 5}, {3, 16, 10, 5}}}
};

const std::vector<std::vector<int64_t>> padsBegin4D_Smoke = {{1, 2, 3, 4}, {0, 2, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd4D_Smoke   = {{2, 1, 4, 3}, {0, 0, 2, 0}};

INSTANTIATE_TEST_SUITE_P(
        GPUPadDynamic4DConst,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic4D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4D_Smoke),
                ::testing::ValuesIn(padsEnd4D_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::ValuesIn(isConstantInput),
                ::testing::Values(ov::op::PadMode::CONSTANT)),
        PadLayerGPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_GPUPadDynamic4D,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic4D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4D_Smoke),
                ::testing::ValuesIn(padsEnd4D_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(isConstantInput),
                ::testing::ValuesIn(padMode)),
        PadLayerGPUTest::getTestCaseName
);

//====================== Dynamic Shapes Tests 5D ======================
const std::vector<InputShape> inputShapesDynamic5D = {
    {{-1, -1, -1, -1, -1}, {{5, 36, 5, 5, 5}, {4, 16, 8, 5, 7}}},
};

const std::vector<std::vector<int64_t>> padsBegin5D_Smoke = {{0, 0, 2, 0, 0}, {1, 2, 3, 1, 2}};
const std::vector<std::vector<int64_t>> padsEnd5D_Smoke   = {{0, 0, 1, 0, 0}, {3, 2, 1, 2, 1}};

INSTANTIATE_TEST_SUITE_P(
        smoke_GPUPadDynamic5DConst,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic5D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Smoke),
                ::testing::ValuesIn(padsEnd5D_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::ValuesIn(isConstantInput),
                ::testing::Values(ov::op::PadMode::CONSTANT)),
        PadLayerGPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_GPUPadDynamic5D,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic5D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Smoke),
                ::testing::ValuesIn(padsEnd5D_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(isConstantInput),
                ::testing::ValuesIn(padMode)),
        PadLayerGPUTest::getTestCaseName
);
} // namespace

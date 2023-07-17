// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/single_layer/activation.hpp"

using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions  {

std::string ActivationLayerCPUTest::getTestCaseName(const testing::TestParamInfo<ActivationLayerCPUTestParamSet> &obj) {
    std::vector<InputShape> inputShapes;
    std::vector<size_t> activationShapes;
    std::pair<ActivationTypes, std::vector<float>> activationTypeAndConstValue;
    ElementType netPrecision, inPrecision, outPrecision;
    CPUSpecificParams cpuParams;
    ov::AnyMap config;
    std::tie(inputShapes, activationShapes, activationTypeAndConstValue, netPrecision, inPrecision, outPrecision, config, cpuParams) = obj.param;

    std::ostringstream result;
    result << LayerTestsDefinitions::activationNames[activationTypeAndConstValue.first] << "_";
    if (inputShapes.front().first.size() != 0) {
        result << "IS=(";
        for (const auto &shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result.seekp(-1, result.cur);
        result << ")_";
    }
    result << "TS=";
    for (const auto& shape : inputShapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "AS=" << ov::test::utils::vec2str(activationShapes) << "_";
    result << "ConstantsValue=" << ov::test::utils::vec2str(activationTypeAndConstValue.second) << "_";
    result << "netPRC=" << netPrecision << "_";
    result << "inPRC=" << inPrecision << "_";
    result << "outPRC=" << outPrecision << "_";
    result << CPUTestsBase::getTestCaseName(cpuParams);

    if (!config.empty()) {
        result << "_PluginConf{";
        for (const auto& configItem : config) {
            result << "_" << configItem.first << "=";
            configItem.second.print(result);
        }
        result << "}";
    }

    return result.str();
}

void ActivationLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    int32_t startFrom = 0;
    uint32_t range = 0;
    int32_t resolution = 0;

    if (activationType == ActivationTypes::Exp && netPrecision == ElementType::bf16) {
        startFrom = 0;
        range = 2;
        resolution = 32768;
    } else if (activationType == ActivationTypes::Acosh) {
        startFrom = 2;
        range = 2;
        resolution = 128;
    } else if (activationType == ActivationTypes::Acos ||
               activationType == ActivationTypes::Asin ||
               activationType == ActivationTypes::Atanh) {
        // range [-1. 1] is required
        startFrom = -1;
        range = 2;
        resolution = 128;
    } else {
        startFrom = 0;
        range = 15;
        resolution = 32768;
    }
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;
        if (funcInput.get_element_type().is_real()) {
            tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i],
                                                             range, startFrom, resolution);
        } else {
            tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
        }
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

void ActivationLayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    std::vector<InputShape> inputShapes;
    std::vector<size_t> activationShapes;
    std::pair<ActivationTypes, std::vector<float>> activationTypeAndConstValue;
    CPUSpecificParams cpuParams;

    std::tie(inputShapes, activationShapes, activationTypeAndConstValue, netPrecision, inType, outType, configuration, cpuParams) = this->GetParam();
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

    activationType = activationTypeAndConstValue.first;
    auto constantsValue = activationTypeAndConstValue.second;

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    selectedType = getPrimitiveType() + "_" + netPrecision.name();
#    if defined(OPENVINO_ARCH_ARM)
    if (activationType == ActivationTypes::GeluErf) // @todo tmp fallback to ref, gelu erf is disabled for 32bit ARM
        selectedType = std::string("ref_") + netPrecision.name();
#    endif
    if (activationType == ActivationTypes::GeluTanh ||  // @todo not supported by ACL, can be decomposed with ngraph transformation
        activationType == ActivationTypes::SoftSign ||  // @todo not supported by ACL, can be decomposed with ngraph transformation
        inputShapes.front().first.rank().get_length() > 5)               // @todo tmp fallback to ref, remove after 6D+ ranks are properly supported
        selectedType = std::string("ref_") + netPrecision.name();
#else
    selectedType = getPrimitiveType();
    if (activationType == ActivationTypes::Log)  // @todo tmp fallback to ref, remove after Log is supported in emitters
        selectedType = std::string("ref");

    selectedType = makeSelectedTypeStr(selectedType, netPrecision, configuration);
#endif

    init_input_shapes(inputShapes);

    auto params = ngraph::builder::makeDynamicParams(netPrecision, {inputDynamicShapes.front()});
    auto activation = ngraph::builder::makeActivation(params[0], netPrecision, activationType, activationShapes, constantsValue);
    activation->get_rt_info() = getCPUInfo();
    function = std::make_shared<ov::Model>(ov::NodeVector{activation}, params, "Activation");
}

TEST_P(ActivationLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Eltwise");
}

namespace Activation {
// list only types supported by eltwise
const std::vector<size_t> activationShapes() {
    return {};
}

const std::map<ActivationTypes, std::vector<std::vector<float>>>& activationTypes() {
    static const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes {
        {Sqrt,        {{}}},
        {Sigmoid,     {{}}},
        {Tanh,        {{}}},
        {Relu,        {{}}},
        {Exp,         {{}}},
        {Clamp,       {{-2.0f, 2.0f}}},
        {Elu,         {{0.1f}}},
        {Swish,       {{0.1f}}},
        {HSwish,      {{}}},
        {Mish,        {{}}},
        {PReLu,       {{-0.01f}}},
        {GeluErf,     {{}}},
        {GeluTanh,    {{}}},
        {SoftSign,    {{}}},
        {SoftPlus,    {{}}},
    };

    return activationTypes;
}

const std::vector<ElementType>& netPrc() {
    static const std::vector<ElementType> netPrc{ElementType::f32};

    return netPrc;
}

/* ============= Activation (1D) ============= */
const std::vector<CPUSpecificParams>& cpuParams3D() {
    static const std::vector<CPUSpecificParams> cpuParams3D {
        CPUSpecificParams({nwc}, {nwc}, {}, {}),
        CPUSpecificParams({ncw}, {ncw}, {}, {}),
    };

    return cpuParams3D;
}

const std::vector<std::vector<ov::Shape>>& basic3D() {
    static const std::vector<std::vector<ov::Shape>> basic3D {
        {{2, 4, 4}},
        {{2, 17, 5}},
    };

    return basic3D;
}

/* ============= Activation (2D) ============= */
const std::vector<CPUSpecificParams>& cpuParams4D() {
    static const std::vector<CPUSpecificParams> cpuParams4D {
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
    };

    return cpuParams4D;
}

const std::vector<std::vector<ov::Shape>>& basic4D() {
    static const std::vector<std::vector<ov::Shape>> basic4D {
        {{2, 4, 4, 1}},
        {{2, 17, 5, 4}},
    };

    return basic4D;
}

/* ============= Activation (3D) ============= */
const std::vector<CPUSpecificParams>& cpuParams5D() {
    static const std::vector<CPUSpecificParams> cpuParams5D {
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {}),
    };

    return cpuParams5D;
}

const std::vector<std::vector<ov::Shape>>& basic5D() {
    static const std::vector<std::vector<ov::Shape>> basic5D {
        {{2, 4, 3, 4, 1}},
        {{2, 17, 7, 5, 4}},
    };

    return basic5D;
}

const std::map<ActivationTypes, std::vector<std::vector<float>>>& activationTypesDynamicMath() {
    static const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesDynamicMath {
        {Log,         {{}}},
        {Sign,        {{}}},
        {Acos,        {{}}},
        {Acosh,       {{}}},
        {Asin,        {{}}},
        {Asinh,       {{}}},
        {Atan,        {{}}},
        {Atanh,       {{}}},
        {Cos,         {{}}},
        {Cosh,        {{}}},
        {Tan,         {{}}},
        {HardSigmoid, {{0.2f, 0.5f}}},
        {Selu,        {{1.6732f, 1.0507f}}},
        {Ceiling,     {{}}},
        {SoftSign,    {{}}},
    };

    return activationTypesDynamicMath;
}

const std::vector<ElementType>& netPrecisions() {
    static const std::vector<ElementType> netPrecisions {
        ElementType::f32
    };

    return netPrecisions;
}

const std::vector<CPUSpecificParams>& cpuParamsDynamicMath() {
    static const std::vector<CPUSpecificParams> cpuParamsDynamicMath{CPUSpecificParams({}, {}, {}, {})};

    return cpuParamsDynamicMath;
}

const std::vector<std::vector<InputShape>>& dynamicMathBasic() {
    static const std::vector<std::vector<InputShape>> dynamicMathBasic {
        {{{-1, -1}, {{1, 50}, {5, 128}, {3, 64}}}},
        {{{-1, -1, -1, -1, -1, -1, -1, -1}, {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 3, 2, 3, 2, 3, 2, 3}, {3, 3, 3, 3, 3, 3, 3, 3}}}},
        {{{{1, 5}, 128}, {{1, 128}, {3, 128}, {5, 128}}}},
    };

    return dynamicMathBasic;
}

} // namespace Activation
} // namespace CPULayerTestsDefinitions

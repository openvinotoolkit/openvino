// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation.hpp"
#include "gtest/gtest.h"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/node_builders/activation.hpp"
#include "shared_test_classes/single_op/activation.hpp"

using namespace CPUTestUtils;
using namespace ov::test::utils;

namespace ov {
namespace test {
std::string ActivationLayerCPUTest::getTestCaseName(const testing::TestParamInfo<ActivationLayerCPUTestParamSet> &obj) {
    std::vector<ov::test::InputShape> inputShapes;
    std::vector<size_t> activationShapes;
    std::pair<utils::ActivationTypes, std::vector<float>> activationTypeAndConstValue;
    ov::element::Type netPrecision, inPrecision, outPrecision;
    CPUTestUtils::CPUSpecificParams cpuParams;
    std::tie(inputShapes, activationShapes, activationTypeAndConstValue, netPrecision, inPrecision, outPrecision, cpuParams) = obj.param;

    std::ostringstream result;
    result << activationNames[activationTypeAndConstValue.first] << "_";
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
    result << "netPRC=" << netPrecision.to_string() << "_";
    result << "inPRC=" << inPrecision.to_string() << "_";
    result << "outPRC=" << outPrecision.to_string() << "_";
    result << CPUTestUtils::CPUTestsBase::getTestCaseName(cpuParams);

    return result.str();
}

void ActivationLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    int32_t startFrom = 0;
    uint32_t range = 0;
    int32_t resolution = 0;

    if (activationType == utils::ActivationTypes::Exp) {
        if (netPrecision == ov::element::bf16) {
            startFrom = 0;
            range = 2;
        } else {
            startFrom = -10;
            range = 25;
        }
        resolution = 32768;
    } else if (activationType == utils::ActivationTypes::Acosh) {
        startFrom = 2;
        range = 2;
        resolution = 128;
    } else if (activationType == utils::ActivationTypes::Acos ||
               activationType == utils::ActivationTypes::Asin ||
               activationType == utils::ActivationTypes::Atanh) {
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
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = startFrom;
            in_data.range = range;
            in_data.resolution = resolution;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            // cover Sign NAN test case
            if ((activationType == utils::ActivationTypes::Sign) && funcInput.get_element_type() == ov::element::f32) {
                static_cast<float*>(tensor.data())[0] = std::numeric_limits<float>::quiet_NaN();
            }
        } else {
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
        }
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

void ActivationLayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    std::vector<ov::test::InputShape> inputShapes;
    std::vector<size_t> activationShapes;
    std::pair<utils::ActivationTypes, std::vector<float>> activationTypeAndConstValue;
    ov::element::Type inPrecision, outPrecision;
    CPUTestUtils::CPUSpecificParams cpuParams;
    std::tie(inputShapes, activationShapes, activationTypeAndConstValue, netPrecision, inPrecision, outPrecision, cpuParams) = this->GetParam();
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    activationType = activationTypeAndConstValue.first;
    auto constantsValue = activationTypeAndConstValue.second;

    inType  = inPrecision;
    outType = outPrecision;
    const auto primitiveType = getPrimitiveType(activationType, inType, inputShapes);
    selectedType = primitiveType.empty() ? "" : getPrimitiveType(activationType, inType, inputShapes) + "_" + netPrecision.to_string();

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
#    if defined(OPENVINO_ARCH_ARM)
    if (activationType == utils::ActivationTypes::GeluErf) // @todo tmp fallback to ref, gelu erf is disabled for 32bit ARM
        selectedType = std::string("ref_") + netPrecision.to_string();
#    endif
    if (activationType == utils::ActivationTypes::GeluTanh ||  // @todo not supported by ACL, can be decomposed with transformation
        activationType == utils::ActivationTypes::SoftSign ||  // @todo not supported by ACL, can be decomposed with transformation
        inputShapes.front().first.rank().get_length() > 5)               // @todo tmp fallback to ref, remove after 6D+ ranks are properly supported
        selectedType = std::string("ref_") + netPrecision.to_string();
#else
    if (activationType == utils::ActivationTypes::Log)  // @todo tmp fallback to ref, remove after Log is supported in emitters
        selectedType = std::string("ref_") + netPrecision.to_string();
#endif

    init_input_shapes(inputShapes);

    auto params = std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes.front());
    auto activation = utils::make_activation(params, netPrecision, activationType, activationShapes, constantsValue);
    activation->get_rt_info() = getCPUInfo();
    function = std::make_shared<ov::Model>(ov::NodeVector{activation}, ov::ParameterVector{params}, "Activation");
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    if (netPrecision == ov::element::f32 && outPrecision == ov::element::f32) {
        abs_threshold = 8e-4;
    }
#endif
    if (netPrecision == ov::element::bf16 && outPrecision == ov::element::f32) {
        abs_threshold = 6e-2;
    }
}

std::string ActivationLayerCPUTest::getPrimitiveType(const utils::ActivationTypes& activation_type,
                                                     const ov::element::Type_t& element_type,
                                                     const std::vector<std::pair<ov::PartialShape, std::vector<ov::Shape>>>& input_shapes) const {
#if defined(OV_CPU_WITH_ACL)
#if defined(OPENVINO_ARCH_ARM64)
    if ((element_type == ov::element::f32) &&
        ((activation_type == utils::ActivationTypes::Clamp) ||
        (activation_type == utils::ActivationTypes::Exp) ||
        (activation_type == utils::ActivationTypes::Relu) ||
        (activation_type == utils::ActivationTypes::Sigmoid) ||
        (activation_type == utils::ActivationTypes::Swish) ||
        (activation_type == utils::ActivationTypes::Tanh))) {
        return "jit";
    }

    if (activation_type == utils::ActivationTypes::Mish) {
        // operation is decomposed and executed by different kernels
        return "";
    }
#endif
    return "acl";
#else
    return CPUTestsBase::getPrimitiveType();
#endif
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

const std::map<utils::ActivationTypes, std::vector<std::vector<float>>>& activationTypes() {
    static const std::map<utils::ActivationTypes, std::vector<std::vector<float>>> activationTypes {
        {Sqrt,        {{}}},
        {Sigmoid,     {{}}},
        {Tanh,        {{}}},
        {Relu,        {{}}},
        {Exp,         {{}}},
        {Clamp,       {{-2.0f, 2.0f}}},
        {Elu,         {{0.1f}}},
        {Swish,       {{0.1f}}},
        {HSwish,      {{}}},
        {PReLu,       {{-0.01f}}},
        {GeluErf,     {{}}},
        {GeluTanh,    {{}}},
        {SoftSign,    {{}}},
        {SoftPlus,    {{}}},
    };

    return activationTypes;
}

const std::vector<ov::element::Type>& netPrc() {
    static const std::vector<ov::element::Type> netPrc{ov::element::f32};

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

const std::map<utils::ActivationTypes, std::vector<std::vector<float>>>& activationTypesDynamicMath() {
    static const std::map<utils::ActivationTypes, std::vector<std::vector<float>>> activationTypesDynamicMath {
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

const std::vector<ov::element::Type>& netPrecisions() {
    static const std::vector<ov::element::Type> netPrecisions {
        ov::element::f32
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

}  // namespace Activation
}  // namespace test
}  // namespace ov

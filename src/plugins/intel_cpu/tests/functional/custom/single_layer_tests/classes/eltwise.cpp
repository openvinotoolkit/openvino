// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"

#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "gtest/gtest.h"
#include "internal_properties.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string EltwiseLayerCPUTest::getTestCaseName(testing::TestParamInfo<EltwiseLayerCPUTestParamsSet> obj) {
    EltwiseTestParams basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    bool enforceSnippets;
    std::tie(basicParamsSet, cpuParams, fusingParams, enforceSnippets) = obj.param;

    std::ostringstream result;
    result << EltwiseLayerTest::getTestCaseName(testing::TestParamInfo<EltwiseTestParams>(
                                                              basicParamsSet, 0));
    result << CPUTestsBase::getTestCaseName(cpuParams);
    result << CpuTestWithFusing::getTestCaseName(fusingParams);
        result << "_enforceSnippets=" << enforceSnippets;

    return result.str();
}

// If adopt_intervals is true then:
// 1) the generated tensor value range is limited by operation result value (especially for multiply)
// which has to be in signed/unsigned int8 type range,
// 2) start value is defined by type sign: for signed int8 it's zero to have symmetric interval.
ov::Tensor EltwiseLayerCPUTest::generate_eltwise_input(const ov::element::Type& type, const ov::Shape& shape, const bool adopt_intervals) {
    struct gen_params {
        uint32_t range;
        int32_t start_from;
        int32_t resolution;

        gen_params(uint32_t range = 10, int32_t start_from = 0, int32_t resolution = 1)
            : range(range), start_from(start_from), resolution(resolution) {}
    };

    gen_params params = gen_params();
    if (type.is_real()) {
        switch (eltwiseType) {
        case utils::EltwiseTypes::POWER:
            params = gen_params(6, -3);
        case utils::EltwiseTypes::MOD:
        case utils::EltwiseTypes::FLOOR_MOD:
            params = gen_params(2, 2, 8);
            break;
        case utils::EltwiseTypes::DIVIDE:
            params = gen_params(2, 2, 8);
            break;
        case utils::EltwiseTypes::ERF:
            params = gen_params(6, -3);
            break;
        default:
            params = gen_params(80, 0, 8);
            break;
        }
    } else {
        switch (type) {
            case ov::element::i8:
                if (adopt_intervals) {
                    params = gen_params(11 * 2, -11);
                } else {
                    params = gen_params(INT8_MAX, INT8_MIN);
                }
                break;
            case ov::element::u8:
                if (adopt_intervals) {
                    params = gen_params(15, 0);
                } else {
                    params = gen_params(UINT8_MAX, 0);
                }
                break;
            case ov::element::i16:
                params = gen_params(INT16_MAX, INT16_MIN);
                break;
            case ov::element::u16:
                params = gen_params(UINT16_MAX, 0);
                break;
            case ov::element::u32:
                if (adopt_intervals) {
                    params = gen_params(INT8_MAX, 0);
                } else {
                    // testing::internal::Random::random limitation
                    params = gen_params(INT32_MAX, 0);
                }
                break;
            default:
                if (adopt_intervals) {
                    params = gen_params(INT8_MAX, INT8_MIN);
                } else {
                    params = gen_params(INT32_MAX, INT32_MIN);
                }
                break;
        }
    }

    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = params.start_from;
    in_data.range = params.range;
    in_data.resolution = params.resolution;
    return ov::test::utils::create_and_fill_tensor(type, shape, in_data);
}

void EltwiseLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        inputs.insert({funcInput.get_node_shared_ptr(), generate_eltwise_input(
            funcInput.get_element_type(),
            targetInputStaticShapes[i],
            (funcInput.get_element_type() == element::i32) || (funcInput.get_element_type() == element::u32) ||
            (funcInput.get_element_type() == element::i8) || (funcInput.get_element_type() == element::u8))});
    }
}

void EltwiseLayerCPUTest::SetUp() {
    EltwiseTestParams basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    bool enforceSnippets;
    std::tie(basicParamsSet, cpuParams, fusingParams, enforceSnippets) = this->GetParam();
    std::vector<InputShape> shapes;
    ElementType netType;
    utils::InputLayerType secondaryInputType;
    ov::test::utils::OpType opType;
    ov::AnyMap additionalConfig;
    std::tie(shapes, eltwiseType, secondaryInputType, opType, netType, inType, outType, targetDevice, additionalConfig) = basicParamsSet;
    // we have to change model precision as well, otherwise inference precision won't affect single-node graph
    // due to enforce inference precision optimization for the eltwise as first node of the model
    if (ov::element::Type(netType).is_real() && additionalConfig.count(ov::hint::inference_precision.name())) {
        netType = additionalConfig[ov::hint::inference_precision.name()].as<ov::element::Type>();
    }

    if (ElementType::bf16 == netType) {
        rel_threshold = 2e-2f;
    } else if (ElementType::i32 == netType) {
        abs_threshold = 0;
    }

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    shapes.resize(2);
    switch (opType) {
    case ov::test::utils::OpType::SCALAR: {
        std::vector<ov::Shape> identityShapes(shapes[0].second.size(), {1});
        shapes[1] = {{}, identityShapes};
        break;
    }
    case ov::test::utils::OpType::VECTOR:
        if (shapes[1].second.empty()) {
            shapes[1] = shapes[0];
        }
        break;
    default:
        FAIL() << "Unsupported Secondary operation type";
    }

    init_input_shapes(shapes);
    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    updateSelectedType(
        getPrimitiveType(eltwiseType, netType, shapes),
        netType,
        configuration);
    // selectedType = makeSelectedTypeStr(getPrimitiveType(), netType);
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    if (eltwiseType == utils::POWER) {
        selectedType = std::regex_replace(selectedType, std::regex("acl"), "ref");
    }
#endif

    if (enforceSnippets) {
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK));
    } else {
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
    }
    ov::ParameterVector parameters{std::make_shared<ov::op::v0::Parameter>(netType, inputDynamicShapes.front())};
    std::shared_ptr<ov::Node> secondaryInput;
    if (eltwiseType != utils::EltwiseTypes::BITWISE_NOT) {
        switch (secondaryInputType) {
            case utils::InputLayerType::PARAMETER: {
                auto param = std::make_shared<ov::op::v0::Parameter>(netType, inputDynamicShapes.back());
                secondaryInput = param;
                parameters.push_back(param);
                break;
            }
            case utils::InputLayerType::CONSTANT: {
                auto pShape = inputDynamicShapes.back();
                ov::Shape shape;
                if (pShape.is_static()) {
                    shape = pShape.get_shape();
                } else {
                    ASSERT_TRUE(pShape.rank().is_static());
                    shape = std::vector<size_t>(pShape.rank().get_length(), 1);
                    for (size_t i = 0; i < pShape.size(); ++i) {
                        if (pShape[i].is_static()) {
                            shape[i] = pShape[i].get_length();
                        }
                    }
                }

                auto data_tensor = generate_eltwise_input(
                    netType,
                    shape,
                    (netType == element::i32) || (netType == element::u32) ||
                    (netType == element::i8) || (netType == element::u8));
                if ((netType == ElementType::i8) || (netType == ElementType::u8)) {
                    auto data_ptr = reinterpret_cast<uint8_t*>(data_tensor.data());
                    std::vector<uint8_t> data(data_ptr, data_ptr + ov::shape_size(shape));
                    secondaryInput = ov::op::v0::Constant::create(netType, shape, data);
                } else if ((netType == ElementType::i16) || (netType == ElementType::u16)) {
                    auto data_ptr = reinterpret_cast<uint16_t*>(data_tensor.data());
                    std::vector<uint16_t> data(data_ptr, data_ptr + ov::shape_size(shape));
                    secondaryInput = ov::op::v0::Constant::create(netType, shape, data);
                } else if ((netType == ElementType::i32) || (netType == ElementType::u32)) {
                    auto data_ptr = reinterpret_cast<uint32_t*>(data_tensor.data());
                    std::vector<uint32_t> data(data_ptr, data_ptr + ov::shape_size(shape));
                    secondaryInput = ov::op::v0::Constant::create(netType, shape, data);
                } else if (netType == ElementType::f16) {
                    auto data_ptr = reinterpret_cast<ov::float16*>(data_tensor.data());
                    std::vector<ov::float16> data(data_ptr, data_ptr + ov::shape_size(shape));
                    secondaryInput = ov::op::v0::Constant::create(netType, shape, data);
                } else {
                    auto data_ptr = reinterpret_cast<float*>(data_tensor.data());
                    std::vector<float> data(data_ptr, data_ptr + ov::shape_size(shape));
                    secondaryInput = ov::op::v0::Constant::create(netType, shape, data);
                }
                break;
            }
            default: {
                FAIL() << "Unsupported InputLayerType";
            }
        }
    }
    auto eltwise = utils::make_eltwise(parameters[0], secondaryInput, eltwiseType);
    function = makeNgraphFunction(netType, parameters, eltwise, "Eltwise");
}

std::string EltwiseLayerCPUTest::getPrimitiveType(const utils::EltwiseTypes& eltwise_type,
                                                  const ov::element::Type_t& element_type,
                                                  const std::vector<std::pair<ov::PartialShape, std::vector<ov::Shape>>>& input_shapes) const {
#if defined(OV_CPU_WITH_ACL)
#if defined(OPENVINO_ARCH_ARM64)
    if ((eltwise_type == utils::EltwiseTypes::ADD) ||
       (eltwise_type == utils::EltwiseTypes::MULTIPLY) ||
       (eltwise_type == utils::EltwiseTypes::SUBTRACT) ||
       (eltwise_type == utils::EltwiseTypes::DIVIDE) ||
       (eltwise_type == utils::EltwiseTypes::MOD)) {
        return "jit";
    }
#endif
    if (eltwise_type == utils::EltwiseTypes::MOD) {
        return "ref";
    } else {
        return "acl";
    }
#elif defined(OV_CPU_WITH_SHL)
    if ((eltwise_type == utils::EltwiseTypes::ADD) ||
        (eltwise_type == utils::EltwiseTypes::SUBTRACT) ||
        (eltwise_type == utils::EltwiseTypes::MULTIPLY) ||
        (eltwise_type == utils::EltwiseTypes::DIVIDE)) {
        return "shl";
    } else {
        return "ref";
    }
#else
    return CPUTestsBase::getPrimitiveType();
#endif
}

TEST_P(EltwiseLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

namespace Eltwise {
const std::vector<ov::AnyMap>& additional_config() {
    static const std::vector<ov::AnyMap> additionalConfig = {
        {{ov::hint::inference_precision.name(), ov::element::f32}},
        {{ov::hint::inference_precision.name(), ov::element::f16}}
    };
    return additionalConfig;
}

const std::vector<ElementType>& netType() {
    static const std::vector<ElementType> netType = {
        ElementType::f32};
    return netType;
}

const std::vector<ov::test::utils::OpType>& opTypes() {
    static const std::vector<ov::test::utils::OpType> opTypes = {
        ov::test::utils::OpType::VECTOR,
    };
    return opTypes;
}

const std::vector<utils::EltwiseTypes>& eltwiseOpTypesBinInp() {
    static const std::vector<utils::EltwiseTypes> eltwiseOpTypesBinInp = {
        utils::EltwiseTypes::ADD,
        utils::EltwiseTypes::MULTIPLY,
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
        utils::EltwiseTypes::SUBTRACT,                // TODO: Fix CVS-105430
        utils::EltwiseTypes::DIVIDE,                  // TODO: Fix CVS-105430
        utils::EltwiseTypes::FLOOR_MOD,               // TODO: Fix CVS-111875
#endif
        utils::EltwiseTypes::SQUARED_DIFF,
        utils::EltwiseTypes::MOD,
    };
    return eltwiseOpTypesBinInp;
}

const std::vector<utils::EltwiseTypes>& eltwiseOpTypesBinInpSnippets() {
    static const std::vector<utils::EltwiseTypes> eltwiseOpTypesBinInp = {
        utils::EltwiseTypes::ADD,
        utils::EltwiseTypes::MULTIPLY,
        utils::EltwiseTypes::MOD,
    };
    return eltwiseOpTypesBinInp;
}

const std::vector<utils::EltwiseTypes>& eltwiseOpTypesDiffInp() {
    static const std::vector<utils::EltwiseTypes> eltwiseOpTypesDiffInp = { // Different number of input nodes depending on optimizations
        utils::EltwiseTypes::POWER,
        // utils::EltwiseTypes::MOD // Does not execute because of transformations
    };
    return eltwiseOpTypesDiffInp;
}

const std::vector<utils::EltwiseTypes>& eltwiseOpTypesBinDyn() {
    static const std::vector<utils::EltwiseTypes> eltwiseOpTypesBinDyn = {
        utils::EltwiseTypes::ADD,
        utils::EltwiseTypes::MULTIPLY,
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) // TODO: Fix CVS-105430
        utils::EltwiseTypes::SUBTRACT,
#endif
        utils::EltwiseTypes::SQUARED_DIFF,
    };
    return eltwiseOpTypesBinDyn;
}

const std::vector<CPUSpecificParams>& cpuParams_4D() {
    static const std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw, nchw}, {nchw}, {}, {})
    };
    return cpuParams_4D;
}

const std::vector<CPUSpecificParams>& cpuParams_4D_Planar() {
    static const std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nchw, nchw}, {nchw}, {}, {})
    };
    return cpuParams_4D;
}

const std::vector<CPUSpecificParams>& cpuParams_4D_PerChannel() {
    static const std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, {}),
    };
    return cpuParams_4D;
}

const std::vector<CPUSpecificParams>& cpuParams_5D() {
    static const std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({ndhwc, ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw, ncdhw}, {ncdhw}, {}, {})
    };
    return cpuParams_5D;
}

const std::vector<CPUSpecificParams>& cpuParams_5D_Planar() {
    static const std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({ncdhw, ncdhw}, {ncdhw}, {}, {})
    };
    return cpuParams_5D;
}

const std::vector<CPUSpecificParams>& cpuParams_5D_PerChannel() {
    static const std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({ndhwc, ndhwc}, {ndhwc}, {}, {}),
    };
    return cpuParams_5D;
}

const std::vector<std::vector<ov::Shape>>& inShapes_4D() {
    static const std::vector<std::vector<ov::Shape>> inShapes_4D = {
        {{2, 4, 4, 1}},
        {{2, 17, 5, 4}},
        {{2, 17, 5, 4}, {1, 17, 1, 1}},
        {{2, 17, 5, 1}, {1, 17, 1, 4}},
    };
    return inShapes_4D;
}

const std::vector<std::vector<ov::Shape>>& inShapes_5D() {
    static const std::vector<std::vector<ov::Shape>> inShapes_5D = {
        {{2, 4, 3, 4, 1}},
        {{2, 17, 7, 5, 4}},
        {{2, 17, 6, 5, 4}, {1, 17, 6, 1, 1}},
        {{2, 17, 6, 5, 1}, {1, 17, 1, 1, 4}},
    };
    return inShapes_5D;
}

const std::vector<utils::EltwiseTypes>& eltwiseOpTypesI32() {
    static const std::vector<utils::EltwiseTypes> eltwiseOpTypesI32 = {
        utils::EltwiseTypes::ADD,
        utils::EltwiseTypes::MULTIPLY,
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) // TODO: Fix CVS-105430
        utils::EltwiseTypes::SUBTRACT,
        utils::EltwiseTypes::DIVIDE,
#endif
        utils::EltwiseTypes::SQUARED_DIFF,
    };
    return eltwiseOpTypesI32;
}

const std::vector<utils::InputLayerType>& secondaryInputTypes() {
    static const std::vector<utils::InputLayerType> secondaryInputTypes = {
        utils::InputLayerType::CONSTANT,
        utils::InputLayerType::PARAMETER,
    };
    return secondaryInputTypes;
}

const std::vector<std::vector<ov::Shape>>& inShapes_4D_1D() {
    static const std::vector<std::vector<ov::Shape>> inShapes_4D_1D = {
        {{2, 17, 5, 4}, {4}},
        {{1, 3, 3, 3}, {3}},
    };
    return inShapes_4D_1D;
}

const std::vector<CPUSpecificParams> & cpuParams_4D_1D_Constant_mode() {
    static const std::vector<CPUSpecificParams> cpuParams_4D_1D_Constant_mode = {
        CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw, nchw}, {nchw}, {}, {})
    };
    return cpuParams_4D_1D_Constant_mode;
}

const std::vector<CPUSpecificParams>& cpuParams_4D_1D_Parameter_mode() {
    static const std::vector<CPUSpecificParams> cpuParams_4D_1D_Parameter_mode = {
        CPUSpecificParams({nchw, x}, {nchw}, {}, {})
    };
    return cpuParams_4D_1D_Parameter_mode;
}

const std::vector<std::vector<ov::Shape>>& inShapes_5D_1D() {
    static const std::vector<std::vector<ov::Shape>> inShapes_5D_1D = {
        {{2, 17, 5, 4, 10}, {10}},
        {{1, 3, 3, 3, 3}, {3}},
    };
    return inShapes_5D_1D;
}

const std::vector<CPUSpecificParams>& cpuParams_5D_1D_parameter() {
    static const std::vector<CPUSpecificParams> cpuParams_5D_1D_parameter = {
        CPUSpecificParams({ncdhw, x}, {ncdhw}, {}, {})
    };
    return cpuParams_5D_1D_parameter;
}

const std::vector<InputShape>& inShapes_4D_dyn_param() {
    static const std::vector<InputShape> inShapes_4D_dyn_param = {
        {
            // dynamic
            {-1, {2, 15}, -1, -1},
            // target
            {
                {3, 2, 1, 1},
                {1, 7, 5, 1},
                {3, 3, 4, 11},
            }
        },
        {
            // dynamic
            {-1, {2, 25}, -1, -1},
            // target
            {
                {1, 2, 5, 1},
                {3, 7, 1, 10},
                {3, 3, 4, 11}
            }
        }
    };
    return inShapes_4D_dyn_param;
}

const std::vector<InputShape>& inShapes_5D_dyn_param() {
    static const std::vector<InputShape> inShapes_5D_dyn_param = {
        {
            // dynamic
            {-1, {2, 15}, -1, -1, -1},
            // target
            {
                {3, 2, 1, 1, 1},
                {1, 7, 5, 1, 12},
                {3, 3, 4, 11, 6},
            }
        },
        {
            // dynamic
            {-1, {2, 25}, -1, -1, -1},
            // target
            {
                {1, 2, 5, 1, 5},
                {3, 7, 1, 10, 1},
                {3, 3, 4, 11, 6}
            }
        }
    };
    return inShapes_5D_dyn_param;
}

const std::vector<InputShape>& inShapes_5D_dyn_const() {
    static const std::vector<InputShape> inShapes_5D_dyn_const = {
        {
            // dynamic
            {3, 2, -1, -1, -1},
            // target
            {
                {3, 2, 1, 1, 1},
                {3, 2, 5, 1, 7},
                {3, 2, 1, 6, 1},
                {3, 2, 4, 11, 2},
            }
        },
    };
    return inShapes_5D_dyn_const;
}

const std::vector<std::vector<InputShape>>& inShapes_4D_dyn_const() {
    static const std::vector<std::vector<InputShape>> inShapes_4D_dyn_const = {
        {
            {
                // dynamic
                {3, 2, -1, -1},
                // target
                {
                    {3, 2, 1, 1},
                    {3, 2, 5, 1},
                    {3, 2, 1, 6},
                    {3, 2, 4, 11},
                }
            }
        },
        {
            {
                // dynamic
                {{1, 10}, 2, 5, 6},
                // target
                {
                    {3, 2, 5, 6},
                    {1, 2, 5, 6},
                    {2, 2, 5, 6},
                }
            }
        },
    };
    return inShapes_4D_dyn_const;
}

const std::vector<std::vector<ov::Shape>>& inShapes_fusing_4D() {
        static const std::vector<std::vector<ov::Shape>> inShapes_4D_fusing = {
                {{1, 3, 16, 16}, {1}},
        };
        return inShapes_4D_fusing;
}

const std::vector<std::vector<ov::Shape>>& inShapes_fusing_5D() {
        static const std::vector<std::vector<ov::Shape>> inShapes_5D_fusing = {
                {{1, 3, 16, 16, 16}, {1}},
        };
        return inShapes_5D_fusing;
}

const std::vector<CPUSpecificParams>& cpuParams_5D_1D_constant() {
    static const std::vector<CPUSpecificParams> cpuParams_5D_1D_constant = {
        CPUSpecificParams({ndhwc, ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw, ncdhw}, {ncdhw}, {}, {})
    };
    return cpuParams_5D_1D_constant;
}

const std::vector<bool>& enforceSnippets() {
    static const std::vector<bool> enforceSnippets = { false, true };
    return enforceSnippets;
}

}  // namespace Eltwise
}  // namespace test
}  // namespace ov

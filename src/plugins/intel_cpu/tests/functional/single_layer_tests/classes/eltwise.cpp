// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

std::string EltwiseLayerCPUTest::getTestCaseName(testing::TestParamInfo<EltwiseLayerCPUTestParamsSet> obj) {
    subgraph::EltwiseTestParams basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    std::tie(basicParamsSet, cpuParams, fusingParams) = obj.param;

    std::ostringstream result;
    result << subgraph::EltwiseLayerTest::getTestCaseName(testing::TestParamInfo<subgraph::EltwiseTestParams>(
            basicParamsSet, 0));
    result << CPUTestsBase::getTestCaseName(cpuParams);
    result << CpuTestWithFusing::getTestCaseName(fusingParams);

    return result.str();
}

ov::Tensor EltwiseLayerCPUTest::generate_eltwise_input(const ov::element::Type& type, const ov::Shape& shape, size_t in_idx) {
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
            case ngraph::helpers::EltwiseTypes::POWER:
                params = gen_params(6, -3);
            case ngraph::helpers::EltwiseTypes::MOD:
            case ngraph::helpers::EltwiseTypes::FLOOR_MOD:
                params = gen_params(2, 2, 8);
                break;
            case ngraph::helpers::EltwiseTypes::DIVIDE:
                params = gen_params(2, 2, 8);
                break;
            case ngraph::helpers::EltwiseTypes::ERF:
                params = gen_params(6, -3);
                break;
            default:
                params = gen_params(80, 0, 8);
                break;
        }
    } else if (type == ov::element::i64 || type == ov::element::u64) {
        uint64_t range = uint64_t(INT64_MAX);
        int64_t start_from = INT64_MIN / 2;

        // The u64 data type is processed as i64 in the plugin, but reference is processed as is,
        // so need to restrict input range for u64 to match the reference.
        if (type == ov::element::u64) {
            start_from = 0ll;
            range = INT64_MAX / 2;
        }

        switch (eltwiseType) {
            case ngraph::helpers::EltwiseTypes::MULTIPLY:
                if (type == ov::element::u64) {
                    range = 1000llu;
                } break;
            case ngraph::helpers::EltwiseTypes::SUBTRACT:
                if (type == ov::element::u64 && in_idx == 0) {
                    start_from = INT64_MAX / 2;
                } break;
            case ngraph::helpers::EltwiseTypes::SQUARED_DIFF:
                if (type == ov::element::u64) {
                    range = static_cast<uint64_t>(pow(2, 31));
                } break;
            case ngraph::helpers::EltwiseTypes::DIVIDE:
            case ngraph::helpers::EltwiseTypes::FLOOR_MOD: {
                    // The division is performed in the f64 precision in the plugin,
                    // so need to restrict input range by fraction size to match the reference.
                    if (type == ov::element::i64) {
                        start_from = -static_cast<int64_t>(pow(2, 52));
                        range = static_cast<uint64_t>(pow(2, 53));
                    } else {
                        range = static_cast<uint64_t>(pow(2, 52));
                    }
                } break;
            default:
                break;
        }

        auto tensor = ov::Tensor{type, shape};
        if (type == ov::element::i64) {
            ov::test::utils::fill_data_random(tensor.data<int64_t>(), tensor.get_size(), range, start_from);
        } else {
            ov::test::utils::fill_data_random(tensor.data<uint64_t>(), tensor.get_size(), range, start_from);
        }
        return tensor;
    } else if (type == ov::element::i32) {
        switch (eltwiseType) {
            case ngraph::helpers::EltwiseTypes::DIVIDE:
            case ngraph::helpers::EltwiseTypes::FLOOR_MOD: {
                // The i32 data division is performed in the f32 precision,
                // so need to restrict input range by fraction size to match the reference.
                const int32_t start_from = pow(2, 23);
                params = gen_params(2 * start_from, -start_from);
            } break;
            default:
                params = gen_params(INT32_MAX, INT32_MIN / 2);
                break;
        }
    } else {
        OPENVINO_THROW("Unsupported data type.");
    }

    return utils::create_and_fill_tensor(type, shape, params.range, params.start_from, params.resolution);
}

void EltwiseLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        inputs.insert({funcInput.get_node_shared_ptr(), generate_eltwise_input(funcInput.get_element_type(), targetInputStaticShapes[i], i)});
    }
}

void EltwiseLayerCPUTest::SetUp() {
    subgraph::EltwiseTestParams basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    std::tie(basicParamsSet, cpuParams, fusingParams) = this->GetParam();
    std::vector<InputShape> shapes;
    ElementType netType;
    ngraph::helpers::InputLayerType secondaryInputType;
    ov::test::utils::OpType opType;
    Config additional_config;
    std::tie(shapes, eltwiseType, secondaryInputType, opType, netType, inType, outType, targetDevice, configuration) = basicParamsSet;

    if (ElementType::bf16 == netType) {
        rel_threshold = 2e-2f;
    } else if (ElementType::i32 == netType) {
        abs_threshold = 0;
    }

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    selectedType = makeSelectedTypeStr(getPrimitiveType(), netType, configuration);
    #if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
        if (eltwiseType == POWER) {
            selectedType = std::regex_replace(selectedType, std::regex("acl"), "ref");
        }
    #endif

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

    configuration.insert(additional_config.begin(), additional_config.end());
    auto parameters = ngraph::builder::makeDynamicParams(netType, {inputDynamicShapes.front()});
    std::shared_ptr<ov::Node> secondaryInput;
    if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
        secondaryInput = ngraph::builder::makeDynamicParams(netType, {inputDynamicShapes.back()}).front();
        parameters.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(secondaryInput));
    } else {
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

        if (netType == ElementType::i32) {
            auto data_tensor = generate_eltwise_input(netType, shape);
            auto data_ptr = reinterpret_cast<int32_t*>(data_tensor.data());
            std::vector<int32_t> data(data_ptr, data_ptr + ov::shape_size(shape));
            secondaryInput = ngraph::builder::makeConstant(netType, shape, data);
        } else if (netType == ElementType::i64 || netType == ElementType::u64) {
            auto data_tensor = generate_eltwise_input(netType, shape, 1llu);
            auto data_ptr = reinterpret_cast<int64_t*>(data_tensor.data());
            std::vector<int64_t> data(data_ptr, data_ptr + ov::shape_size(shape));
            secondaryInput = ngraph::builder::makeConstant(netType, shape, data);
        } else if (netType == ElementType::f32 || netType == ElementType::bf16) {
            auto data_tensor = generate_eltwise_input(ElementType::f32, shape);
            auto data_ptr = reinterpret_cast<float*>(data_tensor.data());
            std::vector<float> data(data_ptr, data_ptr + ov::shape_size(shape));
            secondaryInput = ngraph::builder::makeConstant(netType, shape, data);
        } else {
            OPENVINO_THROW("Unsupported data type.");
        }
    }
    auto eltwise = ngraph::builder::makeEltwise(parameters[0], secondaryInput, eltwiseType);
    function = makeNgraphFunction(netType, parameters, eltwise, "Eltwise");
}

TEST_P(EltwiseLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

namespace Eltwise {

const ov::AnyMap& additional_config() {
        static const ov::AnyMap additional_config;
        return additional_config;
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

const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesBinInp() {
        static const std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesBinInp = {
                ngraph::helpers::EltwiseTypes::ADD,
                ngraph::helpers::EltwiseTypes::MULTIPLY,
        #if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
                ngraph::helpers::EltwiseTypes::SUBTRACT,                // TODO: Fix CVS-105430
                ngraph::helpers::EltwiseTypes::DIVIDE,                  // TODO: Fix CVS-105430
                ngraph::helpers::EltwiseTypes::FLOOR_MOD,               // TODO: Fix CVS-111875
        #endif
                ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
        };
        return eltwiseOpTypesBinInp;
}

const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesDiffInp() {
        static const std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesDiffInp = { // Different number of input nodes depending on optimizations
                ngraph::helpers::EltwiseTypes::POWER,
                // ngraph::helpers::EltwiseTypes::MOD // Does not execute because of transformations
        };
        return eltwiseOpTypesDiffInp;
}

const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesBinDyn() {
        static const std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesBinDyn = {
                ngraph::helpers::EltwiseTypes::ADD,
                ngraph::helpers::EltwiseTypes::MULTIPLY,
        #if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) // TODO: Fix CVS-105430
                ngraph::helpers::EltwiseTypes::SUBTRACT,
        #endif
                ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
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

const std::vector<CPUSpecificParams>& cpuParams_5D() {
        static const std::vector<CPUSpecificParams> cpuParams_5D = {
                CPUSpecificParams({ndhwc, ndhwc}, {ndhwc}, {}, {}),
                CPUSpecificParams({ncdhw, ncdhw}, {ncdhw}, {}, {})
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

const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesI32() {
        static const std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesI32 = {
                ngraph::helpers::EltwiseTypes::ADD,
                ngraph::helpers::EltwiseTypes::MULTIPLY,
        #if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) // TODO: Fix CVS-105430
                ngraph::helpers::EltwiseTypes::SUBTRACT,
                ngraph::helpers::EltwiseTypes::DIVIDE,
        #endif
                ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
        };
        return eltwiseOpTypesI32;
}

const std::vector<ngraph::helpers::InputLayerType>& secondaryInputTypes() {
        static const std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
                ngraph::helpers::InputLayerType::CONSTANT,
                ngraph::helpers::InputLayerType::PARAMETER,
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

const std::vector<CPUSpecificParams>& cpuParams_5D_1D_constant() {
    static const std::vector<CPUSpecificParams> cpuParams_5D_1D_constant = {
        CPUSpecificParams({ndhwc, ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw, ncdhw}, {ncdhw}, {}, {})
    };
    return cpuParams_5D_1D_constant;
}

} // namespace Eltwise
} // namespace CPULayerTestsDefinitions

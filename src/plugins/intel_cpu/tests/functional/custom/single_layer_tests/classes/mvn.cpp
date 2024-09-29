// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn.hpp"
#include "gtest/gtest.h"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

using namespace CPUTestUtils;
using namespace ov::intel_cpu;

namespace ov {
namespace test {

std::string MvnLayerCPUTest::getTestCaseName(testing::TestParamInfo<MvnLayerCPUTestParamSet> obj) {
    basicCpuMvnParams basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    ElementType inputPrecision, outputPrecision;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, cpuParams, fusingParams, inputPrecision, outputPrecision, additionalConfig) = obj.param;

    InputShape inputShapes;
    ElementType netPrecision;
    ov::AxisSet axes;
    bool acrossChanels, normalizeVariance;
    double eps;
    std::tie(inputShapes, netPrecision, axes, acrossChanels, normalizeVariance, eps) = basicParamsSet;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
    result << "TS=";
    for (const auto& shape : inputShapes.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "Precision=" << netPrecision << "_";
    if (!axes.empty()) {
        result << "ReductionAxes=" << ov::test::utils::vec2str(axes.to_vector()) << "_";
    } else {
        result << "AcrossChannels=" << (acrossChanels ? "TRUE" : "FALSE") << "_";
    }
    result << "NormalizeVariance=" << (normalizeVariance ? "TRUE" : "FALSE") << "_";
    result << "Epsilon=" << eps;
    result << "_"
           << "CNNInpPrc=" << inputPrecision;
    result << "_"
           << "CNNOutPrc=" << outputPrecision;

    if (!additionalConfig.empty()) {
        result << "_PluginConf";
        for (auto& item : additionalConfig) {
            result << "_" << item.first << "=" << item.second.as<std::string>();
        }
    }

    result << CPUTestsBase::getTestCaseName(cpuParams);

    result << CpuTestWithFusing::getTestCaseName(fusingParams);

    return result.str();
}

void MvnLayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    basicCpuMvnParams basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    ElementType inPrc;
    ElementType outPrc;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, cpuParams, fusingParams, inPrc, outPrc, additionalConfig) = this->GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    InputShape inputShapes;
    ElementType netPrecision;
    ov::AxisSet axes;
    bool normalizeVariance;
    double eps;
    std::tie(inputShapes, netPrecision, axes, acrossChanels, normalizeVariance, eps) = basicParamsSet;

    init_input_shapes({inputShapes});

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes)
        params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));

    std::shared_ptr<ov::op::v0::MVN> mvn;
    if (!axes.empty()) {
        mvn = std::make_shared<ov::op::v0::MVN>(params[0], axes, normalizeVariance, eps);
    } else {
        mvn = std::make_shared<ov::op::v0::MVN>(params[0], acrossChanels, normalizeVariance, eps);

        // OpenVINO MVN implementation implicitly adds 0th dimension to reduction axes set which is not valid behavior
        ov::AxisSet axes;
        const size_t startAxis = acrossChanels ? 1 : 2;
        const size_t numOfDims = params[0]->output(0).get_partial_shape().size();
        for (size_t i = startAxis; i < numOfDims; i++)
            axes.insert(i);
        mvn->set_reduction_axes(axes);
    }

    rel_threshold = 5e-4;
    if (one_of(additionalConfig[ov::hint::inference_precision.name()], ov::element::f16, ov::element::bf16)) {
        rel_threshold = 1e-2;
        abs_threshold = .03f;
    }
    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    updateSelectedType(getPrimitiveType(), netPrecision, configuration);

    function = makeNgraphFunction(netPrecision, params, mvn, "mvn");
}

TEST_P(MvnLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "MVN");
}

namespace MVN {
const std::vector<ov::AnyMap>& additionalConfig() {
    static const std::vector<ov::AnyMap> additionalConfig = {
        {{ov::hint::inference_precision.name(), ov::element::f32}},
        {{ov::hint::inference_precision.name(), ov::element::f16}}
    };
    return additionalConfig;
}

const std::vector<InputShape>& inputShapes_1D() {
    static const std::vector<InputShape> inputShapes_1D = {
        { {}, {{5}}},
        { {}, {{16}}},
        {
            // dynamic
            {-1},
            // target
            {
                {2},
                {16},
                {1},
                {2}
            }
        },
        {
            // dynamic
            {{1, 20}},
            // target
            {
                {1},
                {16},
                {4},
                {16}
            }
        }
    };
    return inputShapes_1D;
}

const std::vector<InputShape>& inputShapes_2D() {
    static const std::vector<InputShape> inputShapes_2D = {
        { {}, {{1, 32}}},
        { {}, {{16, 64}}},

        {
            // dynamic
            {-1, -1},
            // target
            {
                {2, 16},
                {4, 16},
                {1, 16},
                {4, 16}
            }
        },
        {
            // dynamic
            {{1, 5}, {1, 20}},
            // target
            {
                {1, 1},
                {2, 16},
                {4, 16},
                {2, 16}
            }
        }
    };
    return inputShapes_2D;
}

const std::vector<InputShape>& inputShapes_3D() {
    static const std::vector<InputShape> inputShapes_3D = {
        { {}, {{1, 32, 17}}},
        { {}, {{1, 37, 9}}},
        { {}, {{1, 16, 4}}},
        {
            // dynamic
            {-1, -1, -1},
            // target
            {
                {2, 16, 6},
                {4, 16, 2},
                {2, 16, 6},
                {4, 16, 2}
            }
        },
        {
            // dynamic
            {{1, 5}, {1, 20}, {1, 7}},
            // target
            {
                {1, 1, 1},
                {2, 16, 6},
                {4, 16, 2},
                {2, 16, 6}
            }
        }
    };
    return inputShapes_3D;
}

const std::vector<InputShape>& inputShapes_4D() {
    static const std::vector<InputShape> inputShapes_4D = {
        { {}, {{1, 16, 5, 8}}},
        { {}, {{2, 19, 5, 10}}},
        { {}, {{7, 32, 2, 8}}},
        { {}, {{5, 8, 3, 5}}},
        { {}, {{1, 2, 7, 5}}},
        { {}, {{1, 4, 5, 5}}},
        { {}, {{1, 7, 3, 5}}},
        { {}, {{1, 15, 9, 5}}},
        { {}, {{4, 41, 6, 9}}},
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {2, 16, 10, 6},
                {4, 16, 2, 2},
                {2, 16, 10, 6},
                {4, 16, 2, 2}
            }
        },
        {
            // dynamic
            {{1, 5}, {1, 20}, {1, 10}, {1, 7}},
            // target
            {
                {1, 1, 1, 1},
                {2, 16, 10, 6},
                {4, 16, 2, 2},
                {2, 16, 10, 6}
            }
        }
    };
    return inputShapes_4D;
}

const std::vector<InputShape>& inputShapes_5D() {
    static const std::vector<InputShape> inputShapes_5D = {
        { {}, {{1, 32, 8, 1, 6}}},
        { {}, {{1, 9, 1, 15, 9}}},
        { {}, {{6, 64, 6, 1, 18}}},
        { {}, {{2, 31, 2, 9, 1}}},
        { {}, {{10, 16, 5, 10, 6}}},
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {2, 16, 5, 10, 6},
                {4, 16, 7, 2, 2},
                {2, 16, 5, 10, 6},
                {4, 16, 7, 2, 2}
            }
        },
        {
            // dynamic
            {{1, 5}, {1, 20}, {1, 7}, {1, 10}, {1, 7}},
            // target
            {
                {1, 1, 1, 1, 1},
                {2, 16, 5, 10, 6},
                {4, 16, 7, 2, 2},
                {2, 16, 5, 10, 6}
            }
        }
    };
    return inputShapes_5D;
}

const std::vector<ov::Shape>& inputShapesStatic_2D() {
    static const std::vector<ov::Shape> inputShapesStatic_2D = {
        {1},
        {16},
        {4}
    };
    return inputShapesStatic_2D;
}

const std::vector<ov::Shape>& inputShapesStatic_3D() {
    static const std::vector<ov::Shape> inputShapesStatic_3D = {
        {2, 16, 6},
        {4, 16, 2},
        {1, 16, 4}
    };
    return inputShapesStatic_3D;
}

const std::vector<ov::Shape>& inputShapesStatic_4D() {
    static const std::vector<ov::Shape> inputShapesStatic_4D = {
        {1, 7, 3, 5},
        {1, 15, 9, 5},
        {4, 41, 6, 9},
        // cover channel case 4*16*2+16+3=147
        {1, 147, 2, 2}
    };
    return inputShapesStatic_4D;
}

const std::vector<ov::Shape>& inputShapesStatic_5D() {
    static const std::vector<ov::Shape> inputShapesStatic_5D = {
        {1, 32, 8, 1, 6},
        {1, 9, 1, 15, 9},
        {6, 64, 6, 1, 18},
        // cover channel case 4*16*2+16+9=153
        {6, 153, 2, 2, 2}
    };
    return inputShapesStatic_5D;
}

const std::vector<ov::AxisSet>& emptyReductionAxes() {
    static const std::vector<ov::AxisSet> emptyReductionAxes = {{}};
    return emptyReductionAxes;
}

const std::vector<bool>& acrossChannels() {
    static const std::vector<bool> acrossChannels = {
        true,
        false
    };
    return acrossChannels;
}

const std::vector<double>& epsilon() {
    static const std::vector<double> epsilon = {
        0.000000001
    };
    return epsilon;
}

const std::vector<CPUSpecificParams>& cpuParams_4D() {
    static const std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nchw}, {nchw}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
    };
    return cpuParams_4D;
}

const std::vector<CPUSpecificParams>& cpuParams_5D() {
    static const std::vector<CPUSpecificParams> cpuParams_5D = {
            CPUSpecificParams({ncdhw}, {ncdhw}, {}, {}),
            CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
    };
    return cpuParams_5D;
}

const std::vector<ElementType>& inpPrc() {
    static const std::vector<ElementType> inpPrc = {
            ElementType::i8,
            ElementType::f32,
    };
    return inpPrc;
}

const std::vector<ElementType>& outPrc() {
    static const std::vector<ElementType> outPrc = {
            ElementType::f32,
    };
    return outPrc;
}

const std::vector<fusingSpecificParams>& fusingParamsSet() {
    static const std::vector<fusingSpecificParams> fusingParamsSet = {
            emptyFusingSpec,
    };
    return fusingParamsSet;
}
}  // namespace MVN
}  // namespace test
}  // namespace ov

// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extremum.hpp"
#include "internal_properties.hpp"
#include "common_test_utils/node_builders/extremum.hpp"
#include "shared_test_classes/single_op/minimum_maximum.hpp"

namespace ov {
namespace test {

using namespace CPUTestUtils;
using namespace ov::test::utils;

std::string ExtremumLayerCPUTest::getTestCaseName(const testing::TestParamInfo<ExtremumLayerCPUTestParamSet> &obj) {
    std::vector<ov::test::InputShape> inputShapes;
    utils::MinMaxOpType extremumType;
    ov::element::Type netPrecision, inPrecision, outPrecision;
    CPUTestUtils::CPUSpecificParams cpuParams;
    bool enforceSnippets;
    std::tie(inputShapes, extremumType, netPrecision, inPrecision, outPrecision, cpuParams, enforceSnippets) = obj.param;
    std::ostringstream result;
    result << extremumNames[extremumType] << "_";
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
    result << "netPRC=" << netPrecision.to_string() << "_";
    result << "inPRC=" << inPrecision.to_string() << "_";
    result << "outPRC=" << outPrecision.to_string() << "_";
    result << CPUTestUtils::CPUTestsBase::getTestCaseName(cpuParams);
    result << "_enforceSnippets=" << enforceSnippets;

    return result.str();
}

void ExtremumLayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    std::vector<ov::test::InputShape> inputShapes;
    utils::MinMaxOpType extremumType;
    ov::element::Type netPrecision, inPrecision, outPrecision;
    CPUTestUtils::CPUSpecificParams cpuParams;
    bool enforceSnippets;
    std::tie(inputShapes, extremumType, netPrecision, inPrecision, outPrecision, cpuParams, enforceSnippets) = this->GetParam();
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

    inType  = inPrecision;
    outType = outPrecision;
    const auto primitiveType = getPrimitiveType();
    selectedType = primitiveType.empty() ? "" : primitiveType + "_" + netPrecision.to_string();

    init_input_shapes(inputShapes);

    if (enforceSnippets) {
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK));
    } else {
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
    }

    auto param1 = std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0]);
    auto param2 = std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[1]);
    auto extremum = utils::make_extremum(param1, param2, extremumType);
    extremum->get_rt_info() = getCPUInfo();
    function = std::make_shared<ov::Model>(ov::NodeVector{extremum}, ov::ParameterVector{param1, param2}, "Extremum");
}

std::string ExtremumLayerCPUTest::getPrimitiveType() {
#if defined(OV_CPU_WITH_ACL)
#if defined(OPENVINO_ARCH_ARM64)
    return "jit";
#endif
    return "acl";
#elif defined(OV_CPU_WITH_SHL)
    return "shl";
#else
    return CPUTestsBase::getPrimitiveType();
#endif
}

TEST_P(ExtremumLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Eltwise");
}

namespace extremum {

const std::vector<std::vector<ov::Shape>>& inputShape() {
    static const std::vector<std::vector<ov::Shape>> shape {
        {{2, 4, 4, 1}, {2, 4, 4, 1}},
        {{2, 17, 5, 4}, {2, 17, 5, 4}},
    };

    return shape;
}

const std::vector<utils::MinMaxOpType>& extremumTypes() {
    static std::vector<utils::MinMaxOpType> types {
        MINIMUM,
        MAXIMUM,
    };

    return types;
}

const std::vector<ov::element::Type>& netPrecisions() {
    static const std::vector<ov::element::Type> netPrecisions {
        ov::element::f32
    };

    return netPrecisions;
}

const std::vector<CPUSpecificParams>& cpuParams4D() {
    static const std::vector<CPUSpecificParams> cpuParams4D {
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
    };

    return cpuParams4D;
}

}  // namespace extremum
}  // namespace test
}  // namespace ov

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <debug.h>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include <ov_models/builders.hpp>
#include "common_test_utils/common_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "functional_test_utils/skip_tests_config.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/convolution_params.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using MatMulCompressConvertParams = std::tuple<
    std::vector<InputShape>,            // input shapes
    std::pair<bool, bool>,              // transposeA, transposeB
    element::Type,                      // inference precision
    CPUSpecificParams
>;

class MatMulCompressConvertCPUTest: public testing::WithParamInterface<MatMulCompressConvertParams>,
                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulCompressConvertParams> obj) {
        std::vector<InputShape> inputShapes;
        std::pair<bool, bool> transpose;
        element::Type inferPrecision;
        CPUSpecificParams cpuParams;

        std::tie(inputShapes, transpose, inferPrecision, cpuParams) = obj.param;

        std::ostringstream result;
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "transpose_a=" << transpose.first << "_";
        result << "transpose_b=" << transpose.second << "_";

        result << "infer_precision=" << inferPrecision << "_";

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    template<typename T>
    void transposeShape(T& shape) {
        IE_ASSERT(shape.size() > 1);
        std::swap(*(shape.end() - 1), *(shape.end() - 2));
    }

    void CheckFCWeightsPrecision(element::Type expectedWeiElemType) const {
        auto getExecValue = [](const ov::Node::RTMap& rtInfo, const std::string &paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        const auto execFunction = compiledModel.get_runtime_model();
        ASSERT_NE(nullptr, execFunction);
        for (const auto &fcNode : execFunction->get_ops()) {
            if (getExecValue(fcNode->get_rt_info(), ExecGraphInfoSerialization::LAYER_TYPE) == "FullyConnected") {
                const auto &constNode = fcNode->get_input_node_shared_ptr(1);
                element::Type expectedType(getExecValue(constNode->get_rt_info(), ExecGraphInfoSerialization::OUTPUT_PRECISIONS));
                ASSERT_EQ(expectedType, expectedWeiElemType);
            }
        }
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        std::pair<bool, bool> transpose;
        element::Type inferPrecision;
        CPUSpecificParams cpuParams;

        std::tie(inputShapes, transpose, inferPrecision, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        init_input_shapes(inputShapes);

        bool transpA = transpose.first;
        bool transpB = transpose.second;

        if (transpA) {
            transposeShape(inputDynamicShapes[0]);
            for (auto& shapes : targetStaticShapes) {
                transposeShape(shapes[0]);
            }
        }
        if (transpB) {
            transposeShape(inputDynamicShapes[1]);
            for (auto& shapes : targetStaticShapes) {
                transposeShape(shapes[1]);
            }
        }

        if (inferPrecision == element::f16) {
            convertCount = 2; // convert f32->f16 on the activation input and convert f16->f32 on the output
        }

        const auto& inShapeA = inputDynamicShapes[0];
        const auto& inShapeB = inputDynamicShapes[1];

        configuration.emplace(ov::hint::inference_precision(inferPrecision));

        element::Type netType = element::f32;
        inType = outType = netType;

        std::string cpuNodeType = "FullyConnected";
        selectedType = makeSelectedTypeStr(selectedType, outType);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inShapeA)};
        auto tensor = ov::test::utils::create_and_fill_tensor(element::f32, inShapeB.get_shape());
        std::shared_ptr<Node> inputB = std::make_shared<ov::op::v0::Constant>(tensor);

        auto matMul = std::make_shared<ov::op::v0::MatMul>(params[0], inputB, transpA, transpB);

        function = CPUTestsBase::makeNgraphFunction(netType, params, matMul, cpuNodeType);
    }

    void CheckExecutionGraph() {
        CheckNumberOfNodesWithType(compiledModel, "FullyConnected", 1);
        CheckNumberOfNodesWithType(compiledModel, "Convert", convertCount);
        CheckFCWeightsPrecision(element::f32);
    }

    size_t convertCount = 0;
};

TEST_P(MatMulCompressConvertCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    CheckExecutionGraph();
}

namespace {

const std::vector<std::pair<bool, bool>> transposeParams = {
    {false, true},
};

const std::vector<std::vector<InputShape>> inputShapes2D = {
    static_shapes_to_test_representation({{2, 3}, {3, 4}}),
    {
        {{-1, -1}, {{2, 3}, {5, 3}}},
        {{3, 4}, {{3, 4}, {3, 4}}}
    },
};

const std::vector<element::Type> inferPrecisions = {
    element::f32,
#if defined(OV_CPU_ARM_ENABLE_FP16)
    element::f16,
#endif
};

const auto testParams2D_ARM_smoke = ::testing::Combine(
    ::testing::ValuesIn(inputShapes2D),
    ::testing::ValuesIn(transposeParams),
    ::testing::ValuesIn(inferPrecisions),
    ::testing::Values(CPUSpecificParams{}));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_ARM, MatMulCompressConvertCPUTest, testParams2D_ARM_smoke,
                        MatMulCompressConvertCPUTest::getTestCaseName);

} // namespace

} // namespace test
} // namespace ov

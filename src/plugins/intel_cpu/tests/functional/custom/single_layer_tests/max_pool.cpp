// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/max_pool.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace {

const ov::Shape inputShape{1, 512, 19, 19};
const std::vector<ov::Shape> kernels{{13, 13}, {9, 9}, {5, 5}};

}  // namespace

using MaxPoolI8TestParams = std::tuple<InputShape,  // input shape
                                       ov::Shape,   // kernel
                                       ElementType,
                                       TargetDevice>;

using MaxPoolI8CPUTestParamsSet = std::tuple<MaxPoolI8TestParams, CPUSpecificParams>;

class MaxPoolI8CPUTest : public testing::WithParamInterface<MaxPoolI8CPUTestParamsSet>,
                         virtual public SubgraphBaseTest,
                         public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MaxPoolI8CPUTestParamsSet>& obj) {
        const auto& [basicParamsSet, cpuParams] = obj.param;
        const auto& [inShape, kernel, netPrecision, targetDevice] = basicParamsSet;
        std::ostringstream result;
        result << "MaxPool_";
        result << "IS=" << ov::test::utils::partialShape2str({inShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : inShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << "K=" << ov::test::utils::vec2str(kernel) << "_";
        result << "Prc=" << netPrecision << "_";
        result << targetDevice << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [basicParamsSet, cpuParams] = this->GetParam();
        const auto& [inShape, kernel, netPrecision, _targetDevice] = basicParamsSet;
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        targetDevice = _targetDevice;
        selectedType = "ref_any_i8";

        init_input_shapes({inShape});

        auto input = std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0]);
        auto maxPool = std::make_shared<ov::op::v14::MaxPool>(input,
                                                              ov::Strides{1, 1},
                                                              ov::Strides{1, 1},
                                                              ov::Shape{0, 0},
                                                              ov::Shape{0, 0},
                                                              kernel,
                                                              ov::op::RoundingType::FLOOR,
                                                              ov::op::PadType::SAME_UPPER,
                                                              ov::element::i64,
                                                              0);
        maxPool->get_rt_info() = getCPUInfo();

        // The value output remains i8. The i64 index output is intentionally not part of this test model.
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(maxPool->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "MaxPool");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& input = function->input();
        ov::test::utils::InputGenerateData inputData;
        inputData.start_from = 0;
        inputData.range = 100;
        inputData.resolution = 1;
        inputs.insert({input.get_node_shared_ptr(),
                       ov::test::utils::create_and_fill_tensor(input.get_element_type(),
                                                                targetInputStaticShapes[0],
                                                                inputData)});
    }
};

TEST_P(MaxPoolI8CPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "MaxPool");
}

namespace {

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPool_I8,
    MaxPoolI8CPUTest,
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation({inputShape})),
                                          ::testing::ValuesIn(kernels),
                                          ::testing::Values(ov::element::i8),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::Values(CPUSpecificParams{{}, {}, {}, {}})),
    MaxPoolI8CPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
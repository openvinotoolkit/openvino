// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using GatherElementsParams = std::tuple<std::vector<InputShape>,  // Dynamic shape + Target static shapes
                                        int,                      // Axis
                                        ElementType,              // Data precision
                                        ElementType,              // Indices precision
                                        TargetDevice              // Device name
                                        >;

using GatherElementsCPUTestParamSet = std::tuple<GatherElementsParams, CPUSpecificParams>;

class GatherElementsCPUTest : public testing::WithParamInterface<GatherElementsCPUTestParamSet>,
                              virtual public ov::test::SubgraphBaseTest,
                              public CPUTestsBase {
public:
    static std::string getTestCaseNameCommon(const testing::TestParamInfo<GatherElementsParams>& obj) {
        std::vector<InputShape> shapes;
        ElementType dPrecision, iPrecision;
        int axis;
        std::string device;
        std::tie(shapes, axis, dPrecision, iPrecision, device) = obj.param;

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
        result << "Ax=" << axis << "_";
        result << "DP=" << dPrecision << "_";
        result << "IP=" << iPrecision << "_";
        result << "device=" << device;

        return result.str();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GatherElementsCPUTestParamSet>& obj) {
        GatherElementsParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << getTestCaseNameCommon(testing::TestParamInfo<GatherElementsParams>(basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 15;
            in_data.resolution = 32768;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        std::vector<InputShape> shapes;
        ElementType dPrecision, iPrecision;
        int axis;
        GatherElementsParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::tie(shapes, axis, dPrecision, iPrecision, targetDevice) = basicParamsSet;
        selectedType = std::string("ref_any_") + ov::element::Type(dPrecision).get_type_name();
        init_input_shapes(shapes);

        ov::ParameterVector params = {
            std::make_shared<ov::op::v0::Parameter>(dPrecision, inputDynamicShapes[0]),
            std::make_shared<ov::op::v0::Parameter>(iPrecision, inputDynamicShapes[1]),
        };

        auto gather = std::make_shared<ov::op::v6::GatherElements>(params[0], params[1], axis);
        function = makeNgraphFunction(dPrecision, params, gather, "GatherElements");
    }
};

TEST_P(GatherElementsCPUTest, CompareWithRefs) {
    run();
}

namespace {
std::vector<CPUSpecificParams> cpuParams_4D = {CPUSpecificParams({nchw}, {nchw}, {}, {})};

const std::vector<std::vector<InputShape>> inDynamicShapeParams = {
    {{{-1, -1, -1, -1}, {{2, 3, 5, 7}, {3, 4, 6, 8}}}, {{-1, -1, -1, -1}, {{2, 3, 9, 7}, {3, 4, 4, 8}}}},
    {{{{1, 10}, {1, 10}, {1, 10}, {1, 10}}, {{3, 4, 6, 8}, {2, 3, 5, 7}}},
     {{{1, 10}, {1, 10}, {1, 10}, {1, 10}}, {{3, 4, 4, 8}, {2, 3, 9, 7}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_set1,
                         GatherElementsCPUTest,
                         ::testing::Combine(::testing::Combine(::testing::ValuesIn(inDynamicShapeParams),       // shape
                                                               ::testing::ValuesIn(std::vector<int>({2, -2})),  // Axis
                                                               ::testing::ValuesIn(std::vector<ElementType>(
                                                                   {ElementType::bf16, ElementType::f32})),
                                                               ::testing::Values(ElementType::i32),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D))),
                         GatherElementsCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<
        InputShape,     // Input shape definition
        ElementType     // Net precision
> NonZeroLayerTestParams;

typedef std::tuple<
        NonZeroLayerTestParams,
        std::pair<size_t, size_t>, // start from, range
        CPUSpecificParams> NonZeroLayerCPUTestParamsSet;

class NonZeroLayerCPUTest : public testing::WithParamInterface<NonZeroLayerCPUTestParamsSet>,
                          virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NonZeroLayerCPUTestParamsSet> obj) {
        NonZeroLayerTestParams basicParamsSet;
        std::pair<size_t, size_t> genData;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, genData, cpuParams) = obj.param;
        std::string td;
        ElementType netType = ElementType::dynamic;
        InputShape inputShape;

        std::tie(inputShape, netType) = basicParamsSet;

        std::ostringstream result;
        result << "IS=";
        result  << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << ")_";
        result << "StartFrom=" << genData.first << "_";
        result << "Range=" << genData.second << "_";
        result << "netPRC=" << netType;
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = startFrom;
            in_data.range = range;
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    size_t startFrom = 0, range = 10;
    size_t inferNum = 0;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        NonZeroLayerTestParams basicParamsSet;
        std::pair<size_t, size_t> genData;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, genData, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        ElementType netType = ElementType::dynamic;
        InputShape inputShape;
        std::tie(inputShape, netType) = basicParamsSet;

        std::tie(startFrom, range) = genData;

        init_input_shapes({inputShape});
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(netType, shape));
        }

        auto nonZero = std::make_shared<ov::op::v3::NonZero>(inputParams[0]);
        // I8 was used as a special placeholder during calculating of primitive type if input was U8,
        // real runtime precision is still U8
        selectedType = makeSelectedTypeStr("ref", netType == ElementType::u8 ? ElementType::i8 : netType);
        inputParams[0]->set_friendly_name("input");
        function = makeNgraphFunction(netType, inputParams, nonZero, "NonZero");
    }
};

TEST_P(NonZeroLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "NonZero");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    return std::vector<CPUSpecificParams> {CPUSpecificParams{{}, {nc}, {}, {}}};;
}

const std::vector<ElementType> netPrecisions = {
        ElementType::f32,
        ElementType::bf16,
        ElementType::i32,
        ElementType::i8,
        ElementType::u8
};

const std::vector<std::pair<size_t, size_t>> genData = {
    {0, 10},
    {0, 1}
};

std::vector<InputShape> inShapesDynamic = {
        {
            //dynamic shape
            {-1},
            { //target static shapes
                {100},
                {200},
                {300}
            }
        },
        {
            //dynamic shape
            {-1, -1},
            { //target static shapes
                {4, 100},
                {4, 200},
                {4, 300}
            }
        },
        {
            //dynamic shape
            {-1, -1, -1},
            { //target static shapes
                {4, 4, 100},
                {5, 0, 2},
                {4, 4, 200},
                {4, 4, 300}
            }
        },
        {
            //dynamic shape
            {-1, -1, -1, -1},
            { //target static shapes
                {4, 4, 4, 100},
                {4, 4, 4, 200},
                {5, 0, 0, 2},
                {4, 4, 4, 300}
            }
        },
        {
            //dynamic shape
            {-1, {1, 10}, -1, {1, 500}},
            { //target static shapes
                {4, 4, 4, 100},
                {4, 4, 4, 200},
                {4, 4, 4, 300}
            }
        },
        {
            //dynamic shape
            {{1, 10}, {1, 10}, {1, 10}, {1, 500}},
            { //target static shapes
                {4, 4, 4, 100},
                {4, 4, 4, 200},
                {4, 4, 4, 300}
            }
        },
        {
            // dynamic shape
            {-1, -1, -1, -1, -1},
            { // target static shapes
                {4, 24, 5, 6, 4},
                {8, 32, 9, 10, 2},
                {4, 24, 5, 6, 4},
                {16, 48, 9, 12, 2}
            }
        },
        {
            // dynamic shape
            {-1, {1, 50}, -1, {1, 30}, -1},
            { // target static shapes
                {1, 16, 1, 8, 8},
                {8, 32, 5, 14, 6},
                {4, 16, 9, 10, 8},
                {8, 32, 5, 14, 6}
            }
        }
};
std::vector<ov::Shape> inShapesStatic = {
        { 100 },
        { 4, 100 },
        { 4, 2, 100 },
        { 4, 4, 2, 100 },
        { 4, 4, 4, 2, 100 }
};

const auto paramsStatic = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapesStatic)),
                ::testing::ValuesIn(netPrecisions)),
        ::testing::ValuesIn(genData),
        ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto paramsDynamic = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapesDynamic),
                ::testing::ValuesIn(netPrecisions)),
        ::testing::ValuesIn(genData),
        ::testing::ValuesIn(filterCPUInfoForDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_NonZeroStaticCPUTest, NonZeroLayerCPUTest,
                         paramsStatic, NonZeroLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_NonZeroDynamicCPUTest, NonZeroLayerCPUTest,
                         paramsDynamic, NonZeroLayerCPUTest::getTestCaseName);

} // namespace

}  // namespace test
}  // namespace ov

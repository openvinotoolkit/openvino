// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace CPUTestUtils;

using TileLayerTestParamsSet = typename std::tuple<std::vector<ov::test::InputShape>,  // Input shapes
                                                   std::vector<int64_t>,               // Repeats
                                                   ov::element::Type_t,                // Network precision
                                                   bool,                               // Is Repeats input constant
                                                   std::string>;                       // Device name

typedef std::tuple<TileLayerTestParamsSet, CPUSpecificParams> TileLayerCPUTestParamsSet;

class TileLayerCPUTest : public testing::WithParamInterface<TileLayerCPUTestParamsSet>,
                         virtual public ov::test::SubgraphBaseTest,
                         public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TileLayerCPUTestParamsSet> obj) {
        TileLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::vector<ov::test::InputShape> inputShapes;
        std::vector<int64_t> repeats;
        ov::element::Type_t netPrecision;
        bool isRepeatsConst;
        std::string deviceName;
        std::tie(inputShapes, repeats, netPrecision, isRepeatsConst, deviceName) = basicParamsSet;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "Repeats=" << ov::test::utils::vec2str(repeats) << "_";
        result << "netPrec=" << netPrecision << "_";
        result << "constRepeats=" << (isRepeatsConst ? "True" : "False") << "_";
        result << "trgDev=" << deviceName;

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        TileLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::vector<ov::test::InputShape> inputShapes;
        ov::element::Type_t netPrecision;
        bool isRepeatsConst;
        std::tie(inputShapes, repeatsData, netPrecision, isRepeatsConst, targetDevice) = basicParamsSet;

        selectedType += std::string("_") + ov::element::Type(netPrecision).get_type_name();

        if (inputShapes.front().first.rank() != 0) {
            inputDynamicShapes.push_back(inputShapes.front().first);
            if (!isRepeatsConst) {
                inputDynamicShapes.push_back({static_cast<int64_t>(repeatsData.size())});
            }
        }
        const size_t targetStaticShapeSize = inputShapes.front().second.size();
        targetStaticShapes.resize(targetStaticShapeSize);
        for (size_t i = 0lu; i < targetStaticShapeSize; ++i) {
            targetStaticShapes[i].push_back(inputShapes.front().second[i]);
            if (!isRepeatsConst)
                targetStaticShapes[i].push_back({repeatsData.size()});
        }

        ov::ParameterVector functionParams;
        if (inputDynamicShapes.empty()) {
            functionParams.push_back(
                std::make_shared<ov::op::v0::Parameter>(netPrecision, targetStaticShapes.front().front()));
        } else {
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes.front()));
            if (!isRepeatsConst) {
                functionParams.push_back(
                    std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[1]));
                functionParams.back()->set_friendly_name("repeats");
            }
        }
        functionParams.front()->set_friendly_name("data");

        std::shared_ptr<ov::Node> tileNode;
        if (isRepeatsConst) {
            tileNode = std::make_shared<ov::op::v0::Tile>(
                functionParams[0],
                ov::op::v0::Constant::create(ov::element::i64, {repeatsData.size()}, repeatsData));
        } else {
            tileNode = std::make_shared<ov::op::v0::Tile>(functionParams[0], functionParams[1]);
        }
        function = makeNgraphFunction(netPrecision, functionParams, tileNode, "CPUTile");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0lu; i < funcInputs.size(); i++) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (funcInput.get_node()->get_friendly_name() == "repeats") {
                tensor = ov::Tensor{ov::element::i64, targetInputStaticShapes[i]};
                auto data = tensor.data<ov::element_type_traits<ov::element::i64>::value_type>();
                for (size_t i = 0lu; i < repeatsData.size(); i++) {
                    data[i] = repeatsData[i];
                }
            } else {
                if (funcInput.get_element_type().is_real()) {
                    ov::test::utils::InputGenerateData in_data;
                    in_data.start_from = 0;
                    in_data.range = 10;
                    in_data.resolution = 1000;
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
                } else {
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                     targetInputStaticShapes[i]);
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    std::vector<int64_t> repeatsData;
};

TEST_P(TileLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Tile");
}

namespace {

/* CPU PARAMS */
const auto cpuParams_nchw = CPUSpecificParams{{nchw}, {nchw}, {}, "ref"};
const auto cpuParams_ncdhw = CPUSpecificParams{{ncdhw}, {ncdhw}, {}, "ref"};

const auto cpuParams_nChw16c = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, "ref"};
const auto cpuParams_nCdhw16c = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, "ref"};

const auto cpuParams_nChw8c = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, "ref"};
const auto cpuParams_nCdhw8c = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, "ref"};

const auto cpuParams_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {}, "ref"};
const auto cpuParams_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, "ref"};
/* ========== */

/* PARAMS */
const std::vector<ov::element::Type_t> netPrecisions = {ov::element::f32,
                                                        ov::element::bf16,
                                                        ov::element::i32,
                                                        ov::element::i8};

const std::vector<std::vector<ov::test::InputShape>> staticInputShapes4D = {{{{},
                                                                              {// Static shapes
                                                                               {2, 16, 3, 4}}}},
                                                                            {{{},
                                                                              {// Static shapes
                                                                               {1, 16, 1, 1}}}}};
const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes4D = {
    {{// Origin dynamic shapes
      {ov::Dimension(1, 20), ov::Dimension(10, 20), ov::Dimension(1, 20), ov::Dimension(1, 20)},
      {// Dynamic shapes instances
       {2, 16, 3, 4},
       {1, 16, 1, 1},
       {1, 16, 2, 3}}}},
    {{// Origin dynamic shapes
      {-1, -1, -1, -1},
      {// Dynamic shapes instances
       {3, 15, 5, 7},
       {4, 55, 8, 24}}}}};

const std::vector<std::vector<ov::test::InputShape>> staticInputShapes5D = {{{{},
                                                                              {// Static shapes
                                                                               {2, 16, 2, 3, 4}}}}};
const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes5D = {
    {{// Origin dynamic shapes
      {ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 70)},
      {// Dynamic shapes instances
       {2, 16, 2, 3, 4},
       {1, 16, 8, 5, 4},
       {8, 1, 2, 3, 64}}}},
    {{// Origin dynamic shapes
      {-1, -1, -1, -1, -1},
      {// Dynamic shapes instances
       {2, 16, 2, 3, 4},
       {1, 16, 8, 5, 4},
       {8, 1, 2, 3, 64}}}}};

const std::vector<std::vector<int64_t>> repeats4D =
    {{2, 3}, {1, 2, 3}, {1, 1, 1, 1}, {1, 1, 2, 3}, {1, 2, 1, 3}, {2, 1, 1, 1}, {2, 3, 1, 1}};
const std::vector<std::vector<int64_t>> repeats5D =
    {{1, 2, 3}, {1, 1, 2, 3}, {1, 1, 1, 2, 3}, {1, 2, 1, 1, 3}, {2, 1, 1, 1, 1}, {2, 3, 1, 1, 1}};

const std::vector<CPUSpecificParams> CPUParams4D = {
    cpuParams_nchw,
    cpuParams_nChw16c,
    cpuParams_nChw8c,
    cpuParams_nhwc,
};

const std::vector<CPUSpecificParams> CPUParams5D = {
    cpuParams_ncdhw,
    cpuParams_nCdhw16c,
    cpuParams_nCdhw8c,
    cpuParams_ndhwc,
};
/* ============= */

/* INSTANCES */
INSTANTIATE_TEST_SUITE_P(smoke_StaticShape4D,
                        TileLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(staticInputShapes4D),
                                                              ::testing::ValuesIn(repeats4D),
                                                              ::testing::ValuesIn(netPrecisions),
                                                              ::testing::Values(true),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::ValuesIn(CPUParams4D)),
                        TileLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape4D,
                        TileLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(dynamicInputShapes4D),
                                                              ::testing::ValuesIn(repeats4D),
                                                              ::testing::ValuesIn(netPrecisions),
                                                              ::testing::Values(true, false),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        TileLayerCPUTest::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynBatchInputShapes4D = {{// Origin dynamic shapes
                                                                               {{{1, 20}, 16, 3, 4},
                                                                                {// Dynamic shapes instances
                                                                                 {2, 16, 3, 4},
                                                                                 {1, 16, 3, 4},
                                                                                 {3, 16, 3, 4}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_DynBatch4D,
                        TileLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(dynBatchInputShapes4D),
                                                              ::testing::Values(std::vector<int64_t>{1, 2, 1, 3}),
                                                              ::testing::ValuesIn(netPrecisions),
                                                              ::testing::Values(true),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        TileLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticShape5D,
                        TileLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(staticInputShapes5D),
                                                              ::testing::ValuesIn(repeats5D),
                                                              ::testing::ValuesIn(netPrecisions),
                                                              ::testing::Values(true),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::ValuesIn(CPUParams5D)),
                        TileLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape5D,
                        TileLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(dynamicInputShapes5D),
                                                              ::testing::ValuesIn(repeats5D),
                                                              ::testing::ValuesIn(netPrecisions),
                                                              ::testing::Values(true, false),
                                                              ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                           ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        TileLayerCPUTest::getTestCaseName);
/* ========= */

}  // namespace
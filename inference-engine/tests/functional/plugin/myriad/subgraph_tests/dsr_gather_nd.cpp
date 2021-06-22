// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

const std::vector<ngraph::element::Type> dataTypeVector = {
    ngraph::element::f16,
    ngraph::element::f32,
    ngraph::element::i32,
};

const std::vector<ngraph::element::Type> idxTypeVector = {
    ngraph::element::i32,
};

struct GatherNDTestCase {
    DataShapeWithUpperBound dataShape;
    DataShapeWithUpperBound indicesShape;
    int64_t batchDims;
};

using GatherNDParameters = std::tuple<
    DataType,                     // data type
    DataType,                     // indices type
    GatherNDTestCase,             // GatherND parameters
    LayerTestsUtils::TargetDevice // device name
>;

class DSR_GatherNDBase : public testing::WithParamInterface<GatherNDParameters>,
                         public DSR_TestsCommon {
protected:
    std::set<std::string> m_indicesInputNames;

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        const auto& name = info.name();
        if (m_indicesInputNames.count(name)) {
            const auto& parameters = GetParam();
            const auto& gatherSetup = std::get<2>(parameters);
            const auto& lastIndicesDim = gatherSetup.indicesShape.shape.back();

            const auto endValue = std::min_element(gatherSetup.dataShape.shape.begin() + gatherSetup.batchDims,
                 gatherSetup.dataShape.shape.begin() + gatherSetup.batchDims + lastIndicesDim);

            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), *endValue, 0);
        }
        return DSR_TestsCommon::GenerateInput(info);
    }

    void SetUp() override {
        DSR_TestsCommon::SetUp();
        SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);
    }
};

class DSR_GatherNDDynamicDataStaticIdx : public DSR_GatherNDBase {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& gatherSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputDataSubgraph = createInputSubgraphWithDSR(inDataType, gatherSetup.dataShape);

        const auto indicesParam = createParameter(idxType, gatherSetup.indicesShape.shape);
        m_indicesInputNames.insert(indicesParam->get_friendly_name());

        return std::make_shared<ngraph::opset5::GatherND>(inputDataSubgraph, indicesParam, gatherSetup.batchDims);
    }
};

TEST_P(DSR_GatherNDDynamicDataStaticIdx, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicGatherData, DSR_GatherNDDynamicDataStaticIdx, testing::Combine(
    testing::ValuesIn(dataTypeVector),
    testing::ValuesIn(idxTypeVector),
    testing::Values(
          GatherNDTestCase{DataShapeWithUpperBound{{1, 1000, 4}, {1, 22734, 4}}, DataShapeWithUpperBound{{300, 2}, {}}, 0},
          GatherNDTestCase{DataShapeWithUpperBound{{1, 500, 4}, {1, 22734, 4}}, DataShapeWithUpperBound{{300, 2}, {}}, 0}),
    testing::Values(CommonTestUtils::DEVICE_MYRIAD)));


class DSR_GatherNDStaticDataDynamicIdx : public DSR_GatherNDBase {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& gatherSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto dataParam = createParameter(inDataType, gatherSetup.dataShape.shape);
        const auto inputIdxSubgraph = createInputSubgraphWithDSR(idxType, gatherSetup.indicesShape);
        m_indicesInputNames.insert(inputIdxSubgraph->get_input_node_shared_ptr(0)->get_friendly_name());

        return std::make_shared<ngraph::opset5::GatherND>(dataParam, inputIdxSubgraph, gatherSetup.batchDims);
    }
};

TEST_P(DSR_GatherNDStaticDataDynamicIdx, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicGatherIdx, DSR_GatherNDStaticDataDynamicIdx, testing::Combine(
    testing::ValuesIn(dataTypeVector),
    testing::ValuesIn(idxTypeVector),
    testing::Values(
        GatherNDTestCase{DataShapeWithUpperBound{{1, 22734, 4}, {}}, DataShapeWithUpperBound{{100, 2}, {300, 2}}, 0},
        GatherNDTestCase{DataShapeWithUpperBound{{1, 22734, 4}, {}}, DataShapeWithUpperBound{{1, 2}, {300, 2}}, 0}),
    testing::Values(CommonTestUtils::DEVICE_MYRIAD)));


class DSR_GatherNDDynamicDataDynamicIdx : public DSR_GatherNDBase {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& gatherSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputDataSubgraph = createInputSubgraphWithDSR(inDataType, gatherSetup.dataShape);
        const auto inputIdxSubgraph = createInputSubgraphWithDSR(idxType, gatherSetup.indicesShape);
        m_indicesInputNames.insert(inputIdxSubgraph->get_input_node_shared_ptr(0)->get_friendly_name());

        return std::make_shared<ngraph::opset5::GatherND>(inputDataSubgraph, inputIdxSubgraph, gatherSetup.batchDims);
    }
};

TEST_P(DSR_GatherNDDynamicDataDynamicIdx, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicGather, DSR_GatherNDDynamicDataDynamicIdx, testing::Combine(
    testing::ValuesIn(dataTypeVector),
    testing::ValuesIn(idxTypeVector),
    testing::Values(
            GatherNDTestCase{DataShapeWithUpperBound{{1, 1000, 4}, {1, 22734, 4}}, DataShapeWithUpperBound{{100, 2}, {300, 2}}, 0},
            GatherNDTestCase{DataShapeWithUpperBound{{1, 500, 4}, {1, 22734, 4}}, DataShapeWithUpperBound{{1, 2}, {300, 2}}, 0}),
    testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace

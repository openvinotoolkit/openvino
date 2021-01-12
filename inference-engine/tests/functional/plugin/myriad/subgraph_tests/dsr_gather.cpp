// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <ie_ngraph_utils.hpp>

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

struct GatherTestCase {
    DataShapeWithUpperBound inputShapes;
    DataShapeWithUpperBound indexShape;
    int64_t axis;
};

using GatherParameters = std::tuple<
    DataType,
    DataType,
    GatherTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_GatherBase : public testing::WithParamInterface<GatherParameters>,
                       public DSR_TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherParameters> obj) {
        DataType dataType, idxType;
        GatherTestCase gatherTestCase;
        LayerTestsUtils::TargetDevice targetDevice;
        std::tie(dataType, idxType, gatherTestCase, targetDevice) = obj.param;

        std::ostringstream result;
        result << "DT=" << dataType << "_";
        result << "IT=" << idxType << "_";
        result << "DataRealShape=" << CommonTestUtils::vec2str(gatherTestCase.inputShapes.shape) << "_";
        result << "DataUBShape=" << CommonTestUtils::vec2str(gatherTestCase.inputShapes.upperBoundShape) << "_";
        result << "IdxRealShape=" << CommonTestUtils::vec2str(gatherTestCase.inputShapes.shape) << "_";
        result << "IdxUBShape=" << CommonTestUtils::vec2str(gatherTestCase.inputShapes.upperBoundShape) << "_";
        result << "Axis=" << gatherTestCase.axis << "_";
        result << "trgDev=" << targetDevice;
        return result.str();
    }

protected:
    std::set<std::string> m_indicesInputNames;

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        const auto& name = info.name();
        if (m_indicesInputNames.count(name)) {
            const auto& parameters = GetParam();
            const auto& gatherSetup = std::get<2>(parameters);
            const auto& inputRank = gatherSetup.inputShapes.shape.size();
            const auto axis = gatherSetup.axis < 0 ? gatherSetup.axis + inputRank : gatherSetup.axis;

            const auto endValue = gatherSetup.inputShapes.shape[axis] - 1;

            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), endValue, 0);
        }
        return DSR_TestsCommon::GenerateInput(info);
    }
};

class DSR_GatherDynamicDataStaticIdx : public DSR_GatherBase {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& gatherSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputDataSubgraph = createInputSubgraphWithDSR(inDataType, gatherSetup.inputShapes);

        const auto indicesParam = createParameter(idxType, gatherSetup.indexShape.shape);
        m_indicesInputNames.insert(indicesParam->get_friendly_name());

        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gatherSetup.axis});

        const auto gather = std::make_shared<ngraph::opset3::Gather>(inputDataSubgraph, indicesParam, axis);

        return gather;
    }
};

TEST_P(DSR_GatherDynamicDataStaticIdx, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke_DynamicGatherData, DSR_GatherDynamicDataStaticIdx, testing::Combine(
        testing::ValuesIn(dataTypeVector),
        testing::ValuesIn(idxTypeVector),
        testing::Values(
                GatherTestCase{DataShapeWithUpperBound{{800}, {1000}}, DataShapeWithUpperBound{{700}, {}}, 0},
                GatherTestCase{DataShapeWithUpperBound{{800, 4}, {1000, 4}}, DataShapeWithUpperBound{{700}, {}}, 0},
                GatherTestCase{DataShapeWithUpperBound{{800, 4}, {1000, 4}}, DataShapeWithUpperBound{{700}, {}}, -2}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
        DSR_GatherBase::getTestCaseName);


class DSR_GatherStaticDataDynamicIdx : public DSR_GatherBase {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& gatherSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);
        outPrc = InferenceEngine::details::convertPrecision(inDataType);;

        const auto dataParam = std::make_shared<ngraph::opset3::Parameter>(inDataType, gatherSetup.inputShapes.shape);
        m_parameterVector.push_back(dataParam);
        const auto inputIdxSubgraph = createInputSubgraphWithDSR(idxType, gatherSetup.indexShape);
        m_indicesInputNames.insert(inputIdxSubgraph->get_input_node_shared_ptr(0)->get_friendly_name());

        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gatherSetup.axis});

        const auto gather = std::make_shared<ngraph::opset3::Gather>(dataParam, inputIdxSubgraph, axis);

        return gather;
    }
};

TEST_P(DSR_GatherStaticDataDynamicIdx, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke_DynamicGatherIdx, DSR_GatherStaticDataDynamicIdx, testing::Combine(
        testing::ValuesIn(dataTypeVector),
        testing::ValuesIn(idxTypeVector),
        testing::Values(
                GatherTestCase{DataShapeWithUpperBound{{1000}, {}}, DataShapeWithUpperBound{{800}, {1000}}, 0},
                GatherTestCase{DataShapeWithUpperBound{{1000, 4}, {}}, DataShapeWithUpperBound{{800}, {1000}}, 0},
                GatherTestCase{DataShapeWithUpperBound{{1000, 4}, {}}, DataShapeWithUpperBound{{800}, {1000}}, -2},
                GatherTestCase{DataShapeWithUpperBound{{1, 3, 200, 304}, {}}, DataShapeWithUpperBound{{142, 64}, {300, 64}}, 2}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
        DSR_GatherBase::getTestCaseName);


class DSR_GatherDynamicDataDynamicIdx : public DSR_GatherBase {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& gatherSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputDataSubgraph = createInputSubgraphWithDSR(inDataType, gatherSetup.inputShapes);
        const auto inputIdxSubgraph = createInputSubgraphWithDSR(idxType, gatherSetup.indexShape);
        m_indicesInputNames.insert(inputIdxSubgraph->get_input_node_shared_ptr(0)->get_friendly_name());

        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gatherSetup.axis});

        const auto gather = std::make_shared<ngraph::opset3::Gather>(inputDataSubgraph, inputIdxSubgraph, axis);

        return gather;
    }
};

TEST_P(DSR_GatherDynamicDataDynamicIdx, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke_DynamicGather, DSR_GatherDynamicDataDynamicIdx, testing::Combine(
        testing::ValuesIn(dataTypeVector),
        testing::ValuesIn(idxTypeVector),
        testing::Values(
                GatherTestCase{DataShapeWithUpperBound{{800}, {1000}}, DataShapeWithUpperBound{{700}, {1000}}, 0},
                GatherTestCase{DataShapeWithUpperBound{{800, 4}, {1000, 4}}, DataShapeWithUpperBound{{700}, {1000}}, 0},
                GatherTestCase{DataShapeWithUpperBound{{800, 4}, {1000, 4}}, DataShapeWithUpperBound{{700}, {1000}}, -2}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
        DSR_GatherBase::getTestCaseName);

}  // namespace

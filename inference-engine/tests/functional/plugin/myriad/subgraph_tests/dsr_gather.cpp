// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
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

struct GatherTestCase {
    DataShapeWithUpperBound inputShapes;
    DataShapeWithUpperBound indexShape;
    int64_t axis, firstSplitPoint, secondSplitPoint;
};

using GatherParameters = std::tuple<
    DataType,
    DataType,
    GatherTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_GatherBase : public testing::WithParamInterface<GatherParameters>,
                       public DSR_TestsCommon {
protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        const auto& name = info.name();
        const auto suffix = std::string("/indices");
        if (name.compare(name.length() - suffix.length(), suffix.length(), suffix) == 0) {
            const auto& parameters = GetParam();
            const auto& gatherSetup = std::get<2>(parameters);
            const auto& inputRank = gatherSetup.inputShapes.shape.size();
            const auto axis = gatherSetup.axis < 0 ? gatherSetup.axis + inputRank : gatherSetup.axis;

            const auto startValue = 0;
            const auto endValue = gatherSetup.inputShapes.shape[axis] - 1;

            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), endValue - startValue, startValue);
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

        const auto indicesParam = std::make_shared<ngraph::opset3::Parameter>(idxType, gatherSetup.indexShape.shape);
        indicesParam->set_friendly_name(indicesParam->get_friendly_name() + "/indices");
        m_parameterVector.push_back(indicesParam);
        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gatherSetup.axis});

        const auto gather = std::make_shared<ngraph::opset3::Gather>(inputDataSubgraph, indicesParam, axis);

        return gather;
    }
};

TEST_P(DSR_GatherDynamicDataStaticIdx, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DynamicGatherData, DSR_GatherDynamicDataStaticIdx, testing::Combine(
        testing::ValuesIn(dataTypeVector),
        testing::ValuesIn(idxTypeVector),
        testing::Values(
                GatherTestCase{DataShapeWithUpperBound{{800}, {1000}}, DataShapeWithUpperBound{{700}, {}}, 0, 0, 0},
                GatherTestCase{DataShapeWithUpperBound{{800, 4}, {1000, 4}}, DataShapeWithUpperBound{{700}, {}}, 0, 0, 0}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD)));


class DSR_GatherStaticDataDynamicIdx : public DSR_GatherBase {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& gatherSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto dataParam = std::make_shared<ngraph::opset3::Parameter>(inDataType, gatherSetup.inputShapes.shape);
        m_parameterVector.push_back(dataParam);
        const auto inputIdxSubgraph = createInputSubgraphWithDSR(idxType, gatherSetup.indexShape, "/indices");

        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gatherSetup.axis});

        const auto gather = std::make_shared<ngraph::opset3::Gather>(dataParam, inputIdxSubgraph, axis);

        return gather;
    }
};

TEST_P(DSR_GatherStaticDataDynamicIdx, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DynamicGatherIdx, DSR_GatherStaticDataDynamicIdx, testing::Combine(
        testing::ValuesIn(dataTypeVector),
        testing::ValuesIn(idxTypeVector),
        testing::Values(
                GatherTestCase{DataShapeWithUpperBound{{1000}, {}}, DataShapeWithUpperBound{{800}, {1000}}, 0, 0, 0},
                GatherTestCase{DataShapeWithUpperBound{{1000, 4}, {}}, DataShapeWithUpperBound{{800}, {1000}}, 0, 0, 1}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD)));


class DSR_GatherDynamicDataDynamicIdx : public DSR_GatherBase {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& gatherSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputDataSubgraph = createInputSubgraphWithDSR(inDataType, gatherSetup.inputShapes);
        const auto inputIdxSubgraph = createInputSubgraphWithDSR(idxType, gatherSetup.indexShape, "/indices");

        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gatherSetup.axis});

        const auto gather = std::make_shared<ngraph::opset3::Gather>(inputDataSubgraph, inputIdxSubgraph, axis);

        return gather;
    }
};

TEST_P(DSR_GatherDynamicDataDynamicIdx, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DynamicGather, DSR_GatherDynamicDataDynamicIdx, testing::Combine(
        testing::ValuesIn(dataTypeVector),
        testing::ValuesIn(idxTypeVector),
        testing::Values(
                GatherTestCase{DataShapeWithUpperBound{{800}, {1000}}, DataShapeWithUpperBound{{700}, {1000}}, 0, 0, 0},
                GatherTestCase{DataShapeWithUpperBound{{800, 4}, {1000, 4}}, DataShapeWithUpperBound{{700}, {1000}}, 0, 0, 1}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace

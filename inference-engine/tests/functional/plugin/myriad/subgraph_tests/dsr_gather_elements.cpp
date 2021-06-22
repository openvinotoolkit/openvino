// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"
#include "dsr_gather_base.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <ngraph/opsets/opset6.hpp>

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

class DSR_GatherElements : public DSR_GatherBase {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);

        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& gatherSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto dataParam = std::make_shared<ngraph::opset6::Parameter>(inDataType, gatherSetup.inputShapes.shape);
        m_parameterVector.push_back(dataParam);
        const auto inputIdxSubgraph = createInputSubgraphWithDSR(idxType, gatherSetup.indexShape);
        m_indicesInputNames.insert(inputIdxSubgraph->get_input_node_shared_ptr(0)->get_friendly_name());

        const auto gather = std::make_shared<ngraph::opset6::GatherElements>(dataParam, inputIdxSubgraph, gatherSetup.axis);

        return gather;
    }
};

TEST_P(DSR_GatherElements, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicGatherElements, DSR_GatherElements,
    testing::Combine(
        testing::ValuesIn(dataTypeVector),
        testing::ValuesIn(idxTypeVector),
        testing::Values(
            GatherTestCase{DataShapeWithUpperBound{{1000}, {}}, DataShapeWithUpperBound{{800}, {1000}}, 0},
            GatherTestCase{DataShapeWithUpperBound{{1000, 4}, {}}, DataShapeWithUpperBound{{100, 4}, {800, 4}}, 0},
            GatherTestCase{DataShapeWithUpperBound{{4, 1000}, {}}, DataShapeWithUpperBound{{4, 100}, {4, 800}}, 1},
            GatherTestCase{DataShapeWithUpperBound{{30, 3, 64, 608}, {}}, DataShapeWithUpperBound{{30, 3, 64, 60}, {30, 3, 64, 64}}, 3},
            GatherTestCase{DataShapeWithUpperBound{{800}, {1000}}, DataShapeWithUpperBound{{200}, {800}}, 0},
            GatherTestCase{DataShapeWithUpperBound{{800, 4}, {1000, 4}}, DataShapeWithUpperBound{{300, 4}, {800, 4}}, 0},
            GatherTestCase{DataShapeWithUpperBound{{4, 800}, {4, 1000}}, DataShapeWithUpperBound{{4, 700}, {4, 750}}, 1}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

struct DataTypeConversionPair {
    DataType srcType;
    DataType dstType;
};

using Parameters = std::tuple<
    DataTypeConversionPair,
    DataShapeWithUpperBound,
    LayerTestsUtils::TargetDevice
>;

class DSR_Convert : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataTypes = std::get<0>(parameters);
        const auto& inDataShapes = std::get<1>(parameters);
        targetDevice = std::get<2>(parameters);

        const auto inputSubgraph = createInputSubgraphWithDSR(
                inDataTypes.srcType, inDataShapes);

        const auto convert = std::make_shared<ngraph::opset3::Convert>(
                inputSubgraph, inDataTypes.dstType);

        return convert;
    }
};

TEST_P(DSR_Convert, CompareWithReference) {
    Run();
}

std::vector<DataTypeConversionPair> dataTypeConversionPairVector {
    {ngraph::element::f16, ngraph::element::i32},
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicConvert, DSR_Convert,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypeConversionPairVector),
        ::testing::Values(DataShapeWithUpperBound{ngraph::Shape{1, 800}, ngraph::Shape{2, 1000}},
                          DataShapeWithUpperBound{ngraph::Shape{80, 80}, ngraph::Shape{100, 100}}),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace

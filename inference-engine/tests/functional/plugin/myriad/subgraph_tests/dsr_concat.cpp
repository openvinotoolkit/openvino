// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>

#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

using DataShapeWithUpperBoundVector = std::vector<DataShapeWithUpperBound>;

struct ConcatParam {
    DataShapeWithUpperBoundVector dataShapes;
    int axis;
};
using ConcatTestParam = std::tuple<
    DataType,
    ConcatParam,
    LayerTestsUtils::TargetDevice
>;

class DSR_Concat : public testing::WithParamInterface<ConcatTestParam>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& concatParam = std::get<1>(parameters);
        targetDevice = std::get<2>(GetParam());

        const auto& inDataShapesVector = concatParam.dataShapes;
        const auto& axis = concatParam.axis;

        ngraph::NodeVector inputSubgraphVector;
        for (const auto& inDataShapes : inDataShapesVector) {
            const auto inputSubgraph = createInputSubgraphWithDSR(inDataType, inDataShapes);
            inputSubgraphVector.push_back(inputSubgraph);
        }

        const auto concat = std::make_shared<ngraph::opset3::Concat>(inputSubgraphVector, axis);

        return concat;
    }
};

TEST_P(DSR_Concat, CompareWithReference) {
    Run();
}

std::vector<ngraph::element::Type> dataTypes = {
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::i32,
};

std::vector<ConcatParam> concatParams = {
        {
            DataShapeWithUpperBoundVector{
                DataShapeWithUpperBound{DataShape{128}, DataShape{200}},
                DataShapeWithUpperBound{DataShape{256}, DataShape{300}},
                DataShapeWithUpperBound{DataShape{512}, DataShape{600}},
                DataShapeWithUpperBound{DataShape{1024}, DataShape{1200}}},
            0
        },
        {
            DataShapeWithUpperBoundVector{
                DataShapeWithUpperBound{DataShape{1, 1000}, DataShape{4, 1200}},
                DataShapeWithUpperBound{DataShape{2, 1000}, DataShape{6, 1200}},
                DataShapeWithUpperBound{DataShape{4, 1000}, DataShape{8, 1200}}},
            0
        },
        {
            DataShapeWithUpperBoundVector{
                DataShapeWithUpperBound{DataShape{128, 100}, DataShape{256, 101}},
                DataShapeWithUpperBound{DataShape{128, 200}, DataShape{256, 201}},
                DataShapeWithUpperBound{DataShape{128, 400}, DataShape{256, 401}},
                DataShapeWithUpperBound{DataShape{128, 800}, DataShape{256, 801}}},
            1
        },
        {
            DataShapeWithUpperBoundVector{
                DataShapeWithUpperBound{DataShape{3, 64, 128}, DataShape{5, 64, 256}},
                DataShapeWithUpperBound{DataShape{4, 64, 128}, DataShape{6, 64, 256}},
                DataShapeWithUpperBound{DataShape{5, 64, 128}, DataShape{7, 64, 256}}},
            0
        },
        {
            DataShapeWithUpperBoundVector{
                DataShapeWithUpperBound{DataShape{3, 64, 128}, DataShape{4, 64, 256}},
                DataShapeWithUpperBound{DataShape{3, 64, 256}, DataShape{4, 64, 512}},
                DataShapeWithUpperBound{DataShape{3, 64, 512}, DataShape{4, 64, 1024}}},
            2
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicConcat, DSR_Concat, ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(concatParams),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace

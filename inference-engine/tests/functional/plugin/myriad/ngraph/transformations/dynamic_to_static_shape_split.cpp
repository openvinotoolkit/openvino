// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/ngraph/transformations/dynamic_to_static_shape_split.hpp>

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <common_test_utils/test_common.hpp>

#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/opsets/opset5.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

struct SplitTestCase {
    ngraph::Shape dataShape;
    int64_t axis, numSplits;
};

const auto combinations = testing::Combine(
        testing::Values(
                ngraph::element::f16,
                ngraph::element::f32,
                ngraph::element::i32,
                ngraph::element::i64,
                ngraph::element::u8),
        testing::Values(
                ngraph::element::i32,
                ngraph::element::i64),
        testing::Values(
                SplitTestCase{{6}, 0, 2},
                SplitTestCase{{6, 12, 10, 24}, 1, 4},
                SplitTestCase{{6, 12}, 1, 6},
                SplitTestCase{{6, 12, 10, 24}, 3, 4},
                SplitTestCase{{6, 12, 10, 24}, -1, 4}));


class DynamicToStaticShapeSplit : public CommonTestUtils::TestsCommon,
                                  public testing::WithParamInterface<std::tuple<DataType, DataType, SplitTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& splitSetup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(dataType, idxType, splitSetup),
                                          *reference(dataType, idxType, splitSetup));
    }

protected:
std::shared_ptr<const ngraph::Function> transform(
        const ngraph::element::Type_t& dataType,
        const ngraph::element::Type_t& idxType,
        const SplitTestCase& splitSetup) const {
    const auto data = std::make_shared<ngraph::opset5::Parameter>(dataType, splitSetup.dataShape);
    const auto axis = ngraph::opset5::Constant::create(idxType, {}, {splitSetup.axis});

    const auto dims = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{splitSetup.dataShape.size()});

    const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
    const auto node = std::make_shared<ngraph::opset5::Split>(dsr, axis, splitSetup.numSplits);

    auto outputShape = node->get_output_partial_shape(0);
    const auto function = std::make_shared<ngraph::Function>(
            node->outputs(),
            ngraph::ParameterVector{data, dims},
            "Actual");
    node->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(splitSetup.dataShape.size()));

    const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeSplit}};
    vpu::DynamicToStaticShape(transformations).run_on_function(function);
    return function;
}

std::shared_ptr<const ngraph::Function> reference(
        const ngraph::element::Type_t& dataType,
        const ngraph::element::Type_t& idxType,
        const SplitTestCase& splitSetup) const {
    const auto data = std::make_shared<ngraph::opset5::Parameter>(dataType, splitSetup.dataShape);
    const auto axisScalar = ngraph::opset5::Constant::create(idxType, {}, std::vector<int64_t>{splitSetup.axis});
    const auto axisVec = ngraph::opset5::Constant::create(idxType, {1}, std::vector<int64_t>{splitSetup.axis});

    const auto dims = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{splitSetup.dataShape.size()});

    const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
    const auto node = std::make_shared<ngraph::opset5::Split>(dsr, axisScalar, splitSetup.numSplits);

    const auto dimToSplitBy = std::make_shared<ngraph::opset5::Gather>(dims,
                                                                       axisVec,
                                                                       ngraph::opset5::Constant::create(dims->get_element_type(), {1}, {0}));
    const auto splittedDim = std::make_shared<ngraph::opset5::Divide>(dimToSplitBy,
                                                                      ngraph::opset5::Constant::create(dims->get_element_type(), {1}, {splitSetup.numSplits}));
    const auto newShape = std::make_shared<ngraph::opset5::ScatterElementsUpdate>(dims,
                                                                                  axisVec,
                                                                                  splittedDim,
                                                                                  ngraph::opset5::Constant::create(dims->get_element_type(), {1}, {0}));

    ngraph::NodeVector results;
    for (size_t i = 0; i < node->get_output_size(); i++) {
        results.push_back(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(i), newShape));
    }

    return std::make_shared<ngraph::Function>(
            results,
            ngraph::ParameterVector{data, dims},
            "Expected");
}
};

TEST_P(DynamicToStaticShapeSplit, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeSplit, combinations);

}  // namespace

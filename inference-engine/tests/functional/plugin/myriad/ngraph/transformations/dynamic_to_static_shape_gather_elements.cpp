// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/ngraph/transformations/dynamic_to_static_shape_gather_elements.hpp>

#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <common_test_utils/test_common.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/op/op.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

struct GatherElementsTestCase {
    ngraph::Shape dataShape, indexShape;
    int64_t axis;
};

enum class DataShapeType {
    DYNAMIC,
    STATIC
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
        GatherElementsTestCase{{6, 4, 20, 28}, {15, 4, 20, 28}, 0},
        GatherElementsTestCase{{6, 12, 10, 24}, {3, 12, 10, 24}, 0},
        GatherElementsTestCase{{6, 12}, {6, 20}, 1},
        GatherElementsTestCase{{6, 12, 10, 24}, {6, 12, 10, 28}, 3},
        GatherElementsTestCase{{6, 12, 10, 24}, {6, 12, 10, 28}, -1},
        GatherElementsTestCase{{6, 12, 10, 24}, {15, 12, 10, 24}, -4}),
    testing::Values(
        DataShapeType::DYNAMIC,
        DataShapeType::STATIC));

class DynamicToStaticShapeGatherElements : public CommonTestUtils::TestsCommon,
                                           public testing::WithParamInterface<std::tuple<DataType, DataType, GatherElementsTestCase, DataShapeType>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& gatherElementsSetup = std::get<2>(parameters);
        const auto& dataShapeType = std::get<3>(parameters);

        ngraph::helpers::CompareFunctions(*transform(dataType, idxType, gatherElementsSetup, dataShapeType),
                                          *reference(dataType, idxType, gatherElementsSetup, dataShapeType));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& dataType,
            const ngraph::element::Type_t& idxType,
            const GatherElementsTestCase& gatherElementsSetup,
            DataShapeType dataShapeType) const {
        const auto data = std::make_shared<ngraph::opset6::Parameter>(dataType, gatherElementsSetup.dataShape);
        const auto indices = std::make_shared<ngraph::opset6::Parameter>(idxType, gatherElementsSetup.indexShape);

        const auto indicesDims = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i64, ngraph::Shape{gatherElementsSetup.indexShape.size()});
        const auto indicesDsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(indices, indicesDims);

        ngraph::ParameterVector params{data, indices, indicesDims};
        std::shared_ptr<ngraph::Node> gatherData = data;

        if (dataShapeType == DataShapeType::DYNAMIC) {
            params.push_back(std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i64, ngraph::Shape{gatherElementsSetup.dataShape.size()}));
            gatherData = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, params.back());
        }

        const auto node = std::make_shared<ngraph::opset6::GatherElements>(gatherData, indicesDsr, gatherElementsSetup.axis);

        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{node},
                params,
                "Actual");
        node->set_output_type(0, dataType, ngraph::PartialShape::dynamic(1));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeGatherElements}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& dataType,
            const ngraph::element::Type_t& idxType,
            const GatherElementsTestCase& gatherElementsSetup,
            DataShapeType dataShapeType) const {
        const auto data = std::make_shared<ngraph::opset6::Parameter>(dataType, gatherElementsSetup.dataShape);
        const auto indices = std::make_shared<ngraph::opset6::Parameter>(idxType, gatherElementsSetup.indexShape);

        const auto indicesDims = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i64, ngraph::Shape{gatherElementsSetup.indexShape.size()});
        const auto indicesDsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(indices, indicesDims);

        ngraph::ParameterVector params{data, indices, indicesDims};
        std::shared_ptr<ngraph::Node> gatherData = data;

        if (dataShapeType == DataShapeType::DYNAMIC) {
            params.push_back(std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i64, ngraph::Shape{gatherElementsSetup.dataShape.size()}));
            gatherData = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, params.back());
        }

        const auto node = std::make_shared<ngraph::op::v6::GatherElements>(gatherData, indicesDsr, gatherElementsSetup.axis);

        const auto outDsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node, indicesDims);

        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{outDsr},
                params,
                "Expected");
    }
};

TEST_P(DynamicToStaticShapeGatherElements, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeGatherElements, combinations);

}  // namespace

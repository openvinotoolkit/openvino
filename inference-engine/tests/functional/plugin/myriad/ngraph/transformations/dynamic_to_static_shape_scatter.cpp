// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_unary_elementwise.hpp>
#include <vpu/utils/error.hpp>
#include <numeric>
#include <queue>
#include <random>

namespace {

using DataType = ngraph::element::Type_t;


struct ScatterTestCase {
    ngraph::NodeTypeInfo scatterTypeInfo;
    ngraph::Shape dataShape, indicesShape, updatesShape;
    int64_t axis;
};

enum class ShapeType {
    DYNAMIC,
    STATIC
};

using ScatterParameters = std::tuple<
    DataType,
    DataType,
    ScatterTestCase,
    ShapeType>;

class DynamicToStaticShapeScatter : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<ScatterParameters> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& numericType = std::get<0>(parameters);
        const auto& integerType = std::get<1>(parameters);
        const auto& scatterSetup = std::get<2>(parameters);
        const auto& indicesUpdatesShapeType = std::get<3>(parameters);

        ngraph::helpers::CompareFunctions(
            *transform(numericType, integerType, scatterSetup, indicesUpdatesShapeType),
            *reference(numericType, integerType, scatterSetup, indicesUpdatesShapeType));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& numericType,
            const ngraph::element::Type_t& integerType,
            const ScatterTestCase& scatterSetup,
            ShapeType indicesUpdatesShapeType) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(numericType, scatterSetup.dataShape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(integerType, scatterSetup.indicesShape);
        const auto updates = std::make_shared<ngraph::opset3::Parameter>(numericType, scatterSetup.updatesShape);
        const auto axis = std::make_shared<ngraph::opset3::Constant>(integerType, ngraph::Shape{1}, std::vector<int64_t>{scatterSetup.axis});

        const auto dataDims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{scatterSetup.dataShape.size()});
        const auto dataDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dataDims);

        ngraph::ParameterVector params{data, indices, updates, dataDims};

        std::shared_ptr<ngraph::Node> scatterIndices = indices;
        std::shared_ptr<ngraph::Node> scatterUpdates = updates;
        if (indicesUpdatesShapeType == ShapeType::DYNAMIC) {
            const auto indicesDims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{scatterSetup.indicesShape.size()});
            scatterIndices = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(indices, indicesDims);
            params.push_back(indicesDims);
            const auto updatesDims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{scatterSetup.updatesShape.size()});
            scatterUpdates = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(updates, updatesDims);
            params.push_back(updatesDims);
        }

        const auto node = ngraph::helpers::getNodeSharedPtr(scatterSetup.scatterTypeInfo, {dataDSR, scatterIndices, scatterUpdates, axis});

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{node},
                params,
                "Actual");
        node->set_output_type(0, dataDSR->get_input_element_type(0), ngraph::PartialShape::dynamic(outputShape.rank()));

        const auto transformations = vpu::Transformations{{scatterSetup.scatterTypeInfo, vpu::dynamicToStaticUnaryElementwise}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& numericType,
            const ngraph::element::Type_t& integerType,
            const ScatterTestCase& scatterSetup,
            ShapeType indicesUpdatesShapeType) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(numericType, scatterSetup.dataShape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(integerType, scatterSetup.indicesShape);
        const auto updates = std::make_shared<ngraph::opset3::Parameter>(numericType, scatterSetup.updatesShape);
        const auto axis = std::make_shared<ngraph::opset3::Constant>(integerType, ngraph::Shape{1}, std::vector<int64_t>{scatterSetup.axis});

        const auto dataDims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{scatterSetup.dataShape.size()});
        const auto dataDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dataDims);

        ngraph::ParameterVector params{data, indices, updates, dataDims};

        std::shared_ptr<ngraph::Node> scatterIndices = indices;
        std::shared_ptr<ngraph::Node> scatterUpdates = updates;
        if (indicesUpdatesShapeType == ShapeType::DYNAMIC) {
            const auto indicesDims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{scatterSetup.indicesShape.size()});
            scatterIndices = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(indices, indicesDims);
            params.push_back(indicesDims);
            const auto updatesDims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{scatterSetup.updatesShape.size()});
            scatterUpdates = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(updates, updatesDims);
            params.push_back(updatesDims);
        }


        const auto node = ngraph::helpers::getNodeSharedPtr(scatterSetup.scatterTypeInfo, {dataDSR, scatterIndices, scatterUpdates, axis});

        std::shared_ptr<ngraph::Node> outNode = node;
        const auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node, dataDims);

        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{outDSR},
                params,
                "Expected");
    }
};

TEST_P(DynamicToStaticShapeScatter, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeScatter, testing::Combine(
    testing::Values(
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8),
    testing::Values(
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8),
    testing::Values(
        ScatterTestCase{ngraph::opset3::ScatterUpdate::type_info, {1000, 256, 10, 15}, {125, 20}, {1000, 125, 20, 10, 15}, 1},
        ScatterTestCase{ngraph::opset5::ScatterElementsUpdate::type_info, {300}, {300}, {300}, 0}),
    testing::Values(
        ShapeType::DYNAMIC,
        ShapeType::STATIC)
));

}  // namespace

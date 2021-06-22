// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>
#include <ngraph/op/parameter.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <numeric>
#include <random>
#include <ngraph/opsets/opset3.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_gather.hpp>
#include <queue>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/utils/error.hpp>
#include <vpu/ngraph/utilities.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

struct GatherTestCase {
    ngraph::Shape data_shape, index_shape;
    int64_t axis, first_split_point, second_split_point;
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
            ngraph::element::i64,
            ngraph::element::u8),
    testing::Values(
            GatherTestCase{{6}, {15, 4, 20, 28}, 0, 0, 0},
            GatherTestCase{{6, 12, 10, 24}, {6}, 0, 0, 1},
            GatherTestCase{{6, 12}, {15, 4, 20, 28}, 1, 1, 2},
            GatherTestCase{{6, 12, 10, 24}, {15, 4, 20, 28}, 3, 3, 4},
            GatherTestCase{{6, 12, 10, 24}, {15, 4, 20, 28}, -1, 3, 4},
            GatherTestCase{{6, 12, 10, 24}, {15, 4, 20, 28}, -4, 0, 1}));


class DynamicToStaticShapeGatherDataDSR : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataType, GatherTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& idx_type = std::get<1>(parameters);
        const auto& gather_setup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(data_type, idx_type, gather_setup),
                *reference(data_type, idx_type, gather_setup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
        const ngraph::element::Type_t& data_type,
        const ngraph::element::Type_t& idx_type,
        const GatherTestCase& gather_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, gather_setup.data_shape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(idx_type, gather_setup.index_shape);
        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gather_setup.axis});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{gather_setup.data_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = std::make_shared<ngraph::opset3::Gather>(dsr, indices, axis);

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{node},
            ngraph::ParameterVector{data, dims, indices},
            "Actual");
        node->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(
                gather_setup.data_shape.size() + gather_setup.index_shape.size() - 1));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeGather}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& data_type,
            const ngraph::element::Type_t& idx_type,
            const GatherTestCase& gather_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, gather_setup.data_shape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(idx_type, gather_setup.index_shape);
        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gather_setup.axis});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{gather_setup.data_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = std::make_shared<ngraph::opset3::Gather>(dsr, indices, axis);

        const auto indices_shape = ngraph::opset3::Constant::create(dims->get_element_type(), {gather_setup.index_shape.size()}, gather_setup.index_shape);
        ngraph::OutputVector output_dims;
        if (gather_setup.first_split_point) {
            output_dims.push_back(vpu::gatherShapeElements(dims, 0, gather_setup.first_split_point));
        }
        if (!gather_setup.index_shape.empty())
            output_dims.push_back(indices_shape);
        if (gather_setup.first_split_point + 1 < gather_setup.data_shape.size()) {
            output_dims.push_back(vpu::gatherShapeElements(
                dims,
                gather_setup.second_split_point,
                gather_setup.data_shape.size() - gather_setup.second_split_point));
        }
        const auto output_shape = std::make_shared<ngraph::opset3::Concat>(output_dims, 0);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node, output_shape);
        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr1},
            ngraph::ParameterVector{data, dims, indices},
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeGatherDataDSR, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeGatherDataDSR, combinations);

class DynamicToStaticShapeGatherIdxDSR : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataType, GatherTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& idx_type = std::get<1>(parameters);
        const auto& gather_setup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(data_type, idx_type, gather_setup),
                *reference(data_type, idx_type, gather_setup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
        const ngraph::element::Type_t& data_type,
        const ngraph::element::Type_t& idx_type,
        const GatherTestCase& gather_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, gather_setup.data_shape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(idx_type, gather_setup.index_shape);
        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gather_setup.axis});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{gather_setup.index_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(indices, dims);
        const auto node = std::make_shared<ngraph::opset3::Gather>(data, dsr, axis);

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{node},
            ngraph::ParameterVector{data, dims, indices},
            "Actual");
        node->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(
                gather_setup.data_shape.size() + gather_setup.index_shape.size() - 1));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeGather}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& data_type,
            const ngraph::element::Type_t& idx_type,
            const GatherTestCase& gather_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, gather_setup.data_shape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(idx_type, gather_setup.index_shape);
        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gather_setup.axis});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{gather_setup.index_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(indices, dims);
        const auto node = std::make_shared<ngraph::opset3::Gather>(data, dsr, axis);

        const auto data_shape = ngraph::opset3::Constant::create(dims->get_element_type(), {gather_setup.data_shape.size()}, gather_setup.data_shape);

        ngraph::OutputVector output_dims;
        if (gather_setup.first_split_point) {
            output_dims.push_back(vpu::gatherShapeElements(data_shape, 0, gather_setup.first_split_point));
        }
        if (!gather_setup.index_shape.empty())
            output_dims.push_back(dims);
        if (gather_setup.first_split_point + 1 < gather_setup.data_shape.size()) {
            output_dims.push_back(vpu::gatherShapeElements(
                data_shape,
                gather_setup.second_split_point,
                gather_setup.data_shape.size() - gather_setup.second_split_point));
        }
        const auto output_shape = std::make_shared<ngraph::opset3::Concat>(output_dims, 0);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node, output_shape);
        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr1},
            ngraph::ParameterVector{data, dims, indices},
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeGatherIdxDSR, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeGatherIdxDSR, combinations);

class DynamicToStaticShapeGather : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataType, GatherTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& idx_type = std::get<1>(parameters);
        const auto& gather_setup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(data_type, idx_type, gather_setup),
                *reference(data_type, idx_type, gather_setup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
        const ngraph::element::Type_t& data_type,
        const ngraph::element::Type_t& idx_type,
        const GatherTestCase& gather_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, gather_setup.data_shape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(idx_type, gather_setup.index_shape);
        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gather_setup.axis});

        const auto data_dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{gather_setup.data_shape.size()});
        const auto indices_dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{gather_setup.index_shape.size()});

        const auto data_dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, data_dims);
        const auto indices_dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(indices, indices_dims);
        const auto node = std::make_shared<ngraph::opset3::Gather>(data_dsr, indices_dsr, axis);

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{node},
            ngraph::ParameterVector{data, data_dims, indices, indices_dims},
            "Actual");
        node->set_output_type(0, node->get_input_element_type(0), ngraph::PartialShape::dynamic(
                gather_setup.data_shape.size() + gather_setup.index_shape.size() - 1));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeGather}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& data_type,
            const ngraph::element::Type_t& idx_type,
            const GatherTestCase& gather_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, gather_setup.data_shape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(idx_type, gather_setup.index_shape);
        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gather_setup.axis});

        const auto data_dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{gather_setup.data_shape.size()});
        const auto indices_dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{gather_setup.index_shape.size()});

        const auto data_dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, data_dims);
        const auto indices_dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(indices, indices_dims);
        const auto node = std::make_shared<ngraph::opset3::Gather>(data_dsr, indices_dsr, axis);

        ngraph::OutputVector output_dims;
        if (gather_setup.first_split_point) {
            output_dims.push_back(vpu::gatherShapeElements(data_dims, 0, gather_setup.first_split_point));
        }
        if (!gather_setup.index_shape.empty())
            output_dims.push_back(indices_dims);
        if (gather_setup.first_split_point + 1 < gather_setup.data_shape.size()) {
            output_dims.push_back(vpu::gatherShapeElements(
                data_dims,
                gather_setup.second_split_point,
                gather_setup.data_shape.size() - gather_setup.second_split_point));
        }
        const auto output_shape = std::make_shared<ngraph::opset3::Concat>(output_dims, 0);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node, output_shape);
        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr1},
            ngraph::ParameterVector{data, data_dims, indices, indices_dims},
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeGather, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeGather, combinations);

}  // namespace

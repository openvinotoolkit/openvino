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
#include <vpu/ngraph/transformations/dynamic_to_static_shape_reduce.hpp>
#include <queue>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/utils/error.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

struct ReduceTestCase {
    ngraph::Shape data_shape;
    std::vector<int64_t> axes;
    bool keep_dims;
    std::vector<int64_t> gather_indices;
};

const auto arithmetic_combinations = testing::Combine(
    testing::Values(
            ngraph::opset3::ReduceMax::type_info,
            ngraph::opset3::ReduceMean::type_info,
            ngraph::opset3::ReduceMin::type_info,
            ngraph::opset3::ReduceProd::type_info,
            ngraph::opset3::ReduceSum::type_info),
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
            // data_shape, axes, keep_dims, gather_indices, axes_shape
            ReduceTestCase{{1, 3, 224, 224}, {2, 3}, true, {1, 1}},
            ReduceTestCase{{1, 3, 224, 224}, {2, 3}, false, {0, 1}},
            ReduceTestCase{{1, 3, 224, 224}, {0, 1, 2, 3}, true, {1, 1, 1, 1}},
            ReduceTestCase{{1, 3, 224, 224}, {1, 3}, false, {0, 2}},
            ReduceTestCase{{4}, {0}, true, {1}}));

const auto logical_combinations = testing::Combine(
        testing::Values(
            ngraph::opset3::ReduceLogicalAnd::type_info,
            ngraph::opset3::ReduceLogicalOr::type_info),
        testing::Values(ngraph::element::boolean),
        testing::Values(
            ngraph::element::i32,
            ngraph::element::i64,
            ngraph::element::u8),
        testing::Values(
            // data_shape, axes, keep_dims, gather_indices
            ReduceTestCase{{1, 3, 224, 224}, {2, 3}, true, {1, 1}},
            ReduceTestCase{{1, 3, 224, 224}, {2, 3}, false, {0, 1}},
            ReduceTestCase{{1, 3, 224, 224}, {0, 1, 2, 3}, true, {1, 1, 1, 1}},
            ReduceTestCase{{1, 3, 224, 224}, {1, 3}, false, {0, 2}},
            ReduceTestCase{{4}, {0}, true, {1}}));

class DynamicToStaticShapeReduce: public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<ngraph::NodeTypeInfo, DataType, DataType, ReduceTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& reduce_type = std::get<0>(parameters);
        const auto& data_type = std::get<1>(parameters);
        const auto& axes_type = std::get<2>(parameters);
        const auto& reduce_setup = std::get<3>(parameters);

        ngraph::helpers::CompareFunctions(*transform(reduce_type, data_type, axes_type, reduce_setup),
                *reference(reduce_type, data_type, axes_type, reduce_setup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::NodeTypeInfo type_info,
            const ngraph::element::Type_t& data_type,
            const ngraph::element::Type_t& axes_type,
            const ReduceTestCase& reduce_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, reduce_setup.data_shape);
        const auto axes = ngraph::opset3::Constant::create(axes_type, {reduce_setup.axes.size()}, reduce_setup.axes);

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{reduce_setup.data_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = ngraph::helpers::getNodeSharedPtr(type_info, {dsr, axes});

        if (auto arithmetic_reduce = std::dynamic_pointer_cast<ngraph::op::util::ArithmeticReductionKeepDims>(node))
            arithmetic_reduce->set_keep_dims(reduce_setup.keep_dims);
        else if (auto logical_reduce = std::dynamic_pointer_cast<ngraph::op::util::LogicalReductionKeepDims>(node))
            logical_reduce->set_keep_dims(reduce_setup.keep_dims);
        node->validate_and_infer_types();

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{node},
            ngraph::ParameterVector{data, dims},
            "Actual");
        node->set_output_type(0, data_type, ngraph::PartialShape::dynamic(node->get_output_partial_shape(0).rank()));
        const auto transformations = vpu::Transformations{{type_info, vpu::dynamicToStaticShapeReduce}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::NodeTypeInfo type_info,
            const ngraph::element::Type_t& data_type,
            const ngraph::element::Type_t& axes_type,
            const ReduceTestCase& reduce_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, reduce_setup.data_shape);
        const auto axes = ngraph::opset3::Constant::create(axes_type, {reduce_setup.axes.size()}, reduce_setup.axes);

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{reduce_setup.data_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = ngraph::helpers::getNodeSharedPtr(type_info, {dsr, axes});

        if (auto arithmetic_reduce = std::dynamic_pointer_cast<ngraph::op::util::ArithmeticReductionKeepDims>(node))
            arithmetic_reduce->set_keep_dims(reduce_setup.keep_dims);
        else if (auto logical_reduce = std::dynamic_pointer_cast<ngraph::op::util::LogicalReductionKeepDims>(node))
            logical_reduce->set_keep_dims(reduce_setup.keep_dims);
        node->validate_and_infer_types();

        ngraph::Output<ngraph::Node> output_shape;
        if (reduce_setup.keep_dims) {
            output_shape = std::make_shared<ngraph::opset3::ScatterElementsUpdate>(
                    dims,
                    ngraph::opset3::Constant::create(ngraph::element::i64, {reduce_setup.axes.size()}, reduce_setup.axes),
                    ngraph::opset3::Constant::create(ngraph::element::i64, {reduce_setup.gather_indices.size()}, reduce_setup.gather_indices),
                    ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {0}));
        } else {
            output_shape = std::make_shared<ngraph::opset3::Gather>(
                    dims,
                    ngraph::opset3::Constant::create(ngraph::element::i64, {reduce_setup.gather_indices.size()}, reduce_setup.gather_indices),
                    ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {0}));
        }
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node, output_shape);
        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{dsr1},
                ngraph::ParameterVector{data, dims},
                "Expected");
    }
};

TEST_P(DynamicToStaticShapeReduce, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_Arithmetic, DynamicToStaticShapeReduce, arithmetic_combinations);
INSTANTIATE_TEST_SUITE_P(smoke_Logical, DynamicToStaticShapeReduce, logical_combinations);

}  // namespace

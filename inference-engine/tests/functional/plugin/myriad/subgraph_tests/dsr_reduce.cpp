// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

struct ReduceTestCase {
    ngraph::Shape data_shape;
    std::vector<int64_t> axes;
    bool keep_dims;
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
                // data_shape, axes, keep_dims
                ReduceTestCase{{1, 3, 224, 224}, {2, 3}, true},
                ReduceTestCase{{1, 3, 224, 224}, {2, 3}, false},
                ReduceTestCase{{1, 3, 224, 224}, {0, 1, 2, 3}, true},
                ReduceTestCase{{1, 3, 224, 224}, {1, 3}, false},
                ReduceTestCase{{4}, {0}, true}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD));

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
                // data_shape, axes, keep_dims
                ReduceTestCase{{1, 3, 224, 224}, {2, 3}, true},
                ReduceTestCase{{1, 3, 224, 224}, {2, 3}, false},
                ReduceTestCase{{1, 3, 224, 224}, {0, 1, 2, 3}, true},
                ReduceTestCase{{1, 3, 224, 224}, {1, 3}, false},
                ReduceTestCase{{4}, {0}, true}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD));


using Parameters = std::tuple<
    ngraph::NodeTypeInfo,
    DataType,
    DataType,
    ReduceTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_Reduce : public testing::WithParamInterface<Parameters>,
        virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& reduce_type = std::get<0>(parameters);
        const auto& data_type = std::get<1>(parameters);
        const auto& axes_type = std::get<2>(parameters);
        const auto& reduce_setup = std::get<3>(parameters);
        targetDevice = std::get<4>(parameters);

        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, reduce_setup.data_shape);
        const auto axes = ngraph::opset3::Constant::create(axes_type, {reduce_setup.axes.size()}, reduce_setup.axes);

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{reduce_setup.data_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = ngraph::helpers::getNodeSharedPtr(reduce_type, {dsr, axes});

        if (auto arithmetic_reduce = std::dynamic_pointer_cast<ngraph::op::util::ArithmeticReductionKeepDims>(node))
            arithmetic_reduce->set_keep_dims(reduce_setup.keep_dims);
        else if (auto logical_reduce = std::dynamic_pointer_cast<ngraph::op::util::LogicalReductionKeepDims>(node))
            logical_reduce->set_keep_dims(reduce_setup.keep_dims);
        node->validate_and_infer_types();
        const auto result = std::make_shared<ngraph::opset3::Result>(node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                ngraph::ParameterVector{data, dims}, "DSR-Reduce");
    }
};

TEST_P(DSR_Reduce, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicArithmeticReduce, DSR_Reduce, arithmetic_combinations);
INSTANTIATE_TEST_CASE_P(DISABLED_DynamicLogicalReduce, DSR_Reduce, logical_combinations);

}  // namespace

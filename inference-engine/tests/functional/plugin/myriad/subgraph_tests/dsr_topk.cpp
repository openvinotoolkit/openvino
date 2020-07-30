// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

struct TopKTestCase {
    ngraph::Shape data_shape;
    int64_t k, axis, first_split_point, second_split_point;
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
            TopKTestCase{{6}, 5, 0, 0, 0},
            TopKTestCase{{6, 12, 10, 24}, 5, 0, 0, 1},
            TopKTestCase{{6, 12}, 10, 1, 1, 2},
            TopKTestCase{{6, 12, 10, 24}, 7, 3, 3, 4},
            TopKTestCase{{6, 12, 10, 24}, 20, -1, 3, 4},
            TopKTestCase{{6, 12, 10, 24}, 3, -4, 0, 1}),
    testing::Values(CommonTestUtils::DEVICE_MYRIAD));


using Parameters = std::tuple<
    DataType,
    DataType,
    TopKTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_TopK_Const : public testing::WithParamInterface<Parameters>,
        virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& idx_type = std::get<1>(parameters);
        const auto& topk_setup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, topk_setup.data_shape);
        const auto k = ngraph::opset3::Constant::create(idx_type, {}, std::vector<int64_t>{topk_setup.k});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{topk_setup.data_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = std::make_shared<ngraph::opset3::TopK>(dsr, k, topk_setup.axis, "max", "value");

        // tests are capable to compare functions with one result only, but TopK has 2 of them  and they are of different types
        ngraph::OutputVector converted;
        for (const auto& result : {node->output(0), node->output(1)}) {
            converted.push_back(std::make_shared<ngraph::opset3::Convert>(result, ngraph::element::f32));
        }
        const auto tests_wa = std::make_shared<ngraph::opset3::Concat>(converted, topk_setup.axis);
        const auto result = std::make_shared<ngraph::opset3::Result>(tests_wa);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                ngraph::ParameterVector{data, dims}, "DSR-TopKConst");
    }
};

TEST_P(DSR_TopK_Const, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicTopKConst, DSR_TopK_Const, combinations);

class DSR_TopK : public testing::WithParamInterface<Parameters>,
        virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& idx_type = std::get<1>(parameters);
        const auto& topk_setup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, topk_setup.data_shape);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{topk_setup.data_shape.size()});

        const auto gather = std::make_shared<ngraph::opset3::Gather>(dims,
                                                                     ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {topk_setup.axis}),
                                                                     ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {0}));
        const auto upper_bound = ngraph::opset3::Constant::create(dims->get_element_type(), {1}, {100});
        const auto concat = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{upper_bound, gather}, 0);
        const auto k = std::make_shared<ngraph::opset3::ReduceMin>(concat, ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {0}), false);

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = std::make_shared<ngraph::opset3::TopK>(dsr, k, topk_setup.axis, "max", "value");

        // tests are capable to compare functions with one result only, but TopK has 2 of them  and they are of different types
        ngraph::OutputVector converted;
        for (const auto& result : {node->output(0), node->output(1)}) {
            converted.push_back(std::make_shared<ngraph::opset3::Convert>(result, ngraph::element::f32));
        }
        const auto tests_wa = std::make_shared<ngraph::opset3::Concat>(converted, topk_setup.axis);
        const auto result = std::make_shared<ngraph::opset3::Result>(tests_wa);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                ngraph::ParameterVector{data, dims}, "DSR-TopK");
    }
};

TEST_P(DSR_TopK, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicTopKConst, DSR_TopK, combinations);

}  // namespace

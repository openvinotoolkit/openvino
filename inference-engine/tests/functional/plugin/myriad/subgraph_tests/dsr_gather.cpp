// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

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
            GatherTestCase{{6, 12, 10, 24}, {15, 4, 20, 28}, -4, 0, 1}),
    testing::Values(CommonTestUtils::DEVICE_MYRIAD));


using Parameters = std::tuple<
    DataType,
    DataType,
    GatherTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_GatherData : public testing::WithParamInterface<Parameters>,
        public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& idx_type = std::get<1>(parameters);
        const auto& gather_setup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, gather_setup.data_shape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(idx_type, gather_setup.index_shape);
        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gather_setup.axis});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{gather_setup.data_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = std::make_shared<ngraph::opset3::Gather>(dsr, indices, axis);

        const auto result = std::make_shared<ngraph::opset3::Result>(node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                ngraph::ParameterVector{data, indices, dims}, "DSR-GatherData");
    }
};

TEST_P(DSR_GatherData, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicGatherData, DSR_GatherData, combinations);

class DSR_GatherIdx : public testing::WithParamInterface<Parameters>,
        public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& idx_type = std::get<1>(parameters);
        const auto& gather_setup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, gather_setup.data_shape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(idx_type, gather_setup.index_shape);
        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gather_setup.axis});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{gather_setup.index_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(indices, dims);
        const auto node = std::make_shared<ngraph::opset3::Gather>(data, dsr, axis);

        const auto result = std::make_shared<ngraph::opset3::Result>(node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                ngraph::ParameterVector{data, indices, dims}, "DSR-GatherIdx");
    }
};

TEST_P(DSR_GatherIdx, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicGatherIdx, DSR_GatherIdx, combinations);

class DSR_Gather : public testing::WithParamInterface<Parameters>,
        public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& idx_type = std::get<1>(parameters);
        const auto& gather_setup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, gather_setup.data_shape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(idx_type, gather_setup.index_shape);
        const auto axis = ngraph::opset3::Constant::create(ngraph::element::i32, {1}, std::vector<int64_t>{gather_setup.axis});

        const auto data_dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{gather_setup.data_shape.size()});
        const auto indices_dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{gather_setup.index_shape.size()});

        const auto data_dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, data_dims);
        const auto indices_dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(indices, indices_dims);
        const auto node = std::make_shared<ngraph::opset3::Gather>(data_dsr, indices_dsr, axis);

        const auto result = std::make_shared<ngraph::opset3::Result>(node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                ngraph::ParameterVector{data, indices, data_dims, indices_dims}, "DSR-Gather");
    }
};

TEST_P(DSR_Gather, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicGatherIdx, DSR_Gather, combinations);

}  // namespace

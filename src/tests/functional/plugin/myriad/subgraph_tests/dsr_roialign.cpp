// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

struct ROIAlignTestCase {
    ngraph::Shape data_shape;
    uint64_t num_rois, pooled_h, pooled_w, sampling_ratio;
    float spatial_scale;
    std::string mode;
};

using Parameters = std::tuple<
    DataType,
    DataType,
    ROIAlignTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_ROIAlignDataDSR : public testing::WithParamInterface<Parameters>,
        virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& float_type = std::get<0>(parameters);
        const auto& integer_type = std::get<1>(parameters);
        const auto& roialign_setup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto data = std::make_shared<ngraph::opset3::Parameter>(float_type, roialign_setup.data_shape);
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(float_type, ngraph::Shape{roialign_setup.num_rois, 4});
        const auto rois = std::make_shared<ngraph::opset3::Parameter>(integer_type, ngraph::Shape{roialign_setup.num_rois});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{data->get_shape().size()});
        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto node = std::make_shared<ngraph::opset3::ROIAlign>(dsr, boxes, rois,
                roialign_setup.pooled_h, roialign_setup.pooled_w, roialign_setup.sampling_ratio, roialign_setup.spatial_scale, roialign_setup.mode);

        const auto result = std::make_shared<ngraph::opset3::Result>(node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{data, dims, boxes, rois}, "DSR-ROIAlign");
    }
};

TEST_P(DSR_ROIAlignDataDSR, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(DISABLED_DynamicROIAlignDataDSR, DSR_ROIAlignDataDSR,
    ::testing::Combine(
        ::testing::Values(
                    ngraph::element::f16,
                    ngraph::element::f32),
        ::testing::Values(
                    ngraph::element::i32,
                    ngraph::element::i64,
                    ngraph::element::u8),
        //data_shape, num_rois, pooled_h, pooled_w, sampling_ratio, spatial_scale, mode
        ::testing::Values(
                    ROIAlignTestCase{{7, 256, 200, 200}, 1000, 6, 6, 2, 16., "avg"},
                    ROIAlignTestCase{{7, 256, 200, 200}, 1000, 7, 6, 2, 16., "max"}),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));


class DSR_ROIAlignROIDSR : public testing::WithParamInterface<Parameters>,
        virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& float_type = std::get<0>(parameters);
        const auto& integer_type = std::get<1>(parameters);
        const auto& roialign_setup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto data = std::make_shared<ngraph::opset3::Parameter>(float_type, roialign_setup.data_shape);
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(float_type, ngraph::Shape{roialign_setup.num_rois, 4});
        const auto rois = std::make_shared<ngraph::opset3::Parameter>(integer_type, ngraph::Shape{roialign_setup.num_rois});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{rois->get_shape().size()});
        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(rois, dims);

        const auto node = std::make_shared<ngraph::opset3::ROIAlign>(data, boxes, dsr,
                roialign_setup.pooled_h, roialign_setup.pooled_w, roialign_setup.sampling_ratio, roialign_setup.spatial_scale, roialign_setup.mode);

        const auto result = std::make_shared<ngraph::opset3::Result>(node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{data, dims, boxes, rois}, "DSR-ROIAlign");
    }
};

TEST_P(DSR_ROIAlignROIDSR, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_DynamicROIAlign, DSR_ROIAlignROIDSR,
    ::testing::Combine(
        ::testing::Values(
                    ngraph::element::f16,
                    ngraph::element::f32),
        ::testing::Values(
                    ngraph::element::i32,
                    ngraph::element::i64,
                    ngraph::element::u8),
        //data_shape, num_rois, pooled_h, pooled_w, sampling_ratio, spatial_scale, mode
        ::testing::Values(
                    ROIAlignTestCase{{7, 256, 200, 200}, 1000, 6, 6, 2, 16., "avg"},
                    ROIAlignTestCase{{7, 256, 200, 200}, 1000, 7, 6, 2, 16., "max"}),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

class DSR_ROIAlign : public testing::WithParamInterface<Parameters>,
        virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& float_type = std::get<0>(parameters);
        const auto& integer_type = std::get<1>(parameters);
        const auto& roialign_setup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto data = std::make_shared<ngraph::opset3::Parameter>(float_type, roialign_setup.data_shape);
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(float_type, ngraph::Shape{roialign_setup.num_rois, 4});
        const auto rois = std::make_shared<ngraph::opset3::Parameter>(integer_type, ngraph::Shape{roialign_setup.num_rois});

        const auto roi_dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{rois->get_shape().size()});
        const auto roi_dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(rois, roi_dims);

        const auto data_dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{data->get_shape().size()});
        const auto data_dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, data_dims);

        const auto node = std::make_shared<ngraph::opset3::ROIAlign>(data_dsr, boxes, roi_dsr,
                roialign_setup.pooled_h, roialign_setup.pooled_w, roialign_setup.sampling_ratio, roialign_setup.spatial_scale, roialign_setup.mode);

        const auto result = std::make_shared<ngraph::opset3::Result>(node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                ngraph::ParameterVector{data, data_dims, boxes, rois, roi_dims}, "DSR-ROIAlign");
    }
};

TEST_P(DSR_ROIAlign, CompareWithReference) {
    Run();
}

// #-30909
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_DynamicROIAlign, DSR_ROIAlign,
    ::testing::Combine(
        ::testing::Values(
                    ngraph::element::f16,
                    ngraph::element::f32),
        ::testing::Values(
                    ngraph::element::i32,
                    ngraph::element::i64,
                    ngraph::element::u8),
        //data_shape, num_rois, pooled_h, pooled_w, sampling_ratio, spatial_scale, mode
        ::testing::Values(
                    ROIAlignTestCase{{7, 256, 200, 200}, 1000, 6, 6, 2, 16., "avg"},
                    ROIAlignTestCase{{7, 256, 200, 200}, 1000, 7, 6, 2, 16., "max"}),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace

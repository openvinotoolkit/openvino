// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_roialign.hpp>
#include <vpu/utils/error.hpp>
#include <numeric>
#include <queue>
#include <random>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

struct ROIAlignTestCase {
    ngraph::Shape data_shape;
    uint64_t num_rois, pooled_h, pooled_w, sampling_ratio;
    float spatial_scale;
    std::string mode;
};


class DynamicToStaticShapeROIAlignDataDSR : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataType, ROIAlignTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& float_type = std::get<0>(parameters);
        const auto& integer_type = std::get<1>(parameters);
        const auto& roialign_setup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(float_type, integer_type, roialign_setup),
                *reference(float_type, integer_type, roialign_setup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& float_type,
            const ngraph::element::Type_t& integer_type,
            const ROIAlignTestCase& roialign_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(float_type, roialign_setup.data_shape);
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(float_type, ngraph::Shape{roialign_setup.num_rois, 4});
        const auto rois = std::make_shared<ngraph::opset3::Parameter>(integer_type, ngraph::Shape{roialign_setup.num_rois});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{data->get_shape().size()});
        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto node = std::make_shared<ngraph::opset3::ROIAlign>(dsr, boxes, rois,
                roialign_setup.pooled_h, roialign_setup.pooled_w, roialign_setup.sampling_ratio, roialign_setup.spatial_scale, roialign_setup.mode);

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{node},
                ngraph::ParameterVector{data, boxes, rois, dims},
                "Actual");
        node->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(outputShape.rank()));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeROIAlign}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& float_type,
            const ngraph::element::Type_t& integer_type,
            const ROIAlignTestCase& roialign_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(float_type, roialign_setup.data_shape);
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(float_type, ngraph::Shape{roialign_setup.num_rois, 4});
        const auto rois = std::make_shared<ngraph::opset3::Parameter>(integer_type, ngraph::Shape{roialign_setup.num_rois});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{data->get_shape().size()});
        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto node = std::make_shared<ngraph::opset3::ROIAlign>(dsr, boxes, rois,
                roialign_setup.pooled_h, roialign_setup.pooled_w, roialign_setup.sampling_ratio, roialign_setup.spatial_scale, roialign_setup.mode);


        const auto c_index = ngraph::opset3::Constant::create(dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{1});
        const auto c_axis = ngraph::opset3::Constant::create(dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{0});
        const auto c = std::make_shared<ngraph::opset3::Gather>(dims, c_index, c_axis);

        const auto num_rois = ngraph::opset3::Constant::create(dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{roialign_setup.num_rois});
        const auto pooled_h = ngraph::opset3::Constant::create(dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{roialign_setup.pooled_h});
        const auto pooled_w = ngraph::opset3::Constant::create(dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{roialign_setup.pooled_w});

        const auto output_shape = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{num_rois, c, pooled_h, pooled_w}, 0);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node, output_shape);
        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{dsr1},
                ngraph::ParameterVector{data, boxes, rois, dims},
                "Expected");
    }
};

TEST_P(DynamicToStaticShapeROIAlignDataDSR, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeROIAlignDataDSR, testing::Combine(
    testing::Values(
        ngraph::element::f16,
        ngraph::element::f32),
   testing::Values(
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8),
    //data_shape, num_rois, pooled_h, pooled_w, sampling_ratio, spatial_scale, mode
    testing::Values(
        ROIAlignTestCase{{7, 256, 200, 200}, 1000, 6, 6, 2, 16., "avg"},
        ROIAlignTestCase{{7, 256, 200, 200}, 1000, 7, 6, 2, 16., "max"})));

class DynamicToStaticShapeROIAlignROIDSR : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataType, ROIAlignTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& float_type = std::get<0>(parameters);
        const auto& integer_type = std::get<1>(parameters);
        const auto& roialign_setup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(float_type, integer_type, roialign_setup),
                                          *reference(float_type, integer_type, roialign_setup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& float_type,
            const ngraph::element::Type_t& integer_type,
            const ROIAlignTestCase& roialign_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(float_type, roialign_setup.data_shape);
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(float_type, ngraph::Shape{roialign_setup.num_rois, 4});
        const auto rois = std::make_shared<ngraph::opset3::Parameter>(integer_type, ngraph::Shape{roialign_setup.num_rois});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{rois->get_shape().size()});
        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(rois, dims);

        const auto node = std::make_shared<ngraph::opset3::ROIAlign>(data, boxes, dsr,
                roialign_setup.pooled_h, roialign_setup.pooled_w, roialign_setup.sampling_ratio, roialign_setup.spatial_scale, roialign_setup.mode);

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{node},
                ngraph::ParameterVector{data, boxes, rois, dims},
                "Actual");
        node->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(outputShape.rank()));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeROIAlign}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& float_type,
            const ngraph::element::Type_t& integer_type,
            const ROIAlignTestCase& roialign_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(float_type, roialign_setup.data_shape);
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(float_type, ngraph::Shape{roialign_setup.num_rois, 4});
        const auto rois = std::make_shared<ngraph::opset3::Parameter>(integer_type, ngraph::Shape{roialign_setup.num_rois});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{rois->get_shape().size()});
        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(rois, dims);

        const auto node = std::make_shared<ngraph::opset3::ROIAlign>(data, boxes, dsr,
                roialign_setup.pooled_h, roialign_setup.pooled_w, roialign_setup.sampling_ratio, roialign_setup.spatial_scale, roialign_setup.mode);

        const auto data_shape = ngraph::opset3::Constant::create(dims->get_element_type(), ngraph::Shape{4}, roialign_setup.data_shape);
        const auto c_index = ngraph::opset3::Constant::create(dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{1});
        const auto c_axis = ngraph::opset3::Constant::create(dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{0});
        const auto c = std::make_shared<ngraph::opset3::Gather>(data_shape, c_index, c_axis);

        const auto pooled_h = ngraph::opset3::Constant::create(dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{roialign_setup.pooled_h});
        const auto pooled_w = ngraph::opset3::Constant::create(dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{roialign_setup.pooled_w});

        const auto output_shape = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{dims, c, pooled_h, pooled_w}, 0);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node, output_shape);
        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{dsr1},
                ngraph::ParameterVector{data, boxes, rois, dims},
                "Expected");
    }
};

TEST_P(DynamicToStaticShapeROIAlignROIDSR, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeROIAlignROIDSR, testing::Combine(
        testing::Values(
                ngraph::element::f16,
                ngraph::element::f32),
        testing::Values(
                ngraph::element::i32,
                ngraph::element::i64,
                ngraph::element::u8),
        //data_shape, num_rois, pooled_h, pooled_w, sampling_ratio, spatial_scale, mode
        testing::Values(
                ROIAlignTestCase{{7, 256, 200, 200}, 1000, 6, 6, 2, 16., "avg"},
                ROIAlignTestCase{{7, 256, 200, 200}, 1000, 7, 6, 2, 16., "max"})));


class DynamicToStaticShapeROIAlign : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataType, ROIAlignTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& float_type = std::get<0>(parameters);
        const auto& integer_type = std::get<1>(parameters);
        const auto& roialign_setup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(float_type, integer_type, roialign_setup),
                                          *reference(float_type, integer_type, roialign_setup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& float_type,
            const ngraph::element::Type_t& integer_type,
            const ROIAlignTestCase& roialign_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(float_type, roialign_setup.data_shape);
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(float_type, ngraph::Shape{roialign_setup.num_rois, 4});
        const auto rois = std::make_shared<ngraph::opset3::Parameter>(integer_type, ngraph::Shape{roialign_setup.num_rois});

        const auto roi_dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{rois->get_shape().size()});
        const auto roi_dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(rois, roi_dims);

        const auto data_dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{data->get_shape().size()});
        const auto data_dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, data_dims);

        const auto node = std::make_shared<ngraph::opset3::ROIAlign>(data_dsr, boxes, roi_dsr,
                roialign_setup.pooled_h, roialign_setup.pooled_w, roialign_setup.sampling_ratio, roialign_setup.spatial_scale, roialign_setup.mode);

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{node},
                ngraph::ParameterVector{data, boxes, rois, roi_dims, data_dims},
                "Actual");
        node->set_output_type(0, data_dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(outputShape.rank()));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeROIAlign}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& float_type,
            const ngraph::element::Type_t& integer_type,
            const ROIAlignTestCase& roialign_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(float_type, roialign_setup.data_shape);
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(float_type, ngraph::Shape{roialign_setup.num_rois, 4});
        const auto rois = std::make_shared<ngraph::opset3::Parameter>(integer_type, ngraph::Shape{roialign_setup.num_rois});

        const auto roi_dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{rois->get_shape().size()});
        const auto roi_dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(rois, roi_dims);

        const auto data_dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{data->get_shape().size()});
        const auto data_dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, data_dims);

        const auto node = std::make_shared<ngraph::opset3::ROIAlign>(data_dsr, boxes, roi_dsr,
                roialign_setup.pooled_h, roialign_setup.pooled_w, roialign_setup.sampling_ratio, roialign_setup.spatial_scale, roialign_setup.mode);

        const auto c_index = ngraph::opset3::Constant::create(data_dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{1});
        const auto c_axis = ngraph::opset3::Constant::create(data_dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{0});
        const auto c = std::make_shared<ngraph::opset3::Gather>(data_dims, c_index, c_axis);

        const auto pooled_h = ngraph::opset3::Constant::create(data_dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{roialign_setup.pooled_h});
        const auto pooled_w = ngraph::opset3::Constant::create(data_dims->get_element_type(), ngraph::Shape{1}, std::vector<uint64_t>{roialign_setup.pooled_w});

        const auto output_shape = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{roi_dims, c, pooled_h, pooled_w}, 0);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node, output_shape);
        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{dsr1},
                ngraph::ParameterVector{data, boxes, rois, roi_dims, data_dims},
                "Expected");
    }
};

TEST_P(DynamicToStaticShapeROIAlign, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeROIAlign, testing::Combine(
        testing::Values(
                ngraph::element::f16,
                ngraph::element::f32),
        testing::Values(
                ngraph::element::i32,
                ngraph::element::i64,
                ngraph::element::u8),
        //data_shape, num_rois, pooled_h, pooled_w, sampling_ratio, spatial_scale, mode
        testing::Values(
                ROIAlignTestCase{{7, 256, 200, 200}, 1000, 6, 6, 2, 16., "avg"},
                ROIAlignTestCase{{7, 256, 200, 200}, 1000, 7, 6, 2, 16., "max"})));

}  // namespace

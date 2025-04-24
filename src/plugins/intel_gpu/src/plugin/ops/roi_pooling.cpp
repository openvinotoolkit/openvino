// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/roi_pooling.hpp"
#include "openvino/op/psroi_pooling.hpp"
#include "openvino/op/deformable_psroi_pooling.hpp"

#include "intel_gpu/primitives/roi_pooling.hpp"

namespace ov::intel_gpu {

static cldnn::pooling_mode GetPoolingMode(std::string method) {
    if (method == "bilinear")
        return cldnn::pooling_mode::bilinear;
    else if (method == "max")
        return cldnn::pooling_mode::max;
    else if (method == "average")
        return cldnn::pooling_mode::average;
    else
        return cldnn::pooling_mode::deformable_bilinear;
}

static void CreateDeformablePSROIPoolingOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::DeformablePSROIPooling>& op) {
    validate_inputs_count(op, {2, 3});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    cldnn::pooling_mode mode = GetPoolingMode(op->get_mode());
    float trans_std = op->get_trans_std();
    int part_size = op->get_part_size();
    bool no_trans = op->get_input_size() == 2 ? true : false;

    // temporary workaround due to incorrect usage of group_size in the nGraph operation for the DeformablePSROIPooling
    int pooled_width = op->get_group_size();
    int pooled_height = op->get_group_size();
    int group_size = op->get_group_size();
    int output_dim = op->get_output_dim();
    float spatial_scale = op->get_spatial_scale();
    int spatial_bins_x = op->get_spatial_bins_x();
    int spatial_bins_y = op->get_spatial_bins_y();
    bool position_sensitive = true;

    auto psROIPoolingPrim = cldnn::roi_pooling(layerName,
                                               inputs,
                                               mode,
                                               position_sensitive,
                                               pooled_width,
                                               pooled_height,
                                               spatial_scale,
                                               trans_std,
                                               no_trans,
                                               part_size,
                                               group_size,
                                               output_dim,
                                               spatial_bins_x,
                                               spatial_bins_y);
    p.add_primitive(*op, psROIPoolingPrim);
}

static void CreatePSROIPoolingOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::PSROIPooling>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    cldnn::pooling_mode mode = GetPoolingMode(op->get_mode());
    int group_size = static_cast<int>(op->get_group_size());
    int output_dim = static_cast<int>(op->get_output_dim());
    float spatial_scale = op->get_spatial_scale();
    int spatial_bins_x = op->get_spatial_bins_x();
    int spatial_bins_y = op->get_spatial_bins_y();
    bool position_sensitive = true;

    auto psROIPoolingPrim = cldnn::roi_pooling(layerName,
                                               inputs[0],  // input data
                                               inputs[1],  // input rois
                                               mode,
                                               position_sensitive,
                                               group_size,
                                               group_size,
                                               spatial_scale,
                                               output_dim,
                                               spatial_bins_x,
                                               spatial_bins_y);
    p.add_primitive(*op, psROIPoolingPrim);
}

static void CreateROIPoolingOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::ROIPooling>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    // params
    auto out_size = op->get_output_roi();
    int pooled_height = static_cast<int>(out_size[0]);
    int pooled_width = static_cast<int>(out_size[1]);
    float spatial_scale = op->get_spatial_scale();
    bool position_sensitive = false;

    cldnn::pooling_mode mode = GetPoolingMode(op->get_method());
    auto roiPoolingPrim = cldnn::roi_pooling(layerName,
                                             inputs[0],  // input data
                                             inputs[1],  // input rois
                                             mode,
                                             position_sensitive,
                                             pooled_width,
                                             pooled_height,
                                             spatial_scale,
                                             0,
                                             1,
                                             1);

    p.add_primitive(*op, roiPoolingPrim);
}

REGISTER_FACTORY_IMPL(v1, DeformablePSROIPooling);
REGISTER_FACTORY_IMPL(v0, PSROIPooling);
REGISTER_FACTORY_IMPL(v0, ROIPooling);

}  // namespace ov::intel_gpu

// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/roi_align_rotated.hpp"

#include <memory>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/roi_align.hpp"

namespace ov {
namespace intel_gpu {

namespace {

void CreateROIAlignRotatedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v15::ROIAlignRotated>& op) {
    validate_inputs_count(op, {3});
    auto roi_align_prim = cldnn::roi_align(layer_type_name_ID(op),
                                           p.GetInputInfo(op),
                                           op->get_pooled_h(),
                                           op->get_pooled_w(),
                                           op->get_sampling_ratio(),
                                           op->get_spatial_scale(),
                                           cldnn::roi_align::PoolingMode::avg,
                                           cldnn::roi_align::AlignedMode::asymmetric,
                                           cldnn::roi_align::ROIMode::rotated,
                                           op->get_clockwise_mode());
    p.add_primitive(*op, roi_align_prim);
}

}  // anonymous namespace

REGISTER_FACTORY_IMPL(v15, ROIAlignRotated);

}  // namespace intel_gpu
}  // namespace ov

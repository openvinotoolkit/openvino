// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "ngraph/op/roi_align.hpp"
#include "intel_gpu/primitives/roi_align.hpp"
#include <memory>

namespace ov {
namespace runtime {
namespace intel_gpu {

namespace {

cldnn::roi_align::PoolingMode from(ngraph::op::v3::ROIAlign::PoolingMode mode) {
    switch (mode) {
    case ngraph::op::v3::ROIAlign::PoolingMode::MAX:
        return cldnn::roi_align::PoolingMode::Max;
    case ngraph::op::v3::ROIAlign::PoolingMode::AVG:
    default:
        return cldnn::roi_align::PoolingMode::Avg;
    }
}

void CreateROIAlignOp(Program& p, const std::shared_ptr<ngraph::op::v3::ROIAlign>& op) {
    p.ValidateInputs(op, { 3 });
    auto roi_align_prim = cldnn::roi_align(layer_type_name_ID(op),
                                           p.GetInputPrimitiveIDs(op),
                                           op->get_pooled_h(),
                                           op->get_pooled_w(),
                                           op->get_sampling_ratio(),
                                           op->get_spatial_scale(),
                                           from(op->get_mode()),
                                           op->get_friendly_name());
    p.AddPrimitive(roi_align_prim);
    p.AddPrimitiveToProfiler(op);
}

} // anonymous namespace

REGISTER_FACTORY_IMPL(v3, ROIAlign);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bevpool_v2.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/bevpool_v2.hpp"

namespace ov::intel_gpu {

static void CreateBevPoolV2Op(ProgramBuilder& p, const std::shared_ptr<ov::op::v15::BevPoolV2>& op) {
    auto inputs = p.GetInputInfo(op);

    const auto to_cldnn_bound = [](const ov::op::v15::Bound& b) {
        cldnn::bound3d out;
        out.min = b.min;
        out.max = b.max;
        out.step = b.step;
        return out;
    };

    auto prim = cldnn::bevpool_v2(layer_type_name_ID(op),
                                  inputs,
                                  op->get_input_channels(),
                                  op->get_output_channels(),
                                  op->get_image_width(),
                                  op->get_image_height(),
                                  op->get_feature_width(),
                                  op->get_feature_height(),
                                  to_cldnn_bound(op->get_x_bound()),
                                  to_cldnn_bound(op->get_y_bound()),
                                  to_cldnn_bound(op->get_z_bound()),
                                  to_cldnn_bound(op->get_d_bound()));
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v15, BevPoolV2);

}  // namespace ov::intel_gpu

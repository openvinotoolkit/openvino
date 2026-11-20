// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/histc.hpp"

namespace ov::intel_gpu {

static void CreateHistcOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v17::Histc>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    auto prim = cldnn::histc(layer_type_name_ID(op), inputs[0], op->get_bins(), op->get_min(), op->get_max());
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v17, Histc);

}  // namespace ov::intel_gpu

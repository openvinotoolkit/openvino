// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/histc.hpp"

namespace ov::intel_gpu {

namespace {

void CreateHistcOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v17::Histc>& op) {
    validate_inputs_count(op, {1});

    const cldnn::histc histc_prim(layer_type_name_ID(op),
                                  p.GetInputInfo(op),
                                  cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                                  op->get_bins(),
                                  op->get_min_val(),
                                  op->get_max_val());
    p.add_primitive(*op, histc_prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(v17, Histc);

}  // namespace ov::intel_gpu

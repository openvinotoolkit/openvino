// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/fully_connected_gguf.hpp"

#include "intel_gpu/op/fully_connected_gguf.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

namespace ov::op::internal {
using FullyConnectedGGUF = ov::intel_gpu::op::FullyConnectedGGUF;
}  // namespace ov::op::internal

namespace ov::intel_gpu {

static void CreateFullyConnectedGGUFOp(ProgramBuilder& p, const std::shared_ptr<op::FullyConnectedGGUF>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    const cldnn::fully_connected_gguf prim(layer_type_name_ID(op),
                                           inputs[0],
                                           inputs[1],
                                           op->get_format(),
                                           cldnn::element_type_to_data_type(op->get_output_element_type(0)));
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, FullyConnectedGGUF);

}  // namespace ov::intel_gpu

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/stateless_kv.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/stateless_kv.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {
namespace internal {
using StatelessKV = ov::intel_gpu::op::StatelessKV;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

namespace {

void CreateStatelessKVOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::StatelessKV>& op) {
    validate_inputs_count(op, {3, 4});
    auto inputs = p.GetInputInfo(op);
    int64_t rank = op->get_input_partial_shape(0).size();
    auto prim = cldnn::stateless_kv(layer_type_name_ID(op), inputs, ov::util::normalize(op->get_concat_axis(), rank), op->get_is_present_len());

    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, prim);
}

} // namespace

REGISTER_FACTORY_IMPL(internal, StatelessKV);

}  // namespace ov::intel_gpu

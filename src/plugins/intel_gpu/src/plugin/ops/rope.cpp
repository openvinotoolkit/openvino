// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/rotary_positional_embeddings.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/rope.hpp"
#include "intel_gpu/primitives/permute.hpp"

namespace ov {
namespace op {
namespace internal {
using RoPE = ov::op::internal::RoPE;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateRoPEOp(ProgramBuilder& p, const std::shared_ptr<op::internal::RoPE>& op) {
    validate_inputs_count(op, {3, 4});
    auto inputs = p.GetInputInfo(op);
    const auto& config = op->get_config();

    size_t gather_rank = 0;
    if (config.gather_position_arg_id > 0) {
        gather_rank = op->get_input_partial_shape(config.gather_position_arg_id).size();
    }

    auto rope = cldnn::rope(layer_type_name_ID(op),
                            inputs,
                            config,
                            gather_rank);

    p.add_primitive(*op, rope);
}

REGISTER_FACTORY_IMPL(internal, RoPE);

}  // namespace intel_gpu
}  // namespace ov

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

    if (config.input_trans0213) {
        size_t input_rank = op->get_input_partial_shape(0).size();
        std::vector<uint16_t> transpose_order(input_rank);
        std::iota(transpose_order.begin(), transpose_order.end(), 0);
        std::swap(*(transpose_order.begin() + 1), *(transpose_order.begin() + 2));

        auto permute_name = op->get_friendly_name() + "_trans0213";
        auto permutePrim = cldnn::permute(permute_name,
                                          cldnn::input_info(inputs[0].pid),
                                          transpose_order);
        p.add_primitive(*op, permutePrim);
        inputs[0] = cldnn::input_info(permute_name);
    }

    auto rope = cldnn::rope(layer_type_name_ID(op),
                            inputs,
                            config);

    p.add_primitive(*op, rope);
}

REGISTER_FACTORY_IMPL(internal, RoPE);

}  // namespace intel_gpu
}  // namespace ov

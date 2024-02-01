// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/rope.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/rope.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/slice.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"

namespace ov {
namespace op {
namespace internal {
using RoPE = ov::intel_gpu::op::RoPE;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateRoPEOp(ProgramBuilder& p, const std::shared_ptr<op::RoPE>& op) {
    validate_inputs_count(op, {3, 4});
    auto inputs = p.GetInputInfo(op);
    const auto& config = op->get_config();

    if (config.input_trans0213) {
        auto& input_pshape = op->get_input_partial_shape(0);
        std::vector<uint16_t> transposeOrder(input_pshape.size());
        std::iota(transposeOrder.begin(), transposeOrder.end(), 0);
        std::swap(*(transposeOrder.begin() + 1), *(transposeOrder.begin() + 2));

        auto permuteName = op->get_friendly_name() + "_trans0213";
        auto permutePrim = cldnn::permute(permuteName,
                                          cldnn::input_info(inputs[0].pid),
                                          transposeOrder);
        p.add_primitive(*op, permutePrim);
        inputs[0] = cldnn::input_info(permuteName);
    }

    // if (config.is_interleaved) {
        // add transpose afer RoPE
    // }

    auto rope = cldnn::rope(layer_type_name_ID(op),
                            inputs,
                            config);

    p.add_primitive(*op, rope);
}

REGISTER_FACTORY_IMPL(internal, RoPE);

}  // namespace intel_gpu
}  // namespace ov

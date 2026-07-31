// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/dynamic_quantize.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/dynamic_quantize.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"


namespace ov::intel_gpu {

// Quantization groups are formed along the innermost input dimension, so its length drives the kernel's
// group count. When the input keeps that dimension dynamic, it can still be recovered from the weights of
// the consuming FullyConnected, whose IFM equals it. Returns 0 when it stays unknown, which makes the opt
// kernel reject the primitive so that the ref kernel is used instead.
static size_t get_innermost_size(const std::shared_ptr<ov::op::internal::DynamicQuantize>& op) {
    const auto& in_shape = op->get_input_partial_shape(0);
    const auto& innermost_dim = in_shape[in_shape.size() - 1];
    if (innermost_dim.is_static())
        return innermost_dim.get_length();

    size_t innermost_size = 0;
    for (const auto& target_input : op->get_output_target_inputs(0)) {
        const auto* fc = ov::as_type<const op::FullyConnectedCompressed>(target_input.get_node());
        if (fc == nullptr || target_input.get_index() != 0)
            continue;

        // Weights are [N, K] when transposed, [K, N] otherwise
        const auto& weights_shape = fc->get_input_partial_shape(1);
        const auto& ifm = weights_shape[weights_shape.size() - (fc->get_transpose_b() ? 1 : 2)];
        if (!ifm.is_static())
            continue;

        // Every user consumes the same activations, so they have to agree on the length
        const auto ifm_size = static_cast<size_t>(ifm.get_length());
        OPENVINO_ASSERT(innermost_size == 0 || innermost_size == ifm_size,
                        "[GPU] Users of ", op->get_friendly_name(), " disagree on the innermost size: ",
                        innermost_size, " vs ", ifm_size);
        innermost_size = ifm_size;
    }

    return innermost_size;
}

static void CreateDynamicQuantizeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::DynamicQuantize>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto prim = cldnn::dynamic_quantize(primitive_name,
                                        inputs[0],
                                        op->get_attrs(),
                                        op->get_input_partial_shape(0).size(),
                                        get_innermost_size(op));

    prim.num_outputs = op->get_output_size();

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, DynamicQuantize);

}  // namespace ov::intel_gpu

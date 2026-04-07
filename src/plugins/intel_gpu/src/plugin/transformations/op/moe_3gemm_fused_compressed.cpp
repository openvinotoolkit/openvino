// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"

namespace ov::intel_gpu::op {

MOE3GemmFusedCompressed::MOE3GemmFusedCompressed(const OutputVector& args, const MOECompressed::Config config) : MOECompressed(args, config) {
    constructor_validate_and_infer_types();
}

void MOE3GemmFusedCompressed::validate_and_infer_types() {
    const size_t expected_inputs = m_config.num_shared_expert > 0
                                     ? 24
                                 : m_config.routing_type == MOECompressed::RoutingType::SIGMOID_BIAS
                                     ? (m_config.has_routing_norm_scale ? 14 : 13)
                                 : 11;
    OPENVINO_ASSERT(get_input_size() == expected_inputs,
                    "MOECompressed: expected ",
                    expected_inputs,
                    " inputs for routing type ",
                    m_config.routing_type,
                    ", got ",
                    get_input_size());

    if (m_config.routing_type == MOECompressed::RoutingType::SIGMOID_BIAS) {
        // Input 12 is routing_eps — must be a scalar
        OPENVINO_ASSERT(ov::shape_size(get_input_partial_shape(12).to_shape()) == 1,
                        "MOE3GemmFusedCompressed: routing_eps (input 12) must be scalar, got shape ",
                        get_input_partial_shape(12));
        if (m_config.has_routing_norm_scale) {
            // Input 13 is routing_norm_scale — must be a scalar
            OPENVINO_ASSERT(ov::shape_size(get_input_partial_shape(13).to_shape()) == 1,
                            "MOE3GemmFusedCompressed: routing_norm_scale (input 13) must be scalar, got shape ",
                            get_input_partial_shape(13));
        }
    }

    MOECompressed::validate_and_infer_types();
}

std::shared_ptr<ov::Node> MOE3GemmFusedCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<MOE3GemmFusedCompressed>(new_args, get_config());
}

}  // namespace ov::intel_gpu::op

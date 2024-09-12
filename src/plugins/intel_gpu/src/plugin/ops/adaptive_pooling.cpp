// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/op/adaptive_max_pool.hpp"
#include "openvino/op/adaptive_avg_pool.hpp"

#include "intel_gpu/plugin/program_builder.hpp"

#include "intel_gpu/primitives/adaptive_pooling.hpp"

namespace ov {
namespace intel_gpu {

static void CreateAdaptiveAvgPoolOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::AdaptiveAvgPool>& op) {
    validate_inputs_count(op, {2});

    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);

    const cldnn::adaptive_pooling poolPrim{layer_name, inputs[0], inputs[1]};
    p.add_primitive(*op, poolPrim);
}

static void CreateAdaptiveMaxPoolOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::AdaptiveMaxPool>& op) {
    validate_inputs_count(op, {2});
    OPENVINO_ASSERT(op->get_output_size() == 2, "[GPU] AdaptiveMaxPool requires 2 outputs");

    auto inputs = p.GetInputInfo(op);
    const auto layer_type_name = layer_type_name_ID(op);

    size_t num_outputs = op->get_output_size();

    cldnn::adaptive_pooling poolPrim{layer_type_name,
                                        inputs[0],
                                        inputs[1],
                                        cldnn::element_type_to_data_type(op->get_index_element_type()),
                                        cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                                        num_outputs};
    poolPrim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, poolPrim);
}

REGISTER_FACTORY_IMPL(v8, AdaptiveAvgPool);
REGISTER_FACTORY_IMPL(v8, AdaptiveMaxPool);

}  // namespace intel_gpu
}  // namespace ov

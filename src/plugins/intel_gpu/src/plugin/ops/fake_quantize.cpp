// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/fake_quantize.hpp"

#include "intel_gpu/primitives/quantize.hpp"

namespace ov::intel_gpu {

static void CreateFakeQuantizeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::FakeQuantize>& op) {
    validate_inputs_count(op, {5});
    std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);

    auto input_id       = inputs[0];
    auto input_low_id   = inputs[1];
    auto input_high_id  = inputs[2];
    auto output_low_id  = inputs[3];
    auto output_high_id = inputs[4];

    int levels = static_cast<int>(op->get_levels());
    auto dt = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    auto quantizationPrim = cldnn::quantize(layerName,
                                            input_id,
                                            input_low_id,
                                            input_high_id,
                                            output_low_id,
                                            output_high_id,
                                            levels,
                                            dt);

    p.add_primitive(*op, quantizationPrim);
}

REGISTER_FACTORY_IMPL(v0, FakeQuantize);

}  // namespace ov::intel_gpu

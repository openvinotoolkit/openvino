// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/result.hpp"
#include "openvino/op/nv12_to_rgb.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/i420_to_rgb.hpp"
#include "openvino/op/i420_to_bgr.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov {
namespace intel_gpu {

static void CreateResultOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Result>& op) {
    validate_inputs_count(op, {1});

    auto prev = op->get_input_node_shared_ptr(0);
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto input_id = ov::descriptor::get_ov_tensor_legacy_name(op->get_input_source_output(0).get_tensor());
    OPENVINO_SUPPRESS_DEPRECATED_END
    if (input_id.empty()) {
        input_id = prev->get_friendly_name();
        if (prev->get_output_size() > 1) {
            input_id += "." + std::to_string(op->get_input_source_output(0).get_index());
        }
    }
    auto inputs = p.GetInputInfo(op);

    auto out_rank = op->get_output_partial_shape(0).size();
    auto out_format = cldnn::format::get_default_format(out_rank);

    auto out_primitive_name = layer_type_name_ID(op);
    auto out_data_type = cldnn::element_type_to_data_type(convert_to_supported_device_type(op->get_input_element_type(0)));

    auto reorder_primitive = cldnn::reorder(out_primitive_name,
                                            inputs[0],
                                            out_format,
                                            out_data_type);
    p.add_primitive(*op, reorder_primitive, {input_id, op->get_friendly_name()});
    p.prevPrimitiveIDs[out_primitive_name] = {input_id};
}

REGISTER_FACTORY_IMPL(v0, Result);

}  // namespace intel_gpu
}  // namespace ov

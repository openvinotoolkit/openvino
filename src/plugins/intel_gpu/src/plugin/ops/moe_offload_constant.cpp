// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_offload_constant.hpp"

namespace ov::intel_gpu::moe_offload {

bool is_moe_related_constant(const std::shared_ptr<ov::op::v0::Constant>& op) {
    const auto users = op->get_output_target_inputs(0);
    for (const auto& input : users) {
        const auto* node = input.get_node();
        if (ov::is_type<ov::op::internal::MOECompressed>(node)) {
            return true;
        }
    }
    return false;
}

PartialUploadLogState& get_partial_upload_log_state() {
    static PartialUploadLogState state;
    return state;
}

PartialUploadDesc try_prepare_partial_upload(ProgramBuilder& p,
                                             const std::shared_ptr<ov::op::v0::Constant>& op,
                                             const ov::Shape& const_shape,
                                             cldnn::data_types out_dtype,
                                             const cldnn::format& const_format,
                                             const cldnn::layout& const_layout) {
    PartialUploadDesc desc;

    const size_t otd_ratio = p.get_config().get_moe_offload_ratio();
    const bool partial_moe_const_upload = otd_ratio > 0 && is_moe_related_constant(op);
    if (!partial_moe_const_upload || const_layout.bytes_count() == 0 || const_shape.empty() || const_shape[0] == 0) {
        return desc;
    }

    const size_t otd_expert_num = std::max<size_t>(1, const_shape[0] * otd_ratio / 100);

    desc.enabled = true;
    desc.upload_shape = const_shape;
    desc.upload_shape[0] = std::min<size_t>(const_shape[0], otd_expert_num);

    auto upload_layout = cldnn::layout(desc.upload_shape, out_dtype, const_format);
    auto upload_mem = p.get_engine().allocate_memory(upload_layout, false);
    desc.memory = p.get_engine().reinterpret_buffer(*upload_mem, const_layout);
    desc.upload_bytes = upload_layout.bytes_count();

    get_partial_upload_log_state().log(op->get_friendly_name(),
                                       desc.upload_shape[0],
                                       const_shape[0],
                                       desc.upload_bytes,
                                       const_layout.bytes_count());
    return desc;
}

}  // namespace ov::intel_gpu::moe_offload

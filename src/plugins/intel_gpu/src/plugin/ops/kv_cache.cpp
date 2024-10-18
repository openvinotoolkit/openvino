// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/kv_cache.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {
namespace internal {
using KVCache = ov::intel_gpu::op::KVCache;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

namespace {

void CreateKVCacheOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::KVCache>& op) {
    validate_inputs_count(op, {2, 3, 4, 5});
    auto inputs = p.GetInputInfo(op);
    int64_t rank = op->get_input_partial_shape(0).size();
    auto prim = cldnn::kv_cache(layer_type_name_ID(op),
                                inputs,
                                op->get_variable()->get_info(),
                                ov::util::normalize(op->get_concat_axis(), rank),
                                ov::util::normalize(op->get_gather_axis(), rank),
                                op->get_indirect());

    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);

    if (op->get_kv_compressed()) {
        prim.compressed = true;
        prim.combine_scales_and_zp = op->get_combine_scales_and_zp();
        prim.quantization_config = op->get_quantization_config();
        prim.scales_zp_output_order = op->get_scales_zp_output_order();
    }

    p.add_primitive(*op, prim);
}

} // namespace

REGISTER_FACTORY_IMPL(internal, KVCache);

}  // namespace intel_gpu
}  // namespace ov

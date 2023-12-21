// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/kv_cache.hpp"

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
    validate_inputs_count(op, {2, 3});
    auto inputs = p.GetInputInfo(op);
    const auto prim = cldnn::kv_cache(layer_type_name_ID(op),
                                      inputs,
                                      op->get_variable()->get_info(),
                                      op->get_concat_axis(),
                                      op->get_gather_axis());

    p.add_primitive(*op, prim);
}

} // namespace

REGISTER_FACTORY_IMPL(internal, KVCache);

}  // namespace intel_gpu
}  // namespace ov

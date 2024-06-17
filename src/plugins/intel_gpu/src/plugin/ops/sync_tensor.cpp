// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sync_tensor.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/sync_tensor.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {
namespace internal {
using SyncTensor = ov::intel_gpu::op::SyncTensor;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

namespace {

void CreateSyncTensorOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::SyncTensor>& op) {
    validate_inputs_count(op, {0, 1});
    auto inputs = p.GetInputInfo(op);
    auto prim = cldnn::sync_tensor(layer_type_name_ID(op), inputs[0]);

    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);
    prim.output_paddings = get_output_paddings(op);

    p.add_primitive(*op, prim);
}

} // namespace

REGISTER_FACTORY_IMPL(internal, SyncTensor);

}  // namespace intel_gpu
}  // namespace ov

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/rms.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/rms.hpp"

namespace ov {
namespace op {
namespace internal {
using RMS = ov::intel_gpu::op::RMS;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateRMSOp(ProgramBuilder& p, const std::shared_ptr<op::RMS>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto rms = cldnn::rms(primitive_name,
                          inputs[0],
                          inputs[1],
                          op->get_epsilon());
    p.add_primitive(*op, rms);
}

REGISTER_FACTORY_IMPL(internal, RMS);

}  // namespace intel_gpu
}  // namespace ov

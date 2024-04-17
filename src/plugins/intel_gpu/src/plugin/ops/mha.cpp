// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "intel_gpu/op/mha.hpp"

#include "intel_gpu/primitives/mha.hpp"

#include <memory>

namespace ov {
namespace op {
namespace internal {
using MhaFusion = ov::intel_gpu::op::MhaFusion;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateMhaFusionOp(ProgramBuilder& p, const std::shared_ptr<op::MhaFusion>& op) {
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);
    auto mha = cldnn::mha(layer_type_name_ID(op),
                                   inputs[0],
                                   inputs[1],
                                   inputs[2]);
    p.add_primitive(*op, mha);
}

REGISTER_FACTORY_IMPL(internal, MhaFusion);

}  // namespace intel_gpu
}  // namespace ov

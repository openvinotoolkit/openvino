// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/reverse.hpp"

namespace ov::intel_gpu {

static void CreateReverseOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Reverse>& op) {
    validate_inputs_count(op, {2});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto mode =
        op->get_mode() == ov::op::v1::Reverse::Mode::INDEX ? cldnn::reverse_mode::index : cldnn::reverse_mode::mask;

    const cldnn::reverse reverse{layer_name, inputs[0], inputs[1], mode};

    p.add_primitive(*op, reverse);
}

REGISTER_FACTORY_IMPL(v1, Reverse);

}  // namespace ov::intel_gpu

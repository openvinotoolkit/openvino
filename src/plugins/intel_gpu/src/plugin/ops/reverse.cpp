// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/reverse.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program.hpp"
#include "ngraph/op/reverse_sequence.hpp"

namespace ov {
namespace intel_gpu {

static void CreateReverseOp(Program& p, const std::shared_ptr<ngraph::op::v1::Reverse>& op) {
    validate_inputs_count(op, {2});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto mode =
        op->get_mode() == ngraph::op::v1::Reverse::Mode::INDEX ? cldnn::reverse_mode::index : cldnn::reverse_mode::mask;

    const cldnn::reverse reverse{layer_name, inputs[0], inputs[1], mode};

    p.add_primitive(*op, reverse);
}

REGISTER_FACTORY_IMPL(v1, Reverse);

}  // namespace intel_gpu
}  // namespace ov

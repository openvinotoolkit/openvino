// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/reverse.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program.hpp"
#include "ngraph/op/reverse_sequence.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateReverseOp(Program& p, const std::shared_ptr<ngraph::op::v1::Reverse>& op) {
    const auto input_primitives = p.GetInputPrimitiveIDs(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto op_friendly_name = op->get_friendly_name();
    const auto mode =
        op->get_mode() == ngraph::op::v1::Reverse::Mode::INDEX ? cldnn::reverse_mode::index : cldnn::reverse_mode::mask;

    const cldnn::reverse reverse{layer_name, input_primitives[0], input_primitives[1], mode, op_friendly_name};

    p.AddPrimitive(reverse);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, Reverse);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov

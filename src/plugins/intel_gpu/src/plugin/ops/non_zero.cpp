// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/non_zero.hpp"

#include "intel_gpu/primitives/non_zero.hpp"

namespace ov {
namespace intel_gpu {

static void CreateNonZeroOp(Program& p, const std::shared_ptr<ngraph::Node>& op) {
    p.ValidateInputs(op, {1});
    auto input_primitives = p.GetInputPrimitiveIDs(op);
    std::string layer_name = layer_type_name_ID(op);

    cldnn::primitive_id count_prim_id = layer_name + "_count";
    auto count_prim = cldnn::count_nonzero(count_prim_id,
                                           input_primitives[0],
                                           op->get_friendly_name());

    auto gather_prim = cldnn::gather_nonzero(layer_name,
                                             input_primitives[0],
                                             count_prim_id,
                                             op->get_friendly_name());

    p.AddPrimitive(count_prim);
    p.AddPrimitive(gather_prim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v3, NonZero);

}  // namespace intel_gpu
}  // namespace ov

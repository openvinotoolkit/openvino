// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/multiclass_nms.hpp>
#include "ov_ops/multiclass_nms_ie_internal.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/multiclass_nms.hpp"

namespace ov {
namespace intel_gpu {


static void CreateMulticlassNmsIEInternalOp(ProgramBuilder& p, const std::shared_ptr<op::internal::MulticlassNmsIEInternal>& op) {
    validate_inputs_count(op, {2, 3});

    auto inputs = p.GetInputInfo(op);

    cldnn::multiclass_nms prim{layer_type_name_ID(op), inputs, op->get_attrs()};
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, MulticlassNmsIEInternal);

}  // namespace intel_gpu
}  // namespace ov

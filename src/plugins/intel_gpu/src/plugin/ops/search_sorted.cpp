// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/search_sorted.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/search_sorted.hpp"

namespace ov::intel_gpu {

static void CreateSearchSortedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v15::SearchSorted>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    auto prim = cldnn::search_sorted(layer_type_name_ID(op), inputs[0], inputs[1], op->get_right_mode());
    prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v15, SearchSorted);

}  // namespace ov::intel_gpu

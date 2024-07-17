// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/unique.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/unique.hpp"

namespace ov {
namespace intel_gpu {

namespace {

void CreateUniqueOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v10::Unique>& op) {
    validate_inputs_count(op, {1, 2});

    bool flattened = true;
    int64_t axis{};
    if (op->get_input_size() == 2) {
        auto axis_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(axis_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
        axis = axis_constant->cast_vector<int64_t>().at(0);
        axis = ov::util::try_normalize_axis(axis, op->get_input_partial_shape(0).rank(), *op);
        flattened = false;
    }

    const auto input = p.GetInputInfo(op).front();
    const auto layer_name = layer_type_name_ID(op);
    const auto count_prim_id = layer_name + "_count";

    const cldnn::unique_count unique_count_prim(count_prim_id, input, flattened, axis);
    p.add_primitive(*op, unique_count_prim);

    const cldnn::unique_gather unique_gather_prim(layer_name,
                                                  {input, count_prim_id},
                                                  flattened,
                                                  axis,
                                                  op->get_sorted(),
                                                  cldnn::element_type_to_data_type(op->get_input_element_type(0)),
                                                  cldnn::element_type_to_data_type(op->get_index_element_type()),
                                                  cldnn::element_type_to_data_type(op->get_count_element_type()));
    p.add_primitive(*op, unique_gather_prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(v10, Unique);

}  // namespace intel_gpu
}  // namespace ov

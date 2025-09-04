// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_fill_empty_rows.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/sparse_fill_empty_rows.hpp"
#include "openvino/op/constant.hpp"

namespace ov::intel_gpu {

static void CreateSparseFillEmptyRowsOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v16::SparseFillEmptyRows>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto indices_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(0));
    auto values_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto dense_shape_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    auto default_value_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(3));

    std::vector<int64_t> indices = indices_constant ? indices_constant->cast_vector<int64_t>() : std::vector<int64_t>{};
    std::vector<float> values = values_constant ? values_constant->cast_vector<float>() : std::vector<float>{};
    std::vector<int64_t> dense_shape = dense_shape_constant ? dense_shape_constant->cast_vector<int64_t>() : std::vector<int64_t>{};
    float default_value = default_value_constant ? default_value_constant->cast_vector<float>()[0] : 0.0f;

    std::shared_ptr<cldnn::sparse_fill_empty_rows> prim = nullptr;

    if (indices_constant && values_constant && dense_shape_constant && default_value_constant) {
        prim = std::make_shared<cldnn::sparse_fill_empty_rows>(
            layerName,
            inputs,
            values,
            dense_shape,
            indices,
            default_value);
    } else {
        prim = std::make_shared<cldnn::sparse_fill_empty_rows>(
            layerName,
            inputs);
    }

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v16, SparseFillEmptyRows);

}  // namespace ov::intel_gpu

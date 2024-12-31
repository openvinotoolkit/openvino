// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"

#include "openvino/op/strided_slice.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/strided_slice.hpp"

#include <climits>

namespace ov {
namespace intel_gpu {

static void CreateStridedSliceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::StridedSlice>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto output_pshape = op->get_output_partial_shape(0);
    auto input_pshape = op->get_input_partial_shape(0);

    auto convert_max_val = [](int64_t casted_val, ov::element::Type type) {
        // if the original const is int32_t and the value is max,
        // it should be kept to set the shape as unbounded dynamic shape.
        if ((type == ov::element::i32 && casted_val == INT_MAX) || (type == ov::element::u32 && casted_val == UINT_MAX))
            return static_cast<int64_t>(0x7FFFFFFFFFFFFFFF);
        return casted_val;
    };

    auto begin_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->input_value(1).get_node_shared_ptr());
    std::vector<int64_t> begin;
    if (begin_constant) {
        auto const_vals = begin_constant->cast_vector<int64_t>();
        for (auto val : const_vals) {
            begin.push_back(convert_max_val(val, begin_constant->get_element_type()));
        }
    }
    auto end_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->input_value(2).get_node_shared_ptr());
    std::vector<int64_t> end;
    if (end_constant) {
       auto const_vals = end_constant->cast_vector<int64_t>();
        for (auto val : const_vals) {
            end.push_back(convert_max_val(val, end_constant->get_element_type()));
        }
    }
    auto stride_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->input_value(3).get_node_shared_ptr());
    std::vector<int64_t> strides = stride_constant ? stride_constant->cast_vector<int64_t>() : std::vector<int64_t>{};

    // In case of dynamic shapes pass dummy shape value to strided_slice primitive
    // To be removed once we enable internal shape infer for all operations
    auto output_shape = output_pshape.is_static() ? output_pshape.to_shape() : ov::Shape{};

    auto stridedSlicePrim = std::make_shared<cldnn::strided_slice>(layerName,
                                                                   inputs,
                                                                   begin,
                                                                   end,
                                                                   strides,
                                                                   op->get_begin_mask(),
                                                                   op->get_end_mask(),
                                                                   op->get_new_axis_mask(),
                                                                   op->get_shrink_axis_mask(),
                                                                   op->get_ellipsis_mask(),
                                                                   output_shape);
    p.add_primitive(*op, stridedSlicePrim);
}

REGISTER_FACTORY_IMPL(v1, StridedSlice);

}  // namespace intel_gpu
}  // namespace ov

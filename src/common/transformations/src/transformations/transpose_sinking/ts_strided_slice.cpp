// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_strided_slice.hpp"

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

namespace {

std::vector<size_t> convert_mask_to_axis_vec(const std::vector<int64_t>& mask) {
    std::vector<size_t> axes;
    for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i) {
        if (mask[i] == 1)
            axes.push_back(i);
    }
    return axes;
};

std::vector<size_t> convert_shrink_mask_to_axis_vec(const std::vector<int64_t>& shrink_mask,
                                                    const std::vector<int64_t>& new_axis_mask) {
    std::vector<size_t> axes;
    for (size_t i = 0; i < static_cast<size_t>(shrink_mask.size()); ++i) {
        if (i < new_axis_mask.size() && new_axis_mask[i] == 1) {
            continue;
        } else if (shrink_mask[i] == 1) {
            axes.push_back(i);
        }
    }
    return axes;
};

std::vector<int64_t> transpose_mask(const std::vector<int64_t>& old_mask,
                                    const std::vector<size_t>& transpose_order_values) {
    std::vector<int64_t> new_mask(old_mask.size());
    for (size_t i = 0; i < old_mask.size(); ++i) {
        new_mask[i] = old_mask[transpose_order_values[i]];
    }
    return new_mask;
};

uint64_t get_max_value_by_type(ov::element::Type type) {
    // only integral types supported
    auto size = type.size();
    uint64_t res = 1;
    if (type.is_signed()) {
        return (res << size * 8) / 2 - 1;
    } else {
        return (res << size * 8) - 1;
    }
}

void align_masks(bool is_forward,
                 const std::shared_ptr<ov::op::v1::StridedSlice>& strided_slice,
                 const std::vector<size_t>& transpose_order_values,
                 size_t expected_size,
                 size_t num_elements_in_begin_input) {
    // align masks
    auto shrink_axes_mask = strided_slice->get_shrink_axis_mask();
    auto begin_mask = strided_slice->get_begin_mask();
    auto end_mask = strided_slice->get_end_mask();
    auto new_axis_mask = strided_slice->get_new_axis_mask();
    auto ellipsis_mask = strided_slice->get_ellipsis_mask();
    auto new_axes = convert_mask_to_axis_vec(new_axis_mask);

    begin_mask.resize(expected_size, 0);
    end_mask.resize(expected_size, 0);
    new_axis_mask.resize(expected_size, 0);
    ellipsis_mask.resize(expected_size, 0);

    // shrink_axes_mask is a corner case, it depends on the number of values in the `begin` input,
    // so we cut shrink_axes_mask values and fill it with 0
    shrink_axes_mask.resize(num_elements_in_begin_input);
    for (const auto& idx : new_axes) {
        if (idx < shrink_axes_mask.size()) {
            shrink_axes_mask[idx] = 0;
        }
    }
    shrink_axes_mask.resize(expected_size, 0);

    // transpose the aligned masks according initial Transpose order and new added dims (new_axes_mask).
    if (is_forward) {
        // we don't have to transpose new_axes_mask here for Forward transformation.
        strided_slice->set_new_axis_mask(new_axis_mask);
    } else {
        strided_slice->set_new_axis_mask(transpose_mask(new_axis_mask, transpose_order_values));
    }
    strided_slice->set_begin_mask(transpose_mask(begin_mask, transpose_order_values));
    strided_slice->set_end_mask(transpose_mask(end_mask, transpose_order_values));
    strided_slice->set_shrink_axis_mask(transpose_mask(shrink_axes_mask, transpose_order_values));
    strided_slice->set_ellipsis_mask_mask(ellipsis_mask);
}

bool align_inputs(const std::shared_ptr<ov::op::v1::StridedSlice>& strided_slice,
                  const std::vector<size_t>& transpose_order_values,
                  size_t expected_size,
                  size_t& num_elements_in_begin_input) {
    // align begin, end, stride inputs to insert Gather operation for each of them
    std::vector<std::shared_ptr<ov::op::v0::Constant>> new_inputs(3);
    for (size_t input_idx = 1; input_idx <= 3; ++input_idx) {
        auto input = strided_slice->input_value(input_idx);
        auto input_pshape = input.get_partial_shape();

        if (input_pshape.is_dynamic()) {
            return false;
        }

        auto num_elements = input_pshape[0].get_length();
        // if the number of elements in begin, end, stride inputs less than sum of the data input rank and cnt
        // of new axes (expected_size), then we have to extend it with stub values to successfully
        // execute Gather operation.
        // If this number more than the expected_size, then we have to cut the unnecessary masks/inputs values.
        if (num_elements != expected_size) {
            auto input_const = ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr());
            if (!input_const) {
                return false;
            }

            auto input_const_val = input_const->cast_vector<uint64_t>();
            if (input_idx == 1) {
                // `begin` input have to be initialized with 0
                input_const_val.resize(expected_size, 0);
                num_elements_in_begin_input = num_elements;
            } else if (input_idx == 2) {
                // 'end' input have to be initialized with the corresponding `data` input dim value
                input_const_val.resize(expected_size, get_max_value_by_type(input_const->get_element_type()));
            } else {
                // `stride` input have to be initialized with 1
                input_const_val.resize(expected_size, 1);
            }
            new_inputs[input_idx-1] = ov::op::v0::Constant::create(input_const->get_element_type(), {input_const_val.size()}, input_const_val);

            copy_runtime_info(input_const, new_inputs[input_idx-1]);
        }
    }
    // connect the new begin, end, stride inputs to StridedSlice operation
    auto axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
    for (size_t i = 1; i <= new_inputs.size(); ++i) {
        if (new_inputs[i-1]) {
            strided_slice->input(i).replace_source_output(new_inputs[i-1]);
        }
        // insert Gather
        strided_slice->input(i).replace_source_output(
                ChangeValuesOrder(strided_slice->input_value(i), transpose_order_values, axis));
    }
    return true;
}

}

TSStridedSliceForward::TSStridedSliceForward() {
    MATCHER_SCOPE(TSStridedSliceForward);
    create_pattern<ov::op::v1::StridedSlice>(true, {0});

    auto sinking_transformation = [=](const std::shared_ptr<Node>& main_node,
                                      const TransposeInputsInfo& transpose_info) -> bool {
        auto strided_slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(main_node);
        if (!strided_slice) {
            return false;
        }

        auto data_partial_shape = strided_slice->input_value(0).get_partial_shape();
        auto data_rank = data_partial_shape.rank();
        if (data_rank.is_dynamic()) {
            return false;
        }

        // todo: handle ellipsis_mask
        auto ellipsis_mask = strided_slice->get_ellipsis_mask();
        auto ellipsis_axes = convert_mask_to_axis_vec(ellipsis_mask);
        if (!ellipsis_axes.empty()) {
            return false;
        }

        // values in begin, end, stride inputs and in begin_mask, end_mask, shrink_mask
        // may correspond to new dims added by new_axis_mask
        // we have to ignore these values and do not apply Transpose, Gather ops to them.
        auto new_axis_mask = strided_slice->get_new_axis_mask();
        auto new_axes = convert_mask_to_axis_vec(strided_slice->get_new_axis_mask());
        auto data_rank_val = data_rank.get_length();

        size_t expected_size = data_rank_val + new_axes.size();
        size_t num_elements_in_begin_input = expected_size;

        // delete Transpose op from the 1st input
        utils::sink_forward::UpdateInputTransposes(main_node, transpose_info, {0});

        auto transpose_order_values = transpose_info.transpose_const->cast_vector<size_t>();

        // apply new_axes mask to get the correct order
        transpose_order_values = GetOrderBeforeReduction(new_axes, transpose_order_values);

        if (!align_inputs(strided_slice, transpose_order_values, expected_size, num_elements_in_begin_input)) {
            return false;
        }
        align_masks(true, strided_slice, transpose_order_values, expected_size, num_elements_in_begin_input);

        // apply shrink_mask to get the correct order.
        // the mask have not to be transposed, so apply transpose 2nd time to get the original order.
        auto shrink_axes = convert_mask_to_axis_vec(transpose_mask(strided_slice->get_shrink_axis_mask(), transpose_order_values));

        transpose_order_values = GetOrderAfterReduction(shrink_axes, transpose_order_values);

        // add Transpose op to StridedSlice output
        auto new_transpose_order = std::make_shared<ov::op::v0::Constant>(transpose_info.transpose_const->get_element_type(),
                                                           Shape{transpose_order_values.size()},
                                                           transpose_order_values);

        TransposeInputsInfo transpose_input_info = {transpose_info.transpose, new_transpose_order, 0};
        strided_slice->validate_and_infer_types();
        default_outputs_update(main_node, transpose_input_info);
        return true;
    };

    transpose_sinking(matcher_name, sinking_transformation);
}

TSStridedSliceBackward::TSStridedSliceBackward() {
    MATCHER_SCOPE(TSStridedSliceBackward);

    auto main_node_label = wrap_type<ov::op::v1::StridedSlice>([](const Output<Node> &output) -> bool {
        return has_static_rank()(output) && CheckTransposeConsumers(output);
    });

    auto transpose_const_label = wrap_type<ov::op::v0::Constant>();

    auto transpose_label = wrap_type<ov::op::v1::Transpose>({main_node_label, transpose_const_label},
                                                            [](const Output<Node> &output) -> bool {
                                                                return has_static_rank()(output);
                                                            });

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();
        auto transpose_order = as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(transpose_const_label));
        auto transpose = pattern_to_output.at(transpose_label);
        auto main_node = pattern_to_output.at(main_node_label);
        if (transformation_callback(main_node)) {
            return false;
        }

        auto strided_slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(main_node);
        if (!strided_slice) {
            return false;
        }

        auto data_partial_shape = strided_slice->input_value(0).get_partial_shape();
        auto data_rank = data_partial_shape.rank();
        if (data_rank.is_dynamic()) {
            return false;
        }

        // todo: handle ellipsis_mask
        auto ellipsis_mask = strided_slice->get_ellipsis_mask();
        auto ellipsis_axes = convert_mask_to_axis_vec(ellipsis_mask);
        if (!ellipsis_axes.empty()) {
            return false;
        }

        // values in begin, end, stride inputs and in begin_mask, end_mask, shrink_mask
        // may correspond to new dims added by new_axis_mask
        // we have to ignore these values and do not apply Transpose, Gather ops to them.
        auto new_axis_mask = strided_slice->get_new_axis_mask();
        auto new_axes = convert_mask_to_axis_vec(strided_slice->get_new_axis_mask());
        auto data_rank_val = data_rank.get_length();

        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        size_t expected_size = data_rank_val + new_axes.size();
        size_t num_elements_in_begin_input = expected_size;

        // apply shrink_ mask to get the correct order
        auto shrink_axes = convert_shrink_mask_to_axis_vec(strided_slice->get_shrink_axis_mask(),
                                                           strided_slice->get_new_axis_mask());

        transpose_order_values = GetOrderBeforeReduction(shrink_axes, transpose_order_values);
        if (!align_inputs(strided_slice, transpose_order_values, expected_size, num_elements_in_begin_input)) {
            return false;
        }

        align_masks(false, strided_slice, transpose_order_values, expected_size, num_elements_in_begin_input);

        transpose_order_values = GetOrderAfterReduction(new_axes, transpose_order_values);
        auto new_transpose_order = std::make_shared<ov::op::v0::Constant>(transpose_order->get_element_type(),
                                                                          Shape{transpose_order_values.size()},
                                                                          transpose_order_values);
        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node, new_transpose_order, {0})) {
            register_new_node(new_node);
        }
        main_node->validate_and_infer_types();
        RemoveTransposeConsumers(main_node);
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
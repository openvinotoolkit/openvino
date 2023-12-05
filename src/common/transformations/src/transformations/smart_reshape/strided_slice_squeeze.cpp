// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/strided_slice_squeeze.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations_visibility.hpp"

ov::pass::StridedSliceSqueeze::StridedSliceSqueeze() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(StridedSliceSqueeze);
    auto ss_label = ov::pass::pattern::wrap_type<ov::op::v1::StridedSlice>(pattern::consumers_count(1));
    auto squeeze_label = ov::pass::pattern::wrap_type<ov::op::v0::Squeeze>(
        {ss_label, ov::pass::pattern::wrap_type<ov::op::v0::Constant>()});

    matcher_pass_callback callback = [](pattern::Matcher& m) -> bool {
        const auto& squeeze = m.get_match_root();
        const auto& const_axes = std::dynamic_pointer_cast<ov::op::v0::Constant>(squeeze->get_input_node_shared_ptr(1));
        auto slice = std::dynamic_pointer_cast<ov::op::v1::StridedSlice>(squeeze->get_input_node_shared_ptr(0));
        if (!const_axes || !slice)
            return false;

        auto begin = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
        auto end = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(2).get_node_shared_ptr());
        auto strides = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(3).get_node_shared_ptr());
        if (!begin || !end || !strides)
            return false;

        auto begin_vec = begin->cast_vector<int64_t>();
        auto end_vec = end->cast_vector<int64_t>();
        auto strides_vec = strides->cast_vector<int64_t>();
        auto begin_mask = slice->get_begin_mask();
        auto end_mask = slice->get_end_mask();
        auto new_axis_mask = slice->get_new_axis_mask().empty() ? std::vector<int64_t>(begin_mask.size(), 0)
                                                                : slice->get_new_axis_mask();
        auto shrink_axis_mask = slice->get_shrink_axis_mask().empty() ? std::vector<int64_t>(begin_mask.size(), 0)
                                                                      : slice->get_shrink_axis_mask();
        auto ellipsis_mask = slice->get_ellipsis_mask().empty() ? std::vector<int64_t>(begin_mask.size(), 0)
                                                                : slice->get_ellipsis_mask();
        auto is_zero_vec = [](const std::vector<int64_t>& mask) {
            return std::all_of(mask.begin(), mask.end(), [](const int64_t& i) {
                return i == 0;
            });
        };
        if (!is_zero_vec(new_axis_mask) || !is_zero_vec(shrink_axis_mask) || !is_zero_vec(ellipsis_mask))
            return false;
        if (!std::all_of(strides_vec.begin(), strides_vec.end(), [](const int64_t& i) {
                return i == 1;
            }))
            return false;

        OPENVINO_SUPPRESS_DEPRECATED_START
        const auto& axes = normalize_axes(squeeze->description(),
                                          const_axes->cast_vector<int64_t>(),
                                          squeeze->get_input_partial_shape(0).rank());
        OPENVINO_SUPPRESS_DEPRECATED_END

        // Here squeeze input shape is equal to stridedslice input shape,
        // since new_axis_mask, shrink_axis_mask and ellipsis_mask are all zeros.
        auto tensor_rank = squeeze->get_input_partial_shape(0).rank();
        if (tensor_rank.is_dynamic())
            return false;

        auto tensor_length = tensor_rank.get_length();
        begin_vec.resize(tensor_length, 0);
        end_vec.resize(tensor_length, 0);
        strides_vec.resize(tensor_length, 1);
        begin_mask.resize(tensor_length, 1);  // ignore what is appended to begin_vec and the 'real' beginning of the
                                              // tensor is used along corresponding dimension.
        end_mask.resize(tensor_length, 1);    // igore what is appended to end_vec, and the real 'end' of the tensor is
                                              // used along corresponding dimension.
        new_axis_mask.resize(begin_mask.size(), 0);  // validate: All masks of StridedSlice must have the same size.
        shrink_axis_mask.resize(begin_mask.size(), 0);
        ellipsis_mask.resize(begin_mask.size(), 0);

        for (const auto& axis : axes) {
            if (begin_mask[axis]) {  // corresponding dimension of the begin input is ignored. starting from 0
                begin_vec[axis] = 0;
                end_vec[axis] = 1;
                begin_mask[axis] = 0;
                end_mask[axis] = 0;
            } else {                          // corresponding dimension of the begin input is used for slicing start
                if (begin_vec[axis] == -1) {  // slicing the latest slice
                    end_mask[axis] = 1;
                } else {
                    end_vec[axis] = begin_vec[axis] + 1;
                    end_mask[axis] = 0;
                }
            }
            shrink_axis_mask[axis] = 1;
        }

        auto new_slice = std::make_shared<ov::op::v1::StridedSlice>(
            slice->input_value(0),
            ov::op::v0::Constant::create(element::i64, {begin_vec.size()}, begin_vec),
            ov::op::v0::Constant::create(element::i64, {end_vec.size()}, end_vec),
            ov::op::v0::Constant::create(element::i64, {strides_vec.size()}, strides_vec),
            begin_mask,
            end_mask,
            new_axis_mask,
            shrink_axis_mask,
            ellipsis_mask);

        return replace_output_update_name(squeeze->output(0), new_slice->output(squeeze->input_value(0).get_index()));
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(squeeze_label /*, matcher_name */);
    register_matcher(m, callback);
}
ov::pass::SqueezeStridedSlice::SqueezeStridedSlice() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(SqueezeStridedSlice);
    auto squeeze_label = ov::pass::pattern::wrap_type<ov::op::v0::Squeeze>(
        {pattern::any_input(), ov::pass::pattern::wrap_type<ov::op::v0::Constant>()},
        pattern::consumers_count(1));
    auto ss_label = ov::pass::pattern::wrap_type<ov::op::v1::StridedSlice>(
        {squeeze_label, pattern::any_input(), pattern::any_input(), pattern::any_input()});

    matcher_pass_callback callback = [](pattern::Matcher& m) -> bool {
        auto slice = std::dynamic_pointer_cast<ov::op::v1::StridedSlice>(m.get_match_root());
        if (!slice)
            return false;
        auto squeeze = slice->get_input_node_shared_ptr(0);
        const auto& const_axes = std::dynamic_pointer_cast<ov::op::v0::Constant>(squeeze->get_input_node_shared_ptr(1));
        if (!const_axes)
            return false;

        auto begin = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
        auto end = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(2).get_node_shared_ptr());
        auto strides = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(3).get_node_shared_ptr());
        if (!begin || !end || !strides)
            return false;

        auto begin_vec = begin->cast_vector<int64_t>();
        auto end_vec = end->cast_vector<int64_t>();
        auto strides_vec = strides->cast_vector<int64_t>();
        auto begin_mask = slice->get_begin_mask();
        auto end_mask = slice->get_end_mask();
        auto new_axis_mask = slice->get_new_axis_mask().empty() ? std::vector<int64_t>(begin_mask.size(), 0)
                                                                : slice->get_new_axis_mask();
        auto shrink_axis_mask = slice->get_shrink_axis_mask().empty() ? std::vector<int64_t>(begin_mask.size(), 0)
                                                                      : slice->get_shrink_axis_mask();
        auto ellipsis_mask = slice->get_ellipsis_mask().empty() ? std::vector<int64_t>(begin_mask.size(), 0)
                                                                : slice->get_ellipsis_mask();

        auto is_zero_vec = [](const std::vector<int64_t>& mask) {
            return std::all_of(mask.begin(), mask.end(), [](const int64_t& i) {
                return i == 0;
            });
        };
        if (!is_zero_vec(new_axis_mask) || !is_zero_vec(shrink_axis_mask) || !is_zero_vec(ellipsis_mask))
            return false;
        if (!std::all_of(strides_vec.begin(), strides_vec.end(), [](const int64_t& i) {
                return i == 1;
            }))
            return false;

        OPENVINO_SUPPRESS_DEPRECATED_START
        auto axes = normalize_axes(squeeze->description(),
                                   const_axes->cast_vector<int64_t>(),
                                   squeeze->get_input_partial_shape(0).rank());
        OPENVINO_SUPPRESS_DEPRECATED_END
        std::sort(axes.begin(), axes.end());
        for (const auto& axis : axes) {
            begin_vec.insert(begin_vec.begin() + axis, 0);
            end_vec.insert(end_vec.begin() + axis, 1);
            strides_vec.insert(strides_vec.begin() + axis, 1);
            begin_mask.insert(begin_mask.begin() + axis, 0);
            end_mask.insert(end_mask.begin() + axis, 0);
            new_axis_mask.insert(new_axis_mask.begin() + axis, 0);
            shrink_axis_mask.insert(shrink_axis_mask.begin() + axis, 1);
            ellipsis_mask.insert(ellipsis_mask.begin() + axis, 0);
        }

        auto new_slice = std::make_shared<ov::op::v1::StridedSlice>(
            slice->get_input_node_shared_ptr(0)->input_value(0),
            ov::op::v0::Constant::create(element::i64, {begin_vec.size()}, begin_vec),
            ov::op::v0::Constant::create(element::i64, {end_vec.size()}, end_vec),
            ov::op::v0::Constant::create(element::i64, {strides_vec.size()}, strides_vec),
            begin_mask,
            end_mask,
            new_axis_mask,
            shrink_axis_mask,
            ellipsis_mask);

        replace_node(slice, new_slice);
        new_slice->set_friendly_name(slice->get_friendly_name());
        copy_runtime_info(slice, new_slice);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(ss_label /*, matcher_name */);
    register_matcher(m, callback);
}

namespace {

bool squeezes_perform_the_same(std::shared_ptr<ov::op::v0::Squeeze> lhs, std::shared_ptr<ov::op::v0::Squeeze> rhs) {
    size_t l_input_size = lhs->inputs().size(), r_input_size = rhs->inputs().size();
    if (l_input_size != r_input_size)
        return false;
    if (lhs->inputs().size() == 1 && rhs->inputs().size() == 1)
        return true;
    const auto rank = lhs->get_input_partial_shape(0).rank();
    if (rank.is_dynamic())
        return false;
    const auto l_axes = std::dynamic_pointer_cast<ov::op::v0::Constant>(lhs->get_input_node_shared_ptr(1));
    const auto r_axes = std::dynamic_pointer_cast<ov::op::v0::Constant>(rhs->get_input_node_shared_ptr(1));
    if (l_axes && r_axes) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        return ov::normalize_axes(lhs->description(), l_axes->cast_vector<int64_t>(), rank) ==
               ov::normalize_axes(rhs->description(), r_axes->cast_vector<int64_t>(), rank);
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
    return false;
}

}  // namespace

bool ov::pass::SharedSqueeze::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(SharedSqueeze);

    bool graph_rewritten = false;

    std::map<ov::Output<Node>, std::vector<std::shared_ptr<ov::op::v0::Squeeze>>> source_to_squeeze;
    for (const auto& node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                graph_rewritten |= run_on_model(sub_graph);
            }
        }
        if (auto squeeze = std::dynamic_pointer_cast<ov::op::v0::Squeeze>(node)) {
            source_to_squeeze[squeeze->input_value(0)].push_back(squeeze);
        }
    }

    for (auto& item : source_to_squeeze) {
        if (item.second.size() < 2)
            continue;
        auto root_squeeze = item.second[0];
        for (auto& child_squeeze : item.second) {
            if (root_squeeze->get_instance_id() != child_squeeze->get_instance_id() &&
                squeezes_perform_the_same(root_squeeze, child_squeeze)) {
                graph_rewritten |= replace_output_update_name(child_squeeze->output(0), root_squeeze->output(0));
            }
        }
    }
    return graph_rewritten;
}

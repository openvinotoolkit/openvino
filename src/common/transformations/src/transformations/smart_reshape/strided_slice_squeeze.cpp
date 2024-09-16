// Copyright (C) 2018-2024 Intel Corporation
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
        const auto& const_axes = ov::as_type_ptr<ov::op::v0::Constant>(squeeze->get_input_node_shared_ptr(1));
        auto slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(squeeze->get_input_node_shared_ptr(0));
        if (!const_axes || !slice)
            return false;

        auto begin = ov::as_type_ptr<ov::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
        auto end = ov::as_type_ptr<ov::op::v0::Constant>(slice->input_value(2).get_node_shared_ptr());
        auto strides = ov::as_type_ptr<ov::op::v0::Constant>(slice->input_value(3).get_node_shared_ptr());
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

        // Here squeeze input shape is equal to stridedslice input shape,
        // since new_axis_mask, shrink_axis_mask and ellipsis_mask are all zeros.
        auto tensor_rank = squeeze->get_input_partial_shape(0).rank();
        if (tensor_rank.is_dynamic())
            return false;

        const auto axes = util::try_get_normalized_axis_vector(const_axes->get_tensor_view(), tensor_rank, *squeeze);

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
        auto slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(m.get_match_root());
        if (!slice)
            return false;
        auto squeeze = slice->get_input_node_shared_ptr(0);
        const auto& const_axes = ov::as_type_ptr<ov::op::v0::Constant>(squeeze->get_input_node_shared_ptr(1));
        if (!const_axes)
            return false;

        auto begin = ov::as_type_ptr<ov::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
        auto end = ov::as_type_ptr<ov::op::v0::Constant>(slice->input_value(2).get_node_shared_ptr());
        auto strides = ov::as_type_ptr<ov::op::v0::Constant>(slice->input_value(3).get_node_shared_ptr());
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

        auto axes = const_axes->cast_vector<int64_t>();
        ov::util::try_normalize_axes(axes, squeeze->get_input_partial_shape(0).rank(), *squeeze);

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

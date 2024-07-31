// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_stridedslices_to_variadicsplit.hpp"

#include "intel_gpu/op/fully_connected_compressed.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

ConvertStridedSlicesToVariadicSplit::ConvertStridedSlicesToVariadicSplit() {
    using namespace ov::pass::pattern;

    auto fc_predicate = [](const ov::Output<ov::Node>& output) -> bool {
        const size_t num_users_to_fuse = 3;
        const auto fc = ov::as_type_ptr<op::FullyConnectedCompressed>(output.get_node_shared_ptr());
        size_t user_count = 0;
        for (const auto& user : fc ->get_users()) {
            const auto strided_slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(user);
            if (!strided_slice)
                return false;
            user_count++;
        }
        return (user_count == num_users_to_fuse) && consumers_count(num_users_to_fuse);
    };

    auto data_m = any_input();
    auto weights_m = any_input();
    auto bias_m = any_input();
    auto fully_connected_compressed_m = wrap_type<op::FullyConnectedCompressed>({data_m, weights_m, bias_m, any_input(), any_input()}, fc_predicate);

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        auto fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(m.get_match_root());
        ov::NodeVector strided_slice_nodes;
        std::vector<int64_t> split_lengths;
        int64_t begin_offset = 0;
        int64_t end_offset = 0;
        for (const auto& user : fc->get_users()) {
            const auto strided_slice_node = std::dynamic_pointer_cast<ov::op::v1::StridedSlice>(user);
            if (strided_slice_node) {
                auto valid_ps = [](const ov::PartialShape& shape) -> bool {
                    return shape.rank().is_static() && shape[shape.rank().get_length() - 1].is_static();
                };
                const auto& input_ps = strided_slice_node->get_input_partial_shape(0);
                const auto& output_ps = strided_slice_node->get_output_partial_shape(0);
                if (!valid_ps(input_ps) || !valid_ps(output_ps) || input_ps.rank().get_length() != output_ps.rank().get_length())
                    return false;

                auto& total_length = input_ps[input_ps.rank().get_length() - 1];
                auto& split_length = output_ps[output_ps.rank().get_length() - 1];
                if (total_length.get_length() / 3 != split_length.get_length())
                    return false;

                split_lengths.push_back(split_length.get_length());

                if (!strided_slice_node->get_shrink_axis_mask().empty() ||
                    !strided_slice_node->get_new_axis_mask().empty() ||
                    !strided_slice_node->get_ellipsis_mask().empty()) {
                    return false;
                }

                if (strided_slice_node->get_input_size() == 4 &&
                    !ov::op::util::is_constant_and_all_values_equal_int(strided_slice_node->input_value(3), 1)) {
                    return false;
                }

                end_offset += split_length.get_length();
                auto check_mask = [](const std::vector<int64_t>& mask_to_check) -> bool {
                    if (mask_to_check.back() != 0)
                        return false;
                    for (size_t i = 0; i < mask_to_check.size() - 1; ++i) {
                        if (!mask_to_check[i])
                            return false;
                    }
                    return true;
                };
                auto begin_node = strided_slice_node->get_input_node_shared_ptr(1);
                if (const auto& begin_constant_node = ov::util::get_constant_from_source(begin_node)) {
                    auto values = begin_constant_node->cast_vector<int64_t>();
                    auto begin_mask = strided_slice_node->get_begin_mask();
                    if (values.size() != begin_mask.size())
                        return false;
                    if (!check_mask(begin_mask))
                        return false;
                    if (values.back() != begin_offset)
                        return false;
                } else {
                    return false;
                }

                auto end_node = strided_slice_node->get_input_node_shared_ptr(2);
                if (const auto& end_constant_node = ov::util::get_constant_from_source(end_node)) {
                    int64_t max_value = end_node->get_element_type() == ov::element::i32 ? std::numeric_limits<int32_t>::max()
                                                                                         : std::numeric_limits<int64_t>::max();
                    auto values = end_constant_node->cast_vector<int64_t>();
                    auto end_mask = strided_slice_node->get_end_mask();
                    if (values.size() != end_mask.size())
                        return false;
                    if (!check_mask(end_mask))
                        return false;
                    if (!((values.back() == end_offset) || (values.back() == max_value)))
                        return false;
                } else {
                    return false;
                }
                begin_offset += split_length.get_length();
                strided_slice_nodes.push_back(strided_slice_node);
            }
        }
        auto name = fc->get_friendly_name() + "_split";
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {fc->get_output_partial_shape(0).rank().get_length()- 1});
        auto split_lenghts_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, split_lengths);
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(fc, axis_const, split_lenghts_const);
        variadic_split->set_friendly_name(name);
        ov::copy_runtime_info(strided_slice_nodes, variadic_split);

        for (size_t i = 0; i < strided_slice_nodes.size(); ++i) {
            auto& strided_slice_node = strided_slice_nodes[i];
            for (const auto& user : strided_slice_node->get_users()) {
                for (size_t idx = 0; idx < user->inputs().size(); ++idx) {
                    if (user->get_input_node_shared_ptr(idx) == strided_slice_node) {
                        user->input(idx).replace_source_output(variadic_split->output(i));
                    }
                }
            }
            strided_slice_node->clear_control_dependencies();
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_compressed_m, "ConvertStridedSlicesToVariadicSplit");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov

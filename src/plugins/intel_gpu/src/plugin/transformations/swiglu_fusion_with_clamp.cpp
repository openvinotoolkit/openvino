// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swiglu_fusion_with_clamp.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "intel_gpu/op/swiglu_with_clamp.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

SwiGluFusionWithClamp::SwiGluFusionWithClamp() {
    using namespace ov::pass::pattern;

    // Detect GLU decomposition pattern
    auto gate_up_matmul = wrap_type<ov::op::v0::MatMul>({any_input(), any_input()});
                                                       // {{"transpose_a", false}, {"transpose_b", false}});
    auto gate_up_add = wrap_type<ov::op::v1::Add>({gate_up_matmul, any_input()});

    auto slice1_start_const = wrap_type<ov::op::v0::Constant>();
    auto slice1_end_const = wrap_type<ov::op::v0::Constant>();
    auto slice_axis_const = wrap_type<ov::op::v0::Constant>();
    auto slice_step_const = wrap_type<ov::op::v0::Constant>();
    // Branch 2: Slice_2 -> Minimum_1 -> Swish
    auto slice1 = wrap_type<ov::op::v8::Slice>({gate_up_add, slice1_start_const,
            slice1_end_const, slice_step_const, slice_axis_const});
    auto minimum = wrap_type<ov::op::v1::Minimum>({slice1, wrap_const()});
    auto swish = wrap_type<ov::op::v4::Swish>({minimum, any_input()});

    auto slice2_start_const = wrap_type<ov::op::v0::Constant>();
    auto slice2_end_const = wrap_type<ov::op::v0::Constant>();
    auto slice2 = wrap_type<ov::op::v8::Slice>({gate_up_add, slice2_start_const,
        slice2_end_const, slice_step_const, slice_axis_const});

    // Branch 1: Slice_1 -> Clamp -> Add_1
    auto clamp = wrap_type<ov::op::v0::Clamp>({slice2});
    auto add1 = wrap_type<ov::op::v1::Add>({clamp, wrap_const()});

    // Join: Multiply_1
    auto multiply1 = wrap_type<ov::op::v1::Multiply>({swish, add1});
    // Down projection
    auto down_proj_matmul = wrap_type<ov::op::v0::MatMul>({multiply1, any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto swish_node = ov::as_type_ptr<ov::op::v4::Swish>(pattern_map.at(swish).get_node_shared_ptr());
        auto mul_node = ov::as_type_ptr<ov::op::v1::Multiply>(pattern_map.at(multiply1).get_node_shared_ptr());
        if (!mul_node || transformation_callback(mul_node))
            return false;

        auto clamp_node = ov::as_type_ptr<ov::op::v0::Clamp>(pattern_map.at(clamp).get_node_shared_ptr());
        auto clamp_min_value = clamp_node->get_min();
        auto clamp_max_value = clamp_node->get_max();

        size_t gate_idx = 0;
        if (ov::is_type<ov::op::v4::Swish>(mul_node->get_input_node_shared_ptr(1)))
            gate_idx = 1;

        auto slice_node =
            ov::as_type_ptr<ov::op::v8::Slice>(pattern_map.at(slice1).get_node_shared_ptr());
        auto slice_in_ps = slice_node->get_input_partial_shape(0);
        auto last_dim = slice_in_ps.rank().get_length() - 1;

        auto axis = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(slice_axis_const).get_node_shared_ptr());
        bool valid_axis_const_values = ov::op::util::has_constant_value<int64_t>(axis, -1) ||
                                       ov::op::util::has_constant_value<int64_t>(axis, last_dim);
        if (!valid_axis_const_values)
             return false;
        auto axis_value = axis->cast_vector<int64_t>()[0];

        auto slice1_start_node =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(slice1_start_const).get_node_shared_ptr());
        auto slice1_end_node =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(slice1_end_const).get_node_shared_ptr());
        auto split_lengths_value = slice1_end_node->cast_vector<int64_t>()[0] -  slice1_start_node->cast_vector<int64_t>()[0];
        // Allow only case that exactly splits in half along the last dimension
        auto split_length = slice_in_ps[last_dim].get_length() / 2;
        if (split_lengths_value != split_length)
            return false;

        auto slice_step_node =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(slice_step_const).get_node_shared_ptr());
        auto step_value = slice_step_node->cast_vector<int64_t>()[0];
        if (step_value != 1)
            return false;

        auto data_node = pattern_map.at(gate_up_matmul);
        auto output_type = m.get_match_root()->get_output_element_type(0);

        auto swiglu = std::make_shared<op::SwiGluWithClamp>(data_node,
                                                            axis_value,
                                                            split_lengths_value,
                                                            ov::op::internal::GLU::GluType::Swish,
                                                            gate_idx,
                                                            clamp_min_value,
                                                            clamp_max_value,
                                                            1.0f,  // TODO : handle case when swish_beta input is given
                                                            0.0f,  // TODO  handle case when up_add_val input is given
                                                            output_type);
        swiglu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), swiglu);
        ov::replace_node(m.get_match_root(), swiglu);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(down_proj_matmul,
        "SWIGLU_FUSION_WITH_CLAMP");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov

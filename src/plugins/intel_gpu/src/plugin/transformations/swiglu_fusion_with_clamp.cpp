// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swiglu_fusion_with_clamp.hpp"

#include "intel_gpu/op/swiglu_with_clamp.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {
    SwiGluFusionWithClamp::SwiGluFusionWithClamp() {
        using namespace ov::pass::pattern;

        // Detect GLU decomposition pattern
        auto gate_up_matmul_m = wrap_type<ov::op::v0::MatMul>({any_input(), any_input()});
        auto gate_up_add_m = wrap_type<ov::op::v1::Add>({gate_up_matmul_m, any_input()});

        // Branch 1: Slice_1 -> Minimum_1 -> Swish
        auto slice1_start_const_m = wrap_type<ov::op::v0::Constant>();
        auto slice1_end_const_m = wrap_type<ov::op::v0::Constant>();
        auto slice1_axis_const_m = wrap_type<ov::op::v0::Constant>();
        auto slice1_step_const_m = wrap_type<ov::op::v0::Constant>();
        auto slice1_m = wrap_type<ov::op::v8::Slice>({gate_up_add_m, slice1_start_const_m, slice1_end_const_m, slice1_step_const_m, slice1_axis_const_m});
        auto minimum_m = wrap_type<ov::op::v1::Minimum>({slice1_m, wrap_const()});
        auto swish_beta_m = wrap_type<ov::op::v0::Constant>();
        auto swish_m = wrap_type<ov::op::v4::Swish>({minimum_m, swish_beta_m});

        // Branch 2: Slice_2 -> Clamp -> Add_1
        auto slice2_start_const_m = wrap_type<ov::op::v0::Constant>();
        auto slice2_end_const_m = wrap_type<ov::op::v0::Constant>();
        auto slice2_step_const_m = wrap_type<ov::op::v0::Constant>();
        auto slice2_axis_const_m = wrap_type<ov::op::v0::Constant>();
        auto slice2_m = wrap_type<ov::op::v8::Slice>({gate_up_add_m, slice2_start_const_m, slice2_end_const_m, slice2_step_const_m, slice2_axis_const_m});
        auto clamp_m = wrap_type<ov::op::v0::Clamp>({slice2_m});
        auto added_const_m = wrap_type<ov::op::v0::Constant>();
        auto add1_m = wrap_type<ov::op::v1::Add>({clamp_m, added_const_m});

        // Join: Multiply_1
        auto multiply1_m = wrap_type<ov::op::v1::Multiply>({swish_m, add1_m});
        // Down projection
        auto down_proj_matmul_m = wrap_type<ov::op::v0::MatMul>({multiply1_m, any_input()});

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto swish_node_ptr = ov::as_type_ptr<ov::op::v4::Swish>(pattern_map.at(swish_m).get_node_shared_ptr());
            auto mul_node_ptr = ov::as_type_ptr<ov::op::v1::Multiply>(pattern_map.at(multiply1_m).get_node_shared_ptr());
            if (!mul_node_ptr || transformation_callback(mul_node_ptr))
                return false;

            auto clamp_node = ov::as_type_ptr<ov::op::v0::Clamp>(pattern_map.at(clamp_m).get_node_shared_ptr());
            auto clamp_min_value = clamp_node->get_min();
            auto clamp_max_value = clamp_node->get_max();

            size_t gate_idx = 0;
            if (ov::is_type<ov::op::v4::Swish>(mul_node_ptr->get_input_node_shared_ptr(1)))
                gate_idx = 1;

            auto slice1_node_ptr = ov::as_type_ptr<ov::op::v8::Slice>(pattern_map.at(slice1_m).get_node_shared_ptr());
            auto slice1_in_ps = slice1_node_ptr->get_input_partial_shape(0);
            auto last_dim = slice1_in_ps.rank().get_length() - 1;

            auto axis_node_ptr = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(slice1_axis_const_m).get_node_shared_ptr());
            bool valid_axis_const_values = ov::op::util::has_constant_value<int64_t>(axis_node_ptr, -1)
                                            || ov::op::util::has_constant_value<int64_t>(axis_node_ptr, last_dim);
            // only innermost axis supported
            if (!valid_axis_const_values)
                return false;
            auto axis_dim = axis_node_ptr->cast_vector<int64_t>()[0];

            auto slice1_start_node_ptr = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(slice1_start_const_m).get_node_shared_ptr());
            auto slice1_end_node_ptr = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(slice1_end_const_m).get_node_shared_ptr());
            auto slice1_step_node_ptr = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(slice1_step_const_m).get_node_shared_ptr());
            auto slice_1_start_val = slice1_start_node_ptr->cast_vector<int64_t>()[0];
            auto slice_1_end_val = slice1_end_node_ptr->cast_vector<int64_t>()[0];
            auto slice_1_step_val = slice1_step_node_ptr->cast_vector<int64_t>()[0];
            int64_t glu_stride = slice_1_end_val - slice_1_start_val;
            if (slice_1_step_val == 2) {
                // Alternating
                glu_stride = slice_1_step_val;
            } else if (slice_1_step_val == 1) {
                if (glu_stride != slice1_in_ps[axis_dim].get_length() / 2) {
                    // Allow only case that exactly splits in half along the last dimension
                    return false;
                }
            } else {
                return false;
            }
            if (pattern_map.find(swish_beta_m) == pattern_map.end())
                return false;
            auto swish_beta_node_ptr = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(swish_beta_m).get_node_shared_ptr());
            auto swish_beta_val = static_cast<float>(swish_beta_node_ptr->cast_vector<double>()[0]);
            auto add_const_node_ptr = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(added_const_m).get_node_shared_ptr());
            auto added_const_val = static_cast<float>(add_const_node_ptr->cast_vector<double>()[0]);

            auto data_node = pattern_map.at(gate_up_matmul_m);
            auto output_type = m.get_match_root()->get_output_element_type(0);

            auto swiglu = std::make_shared<op::SwiGluWithClamp>(data_node,
                    axis_dim,
                    glu_stride,
                    ov::op::internal::GLU::GluType::Swish,
                    gate_idx,
                    clamp_min_value,
                    clamp_max_value,
                    swish_beta_val,
                    added_const_val,
                    output_type);
            swiglu->set_friendly_name(m.get_match_root()->get_friendly_name());
            ov::copy_runtime_info(m.get_matched_nodes(), swiglu);
            ov::replace_node(m.get_match_root(), swiglu);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(down_proj_matmul_m, "SWIGLU_FUSION_WITH_CLAMP");
        this->register_matcher(m, callback);
    }

}  // namespace intel_gpu
}  // namespace ov

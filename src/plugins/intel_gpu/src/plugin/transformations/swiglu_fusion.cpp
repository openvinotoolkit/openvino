// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swiglu_fusion.hpp"

#include "intel_gpu/op/swiglu.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

SwiGLUFusion::SwiGLUFusion() {
    using namespace ov::pass::pattern;

    auto data = any_input();
    auto const_axis = wrap_type<ov::op::v0::Constant>();
    auto const_split_lengths = wrap_type<ov::op::v0::Constant>();
    auto variadic_split = wrap_type<ov::op::v1::VariadicSplit>({data, const_axis, const_split_lengths});
    auto swish = wrap_type<ov::op::v4::Swish>({variadic_split});
    auto mul = wrap_type<ov::op::v1::Multiply>({swish, variadic_split});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto input = pattern_map.at(data);

        auto axis = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(const_axis).get_node_shared_ptr());
        if (!axis)
            return false;
        if (!ov::op::util::has_constant_value<int64_t>(axis, -1))
            return false;
        auto split_lengths = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(const_split_lengths).get_node_shared_ptr());
        if (!split_lengths)
            return false;
        auto axis_value = axis->cast_vector<int64_t>();
        auto split_lengths_value = split_lengths->cast_vector<int64_t>();

        auto output_type = m.get_match_root()->get_output_element_type(0);

        auto swiglu = std::make_shared<op::SwiGLU>(input,
                                                   axis_value,
                                                   split_lengths_value,
                                                   output_type);
        swiglu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), swiglu);
        ov::replace_node(m.get_match_root(), swiglu);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul, "SwiGLUFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov

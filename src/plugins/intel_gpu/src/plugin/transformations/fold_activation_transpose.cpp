// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fold_activation_transpose.hpp"

#include <memory>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/core/graph_util.hpp"

#include "compressed_weights_pattern.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

FoldActivationTranspose::FoldActivationTranspose() {
    auto a_input_m = any_input();
    auto a_order_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::shape_matches("[4]"));
    auto a_transpose_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({a_input_m, a_order_m});
    auto swish_m = ov::pass::pattern::wrap_type<ov::op::v4::Swish>({a_transpose_m});
    auto b_input_m = any_input();
    auto b_order_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::shape_matches("[4]"));
    auto b_transpose_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({b_input_m, b_order_m});
    auto mul_m = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({swish_m, b_transpose_m});
    auto c_order_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::shape_matches("[4]"));
    auto c_transpose_m = wrap_type<ov::op::v1::Transpose>({mul_m, c_order_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        // verify that the orders of A and B match and are the inverse of order of C
        auto a_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(a_order_m).get_node_shared_ptr());
        const auto a_order_value = a_order->cast_vector<int64_t>();
        auto b_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(b_order_m).get_node_shared_ptr());
        const auto b_order_value = b_order->cast_vector<int64_t>();
        auto c_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(c_order_m).get_node_shared_ptr());
        const auto c_order_value = c_order->cast_vector<int64_t>();
        for (int i = 0; i < 4; ++i) {
            if (a_order_value[i] != b_order_value[i] || c_order_value[a_order_value[i]] != i)
                return false;
        }

        auto a_input = pattern_map.at(a_input_m).get_node_shared_ptr();
        auto swish = pattern_map.at(swish_m).get_node_shared_ptr();
        auto swish_new = swish->clone_with_new_inputs({a_input});
        auto b_input = pattern_map.at(b_input_m).get_node_shared_ptr();
        auto mul = pattern_map.at(mul_m).get_node_shared_ptr();
        auto mul_new = mul->clone_with_new_inputs({swish_new, b_input});
        auto c_transpose = pattern_map.at(c_transpose_m).get_node_shared_ptr();

        ov::copy_runtime_info(swish, swish_new);
        swish_new->set_friendly_name(swish->get_friendly_name());
        ov::copy_runtime_info(mul, mul_new);
        mul_new->set_friendly_name(c_transpose->get_friendly_name());
        ov::replace_node(c_transpose, mul_new);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(c_transpose_m, "FoldActivationTranspose");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu

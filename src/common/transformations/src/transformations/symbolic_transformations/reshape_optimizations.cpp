// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/reshape_optimizations.hpp"

#include "compare.hpp"
#include "itt.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/symbolic_transformations/utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::symbol::util;

ov::pass::ReshapeOptimizations::ReshapeOptimizations() {
    MATCHER_SCOPE(ReshapeOptimizations);
    auto data_label = pattern::any_input(pattern::has_static_rank());
    auto pattern_label = pattern::any_input(pattern::has_static_shape() && pattern::class_other_than<v0::Constant>());
    auto reshape_label = pattern::wrap_type<op::v1::Reshape>({data_label, pattern_label}, pattern::has_static_rank());

    ov::matcher_pass_callback matcher_pass_callback = [](pattern::Matcher& m) {
        const auto& reshape = ov::as_type_ptr<v1::Reshape>(m.get_match_root());
        if (!reshape)
            return false;
        const auto& in_shape = reshape->get_input_partial_shape(0);
        const auto& in_rank = in_shape.size();
        const auto& out_shape = reshape->get_output_partial_shape(0);
        const auto& out_rank = out_shape.size();

        int64_t cnt_static_zeros = 0;
        std::vector<int64_t> output_pattern(out_rank, -1);
        for (size_t i = 0; i < out_rank; ++i) {
            if (out_shape[i].is_static()) {
                output_pattern[i] = out_shape[i].get_length();
                if (output_pattern[i] == 0) {
                    ++cnt_static_zeros;
                }
            } else if (i >= in_rank) {
                break;
            } else if (dims_are_equal(in_shape[i], out_shape[i])) {
                output_pattern[i] = 0;
            }
        }

        int64_t cnt_neg_ones = std::count(output_pattern.begin(), output_pattern.end(), -1);
        if (cnt_neg_ones == 0 || (cnt_neg_ones == 1 && cnt_static_zeros == 0)) {
            auto new_pattern = ov::op::v0::Constant::create(element::i64, Shape{output_pattern.size()}, output_pattern);
            ov::copy_runtime_info(reshape->get_input_node_shared_ptr(1), new_pattern);
            reshape->set_special_zero(true);
            reshape->input(1).replace_source_output(new_pattern->output(0));
            return true;
        }
        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

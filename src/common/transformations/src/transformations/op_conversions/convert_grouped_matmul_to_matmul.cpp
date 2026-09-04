// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_grouped_matmul_to_matmul.hpp"

#include <cstdint>
#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

namespace {

namespace v0 = ov::op::v0;
namespace v8 = ov::op::v8;
namespace v17 = ov::op::v17;

}  // namespace

ConvertGroupedMatMulToMatMul::ConvertGroupedMatMulToMatMul() {
    MATCHER_SCOPE(ConvertGroupedMatMulToMatMul);
    using namespace ov::pass::pattern;

    auto matrix_b_3d = any_input(rank_equals(3));

    // 3Dx3D: A:[G,M,K] B:[G,N,K]
    auto matrix_a_3d = any_input(rank_equals(3));
    auto gmm_3d_3d = wrap_type<v17::GroupedMatMul>({matrix_a_3d, matrix_b_3d});

    // 2Dx3D: A:[T,K] B:[G,N,K] offsets:[G]
    auto matrix_a_2d = any_input(rank_equals(2));
    auto offsets = any_input(rank_equals(1));
    auto gmm_2d_3d = wrap_type<v17::GroupedMatMul>({matrix_a_2d, matrix_b_3d, offsets});

    auto gmm_pattern = gmm_3d_3d | gmm_2d_3d;

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto gmm = ov::as_type_ptr<v17::GroupedMatMul>(m.get_match_root());
        if (!gmm || transformation_callback(gmm)) {
            return false;
        }
        const auto mat_a = gmm->input_value(0);
        const auto mat_b = gmm->input_value(1);

        NodeRegistry rg;
        std::shared_ptr<ov::Node> replacement;
        const auto& pattern_map = m.get_pattern_value_map();
        if (pattern_map.count(gmm_3d_3d)) {
            // A:[G,M,K] B:[G,N,K] -> MatMul(A, B, transpose_b=true) -> [G,M,N]
            replacement = rg.make<v0::MatMul>(mat_a, mat_b, false, true);
        } else if (pattern_map.count(gmm_2d_3d)) {
            // A fixed MatMul per group is emitted, so the group count G must be static.
            const auto& b_partial_shape = mat_b.get_partial_shape();
            if (b_partial_shape[0].is_dynamic()) {
                return false;
            }
            const auto num_groups = b_partial_shape[0].get_length();

            const auto i64 = ov::element::i64;

            const auto offsets_in = gmm->input_value(2);
            ov::Output<ov::Node> offsets_i64 = offsets_in;
            if (offsets_in.get_element_type() != i64) {
                offsets_i64 = rg.make<v0::Convert>(offsets_in, i64);
            }

            auto step = rg.make<v0::Constant>(i64, ov::Shape{1}, 1);
            auto slice_axis = rg.make<v0::Constant>(i64, ov::Shape{1}, 0);
            auto gather_axis = rg.make<v0::Constant>(i64, ov::Shape{}, 0);

            ov::OutputVector group_outputs;
            group_outputs.reserve(static_cast<size_t>(num_groups));

            ov::Output<ov::Node> start = rg.make<v0::Constant>(i64, ov::Shape{1}, 0);
            for (int64_t g = 0; g < num_groups; ++g) {
                auto end_index = rg.make<v0::Constant>(i64, ov::Shape{1}, g);
                ov::Output<ov::Node> end = rg.make<v8::Gather>(offsets_i64, end_index, gather_axis);

                auto a_g = rg.make<v8::Slice>(mat_a, start, end, step, slice_axis);  // [Mg, K]
                auto b_index = rg.make<v0::Constant>(i64, ov::Shape{}, g);
                auto b_g = rg.make<v8::Gather>(mat_b, b_index, gather_axis);  // [N, K]
                auto mm = rg.make<v0::MatMul>(a_g, b_g, false, true);  // [Mg, N]
                group_outputs.push_back(mm);

                start = end;
            }

            replacement = rg.make<v0::Concat>(group_outputs, 0);  // [T, N]
        } else {
            return false;
        }

        replacement->set_friendly_name(gmm->get_friendly_name());
        ov::copy_runtime_info(gmm, rg.get());
        ov::replace_node(gmm, replacement);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gmm_pattern, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::pass

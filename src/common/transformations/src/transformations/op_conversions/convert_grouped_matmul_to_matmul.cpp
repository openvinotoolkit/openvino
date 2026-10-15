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
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;
namespace v17 = ov::op::v17;

}  // namespace

ConvertGroupedMatMul3Dx3DToMatMul::ConvertGroupedMatMul3Dx3DToMatMul() {
    MATCHER_SCOPE(ConvertGroupedMatMul3Dx3DToMatMul);
    using namespace ov::pass::pattern;

    // 3Dx3D: A:[G,M,K] B:[G,N,K]
    auto matrix_a_3d = any_input(rank_equals(3));
    auto matrix_b_3d = any_input(rank_equals(3));
    auto gmm_3d_3d = wrap_type<v17::GroupedMatMul>({matrix_a_3d, matrix_b_3d});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto gmm = ov::as_type_ptr<v17::GroupedMatMul>(m.get_match_root());
        if (!gmm || transformation_callback(gmm)) {
            return false;
        }
        const auto mat_a = gmm->input_value(0);
        const auto mat_b = gmm->input_value(1);

        NodeRegistry rg;
        // A:[G,M,K] B:[G,N,K] -> MatMul(A, B, transpose_b=true) -> [G,M,N]
        auto replacement = rg.make<v0::MatMul>(mat_a, mat_b, false, true);

        replacement->set_friendly_name(gmm->get_friendly_name());
        ov::copy_runtime_info(gmm, rg.get());
        ov::replace_node(gmm, replacement);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gmm_3d_3d, matcher_name);
    register_matcher(matcher, callback);
}

ConvertGroupedMatMul2Dx3DToMatMul::ConvertGroupedMatMul2Dx3DToMatMul() {
    MATCHER_SCOPE(ConvertGroupedMatMul2Dx3DToMatMul);
    using namespace ov::pass::pattern;

    // 2Dx3D: A:[T,K] B:[G,N,K] offsets:[G]
    // A fixed MatMul per group is emitted, so the group count G (B.shape[0]) must be static.
    auto matrix_a_2d = any_input(rank_equals(2));
    auto matrix_b_3d = any_input(rank_equals(3) && has_static_dim(0));
    auto offsets = any_input(rank_equals(1));
    auto gmm_2d_3d = wrap_type<v17::GroupedMatMul>({matrix_a_2d, matrix_b_3d, offsets});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto gmm = ov::as_type_ptr<v17::GroupedMatMul>(m.get_match_root());
        if (!gmm || transformation_callback(gmm)) {
            return false;
        }
        const auto mat_a = gmm->input_value(0);
        const auto mat_b = gmm->input_value(1);

        const auto num_groups = mat_b.get_partial_shape()[0].get_length();

        using ov::element::i32;

        NodeRegistry rg;

        const auto offsets_in = gmm->input_value(2);
        const auto offsets_et = offsets_in.get_element_type();
        ov::Output<ov::Node> offsets_i32 = offsets_in;
        if (offsets_et == ov::element::i64) {
            offsets_i32 = rg.make<v0::Convert>(offsets_in, i32);
        } else if (offsets_et != i32) {
            return false;
        }

        auto axis_scalar = rg.make<v0::Constant>(i32, ov::Shape{}, 0);
        auto axis_1d = rg.make<v0::Constant>(i32, ov::Shape{1}, 0);

        // offsets are cumulative ends, so per-group row counts are offsets - [0, offsets[:-1]].
        ov::Output<ov::Node> split_lengths = offsets_i32;
        if (num_groups > 1) {
            auto zero = rg.make<v0::Constant>(i32, ov::Shape{1}, 0);
            auto head_stop = rg.make<v0::Constant>(i32, ov::Shape{1}, num_groups - 1);
            auto step = rg.make<v0::Constant>(i32, ov::Shape{1}, 1);
            auto offsets_head = rg.make<v8::Slice>(offsets_i32, zero, head_stop, step, axis_1d);
            auto shifted = rg.make<v0::Concat>(ov::OutputVector{zero, offsets_head}, 0);
            split_lengths = rg.make<v1::Subtract>(offsets_i32, shifted);
        }

        auto a_split = rg.make<v1::VariadicSplit>(mat_a, axis_scalar, split_lengths);  // G x [Mg, K]
        auto b_split = rg.make<v1::Split>(mat_b, axis_scalar, num_groups);             // G x [1, N, K]

        ov::OutputVector group_outputs;
        group_outputs.reserve(static_cast<size_t>(num_groups));

        for (int64_t g = 0; g < num_groups; ++g) {
            auto b_g = rg.make<v0::Squeeze>(b_split->output(g), axis_1d);                  // [N, K]
            auto mm = rg.make<v0::MatMul>(a_split->output(g), b_g, false, true);           // [Mg, N]
            group_outputs.push_back(std::move(mm));
        }

        auto replacement = rg.make<v0::Concat>(group_outputs, 0);  // [T, N]

        replacement->set_friendly_name(gmm->get_friendly_name());
        ov::copy_runtime_info(gmm, rg.get());
        ov::replace_node(gmm, replacement);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gmm_2d_3d, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::pass

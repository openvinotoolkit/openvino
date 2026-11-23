// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/pattern_blocks/rope_pattern.hpp"

#include <cstdint>
#include <limits>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass::pattern {

GPTNeoXRoPEPattern::GPTNeoXRoPEPattern(const std::shared_ptr<Node>& input, int rank, std::string half_ndims_symbol)
    : half_ndims_symbol(std::move(half_ndims_symbol)) {
    namespace v0 = ov::op::v0;
    namespace v1 = ov::op::v1;

    OPENVINO_ASSERT(input && input->get_output_size() == 1, "GPTNeoXRoPEPattern expects a single-output input");
    OPENVINO_ASSERT(rank > 0, "GPTNeoXRoPEPattern expects a positive rank");
    OPENVINO_ASSERT(!this->half_ndims_symbol.empty(), "GPTNeoXRoPEPattern expects a half-dimension symbol");

    anchors.input = {"input", input};
    anchors.cos_input_1 = {"cos_input_1", any_input(rank_equals(rank))};
    anchors.cos_input_2 = {"cos_input_2", any_input(rank_equals(rank))};
    anchors.sin = {"sin", any_input(rank_equals(rank))};

    auto split =
        wrap_type_strict_index<v1::VariadicSplit>({input, rank - 1, {this->half_ndims_symbol, std::string("?")}});
    auto x2 = ov::op::util::NewGenSlice(input,
                                        this->half_ndims_symbol,
                                        std::numeric_limits<std::int32_t>::max(),
                                        1,
                                        rank - 1);
    auto minus_one = wrap_type<v0::Constant>(value_matches("-1"));
    anchors.minus_one_convert = {"minus_one_convert", optional<v0::Convert>({minus_one})};
    anchors.negated_half = {"negated_half",
                            wrap_type<v1::Multiply>({x2 | split->output(1), anchors.minus_one_convert.node},
                                                    {{"auto_broadcast", "numpy"}})};
    auto x1 = ov::op::util::NewGenSlice(input, 0, this->half_ndims_symbol, 1, rank - 1);
    anchors.rotated_input = {"rotated_input",
                             wrap_type<v0::Concat>({anchors.negated_half.node, x1 | split->output(0)}, {{"axis", -1}})};

    anchors.mul_cos = {
        "mul_cos",
        wrap_type<v1::Multiply>({anchors.cos_input_1.node, anchors.cos_input_2.node}, {{"auto_broadcast", "numpy"}})};
    anchors.mul_sin = {
        "mul_sin",
        wrap_type<v1::Multiply>({anchors.rotated_input.node, anchors.sin.node}, {{"auto_broadcast", "numpy"}})};
    anchors.root = {"root",
                    wrap_type<v1::Add>({anchors.mul_cos.node, anchors.mul_sin.node}, {{"auto_broadcast", "numpy"}})};
}

std::array<const PatternAnchor*, 10> GPTNeoXRoPEPattern::Anchors::all() const& {
    return {&input,
            &cos_input_1,
            &cos_input_2,
            &sin,
            &minus_one_convert,
            &negated_half,
            &rotated_input,
            &mul_cos,
            &mul_sin,
            &root};
}

}  // namespace ov::pass::pattern

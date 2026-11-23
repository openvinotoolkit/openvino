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
    : input(input),
      half_ndims_symbol(std::move(half_ndims_symbol)) {
    namespace v0 = ov::op::v0;
    namespace v1 = ov::op::v1;

    OPENVINO_ASSERT(input && input->get_output_size() == 1, "GPTNeoXRoPEPattern expects a single-output input");
    OPENVINO_ASSERT(rank > 0, "GPTNeoXRoPEPattern expects a positive rank");
    OPENVINO_ASSERT(!this->half_ndims_symbol.empty(), "GPTNeoXRoPEPattern expects a half-dimension symbol");

    cos_input_1 = any_input(rank_equals(rank));
    cos_input_2 = any_input(rank_equals(rank));
    sin = any_input(rank_equals(rank));

    auto split =
        wrap_type_strict_index<v1::VariadicSplit>({input, rank - 1, {this->half_ndims_symbol, std::string("?")}});
    auto x2 = ov::op::util::NewGenSlice(input,
                                        this->half_ndims_symbol,
                                        std::numeric_limits<std::int32_t>::max(),
                                        1,
                                        rank - 1);
    auto minus_one = wrap_type<v0::Constant>(value_matches("-1"));
    minus_one_convert = optional<v0::Convert>({minus_one});
    negated_half = wrap_type<v1::Multiply>({x2 | split->output(1), minus_one_convert}, {{"auto_broadcast", "numpy"}});
    auto x1 = ov::op::util::NewGenSlice(input, 0, this->half_ndims_symbol, 1, rank - 1);
    rotated_input = wrap_type<v0::Concat>({negated_half, x1 | split->output(0)}, {{"axis", -1}});

    mul_cos = wrap_type<v1::Multiply>({cos_input_1, cos_input_2}, {{"auto_broadcast", "numpy"}});
    mul_sin = wrap_type<v1::Multiply>({rotated_input, sin}, {{"auto_broadcast", "numpy"}});
    root = wrap_type<v1::Add>({mul_cos, mul_sin}, {{"auto_broadcast", "numpy"}});
}

std::array<PatternAnchor, 10> GPTNeoXRoPEPattern::anchors() const {
    return {{{"input", input},
             {"cos_input_1", cos_input_1},
             {"cos_input_2", cos_input_2},
             {"sin", sin},
             {"minus_one_convert", minus_one_convert},
             {"negated_half", negated_half},
             {"rotated_input", rotated_input},
             {"mul_cos", mul_cos},
             {"mul_sin", mul_sin},
             {"root", root}}};
}

}  // namespace ov::pass::pattern

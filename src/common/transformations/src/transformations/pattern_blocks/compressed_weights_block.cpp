// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/pattern_blocks/compressed_weights_block.hpp"

#include <algorithm>
#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/fully_connected.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "transformations/utils/utils.hpp"

using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
ov::pass::pattern::op::CompressedWeightsBlock::CompressedWeightsBlock(
    const std::vector<ov::element::Type>& supported_weights_types,
    const std::set<size_t>& supported_weights_ranks)
    : Block({}, {}, "CompressedWeightsBlock") {
    auto weights = wrap_type<v0::Constant>(ov::pass::pattern::type_matches_any(supported_weights_types));
    auto convert = wrap_type<v0::Convert>({weights});

    auto sub_const = wrap_type<v0::Constant>();
    auto sub_convert_const = wrap_type<v0::Convert>({sub_const});
    auto sub_with_convert = wrap_type<v1::Subtract>({convert, sub_convert_const});
    auto sub_no_convert = wrap_type<v1::Subtract>({convert, sub_const});
    auto subtract = sub_with_convert | sub_no_convert;

    auto mul_const = wrap_type<v0::Constant>();
    auto mul_convert_const = wrap_type<v0::Convert>({mul_const});
    auto mul_scale = mul_const | mul_convert_const;

    auto mul_with_sub = wrap_type<v1::Multiply>({subtract, mul_scale});
    auto mul_no_sub = wrap_type<v1::Multiply>({convert, mul_scale});
    auto mul = mul_with_sub | mul_no_sub;

    auto reshape_predicate = [supported_weights_ranks](const ov::Output<ov::Node>& output) {
        const auto& in_ps = output.get_node()->get_input_partial_shape(0);
        const auto& out_ps = output.get_node()->get_output_partial_shape(0);
        std::set<size_t> supported_weights_ranks_before_reshape;
        for (auto r : supported_weights_ranks) {
            supported_weights_ranks_before_reshape.insert(r + 1);
        }
        return in_ps.rank().is_static() && out_ps.rank().is_static() &&
               supported_weights_ranks_before_reshape.count(in_ps.size()) &&
               supported_weights_ranks.count(out_ps.size());
    };
    auto reshape_const = wrap_type<v0::Constant>();
    auto reshape = wrap_type<v1::Reshape>({mul, reshape_const}, reshape_predicate);

    auto transpose_input = reshape | mul;
    auto transpose_const = wrap_type<v0::Constant>();
    auto transpose = wrap_type<v1::Transpose>({transpose_input, transpose_const});

    auto weights_input = ov::pass::pattern::optional<v0::Convert>({reshape | transpose | mul});

    // Block initialization
    m_inputs = ov::OutputVector{weights};
    m_outputs = ov::OutputVector{weights_input};
    REGISTER_ANCHORS(this,
                     weights,
                     convert,
                     sub_const,
                     sub_with_convert,
                     sub_no_convert,
                     mul_const,
                     transpose,
                     transpose_const);
}
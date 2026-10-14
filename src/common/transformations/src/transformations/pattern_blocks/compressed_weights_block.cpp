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
    const std::set<size_t>& supported_weights_ranks,
    bool allow_two_level_scale)
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

    // A group scale that is itself quantized: an 8-bit per-group scale times one per-tensor factor.
    //     Constant(f8) -> Convert(f32) -> Multiply(Constant f32) -> Convert(f16)
    // NNCF writes this for NVFP4 and, via --experts-fp8-scale, for INT2 MoE experts. Note the anchors
    // fall out for free: `mul_const` binds to the 8-bit group scale exactly as in the plain case, so
    // everything downstream that reads it (combine_groups, the scale_shape checks) is unaffected --
    // only the new `scale_global_const` anchor is extra. Off by default because a consumer that
    // ignores that anchor would scale every weight by the wrong factor, silently.
    auto scale_global_const = wrap_type<v0::Constant>();
    if (allow_two_level_scale) {
        auto scale_dequant = wrap_type<v1::Multiply>({mul_convert_const, scale_global_const});
        mul_scale = mul_scale | wrap_type<v0::Convert>({scale_dequant});
    }

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
                     scale_global_const,
                     transpose,
                     transpose_const);
}
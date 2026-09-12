// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mark_compressed_weights_cast.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

namespace {
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
}  // namespace

MarkCompressedWeightsCast::MarkCompressedWeightsCast() {
    using namespace ov::pass::pattern;
    using ov::pass::operator|;  // declared in ov::pass, not in ov::pass::pattern

    // Kept deliberately in step with ov::pass::pattern::op::CompressedWeightsBlock: the point of
    // the mark is to let that block absorb the chain, so matching a shape it cannot absorb would
    // only disable constant folding for no gain. Hence at most one Reshape and one Transpose
    // between the Multiply and the cast, in that order.
    auto weights = wrap_type<v0::Constant>(type_matches_any({element::u4, element::i4, element::u8, element::i8}));
    auto convert = wrap_type<v0::Convert>({weights});

    // Zero point, present only for asymmetric compression. Left unconstrained (it may itself be a
    // Convert of a Constant), same as the scale below -- the compressed form is established by the
    // integer Constant and the dequantizing Multiply, not by how those operands are spelled.
    auto subtract = wrap_type<v1::Subtract>({convert, any_input()});
    auto scaled_input = subtract | convert;
    auto multiply = wrap_type<v1::Multiply>({scaled_input, any_input()});

    auto reshape = wrap_type<v1::Reshape>({multiply, any_input()});
    auto transpose = wrap_type<v1::Transpose>({reshape | multiply, any_input()});
    auto weights_output = transpose | reshape | multiply;

    auto cast = wrap_type<v0::Convert>(PatternOps{weights_output});

    register_matcher(std::make_shared<Matcher>(cast, "ov::frontend::pytorch::pass::MarkCompressedWeightsCast"),
                     [=](Matcher& m) {
                         auto cast_node = ov::as_type_ptr<v0::Convert>(m.get_match_root());
                         if (!cast_node) {
                             return false;
                         }
                         // A cast out of the float domain is not a precision cast of the dequantized weights.
                         if (!cast_node->get_destination_type().is_real()) {
                             return false;
                         }
                         if (ov::is_decompression(cast_node)) {
                             return false;
                         }
                         ov::mark_as_decompression(cast_node);
                         // The weights must not be materialized in the cast's destination type; the integer
                         // Constant underneath is already protected by MarkDequantization in the consuming
                         // plugin, this covers the cast itself for pipelines that mark neither.
                         ov::disable_constant_folding(cast_node);
                         // Nothing else in the graph changed, so report no rewrite.
                         return false;
                     });
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

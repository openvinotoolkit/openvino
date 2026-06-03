// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/common_optimizations/erase_redundant_convert_pair.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

// Frameworks that materialize a "model dtype" residual stream produce graphs
// where every wide-precision intermediate (typically f32 from a MatMul or
// from a decompressed weight) is wrapped in:
//
//   wide_value -> Convert(narrow) -> next op (residual / consumer expecting
//                                              narrow)
//                                 -> Convert(wide) -> next wide consumer
//
// The Convert(narrow) -> Convert(wide) chain on the wide-consumer branch is
// a value-preserving round-trip (modulo the narrowing rounding) and just
// costs two scans across the activation. This pass elides that round-trip
// by short-circuiting the second Convert's output to the first Convert's
// source, so the wide consumer reads the pre-narrowing value directly. The
// inner narrow Convert is left in place for the genuinely narrow-consuming
// branch (the residual stream); when it becomes unused, DCE removes it.
//
// Dtype-agnostic: matches any narrow-then-wide round-trip where the outer
// destination type equals the inner source type. ISA-agnostic: a pure graph
// rewrite, no assumption about AVX2/AVX-512/AMX.
EraseRedundantConvertPair::EraseRedundantConvertPair() {
    MATCHER_SCOPE(EraseRedundantConvertPair);
    using namespace pattern;

    auto inner_src = any_input();
    auto cvt_inner = wrap_type<ov::op::v0::Convert>({inner_src});
    auto cvt_outer = wrap_type<ov::op::v0::Convert>({cvt_inner});

    auto callback = [=](Matcher& m) -> bool {
        auto outer = ov::as_type_ptr<ov::op::v0::Convert>(m.get_match_root());
        if (!outer) return false;

        auto inner = ov::as_type_ptr<ov::op::v0::Convert>(outer->get_input_node_shared_ptr(0));
        if (!inner) return false;

        const auto src_type = inner->input_value(0).get_element_type();
        const auto narrow_type = inner->get_destination_type();
        const auto outer_dst = outer->get_destination_type();

        // Round-trip semantics require the outer destination to match the
        // original source type, and the inner destination to be a narrower
        // float type.
        if (outer_dst != src_type) return false;
        if (!src_type.is_real()) return false;
        if (!narrow_type.is_real()) return false;
        if (narrow_type.bitwidth() >= src_type.bitwidth()) return false;

        // Bypass the round-trip: feed the outer Convert's consumers from the
        // original wide source. The inner Convert is left in place; if its
        // narrow output is still needed by another branch (residual stream),
        // it stays; otherwise DCE removes it.
        auto src_val = inner->input_value(0);
        outer->output(0).replace(src_val);
        return true;
    };

    auto m = std::make_shared<Matcher>(cvt_outer, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass

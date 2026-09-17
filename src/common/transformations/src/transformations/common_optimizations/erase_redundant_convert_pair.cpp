// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/erase_redundant_convert_pair.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

// Eliminates redundant Converts: a round-trip pair
// wide->Convert(narrow)->Convert(wide), and an identity Convert(T)->T.
EraseRedundantConvertPair::EraseRedundantConvertPair() {
    MATCHER_SCOPE(EraseRedundantConvertPair);
    using namespace pattern;

    // Match any Convert; classify round-trip vs identity in the callback.
    auto cvt = wrap_type<ov::op::v0::Convert>();

    auto callback = [=](Matcher& m) -> bool {
        auto outer = ov::as_type_ptr<ov::op::v0::Convert>(m.get_match_root());
        if (!outer) return false;

        const auto outer_dst = outer->get_destination_type();
        auto outer_src_val = outer->input_value(0);
        const auto outer_src_type = outer_src_val.get_element_type();

        // Identity Convert: source dtype already matches the destination.
        if (outer_src_type == outer_dst) {
            outer->output(0).replace(outer_src_val);
            return true;
        }

        // Round-trip Convert pair: inner Convert narrows, outer widens back
        // to the original type.
        auto inner = ov::as_type_ptr<ov::op::v0::Convert>(outer_src_val.get_node_shared_ptr());
        if (!inner) return false;

        const auto src_type = inner->input_value(0).get_element_type();
        const auto narrow_type = inner->get_destination_type();

        if (outer_dst != src_type) return false;
        if (!src_type.is_real()) return false;
        if (!narrow_type.is_real()) return false;
        if (narrow_type.bitwidth() >= src_type.bitwidth()) return false;

        outer->output(0).replace(inner->input_value(0));
        return true;
    };

    auto m = std::make_shared<Matcher>(cvt, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass

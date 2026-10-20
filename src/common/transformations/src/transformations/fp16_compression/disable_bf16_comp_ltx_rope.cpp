// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/disable_bf16_comp_ltx_rope.hpp"

#include "itt.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace ov::pass {

DisableBF16CompForLtxVideoRopePattern::DisableBF16CompForLtxVideoRopePattern() {
    MATCHER_SCOPE(DisableBF16CompForLtxVideoRopePattern);
    using namespace ov::pass::pattern;

    // grid values are small; the large magnitude appears only from the frequency Multiply onwards
    auto mul = wrap_type<v1::Multiply>({any_input(), any_input()});
    auto add_constant = wrap_type<v0::Constant>();
    auto add = wrap_type<v1::Add>({mul, add_constant});
    auto transpose = wrap_type<v1::Transpose>({add, any_input()});
    auto reshape = wrap_type<v1::Reshape>({transpose, any_input()});
    auto sin_or_cos = wrap_type<v0::Sin, v0::Cos>({reshape});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        // disable_conversion tags each node with the shared "keep original precision" marker. This pass is
        // registered by the CPU plugin only under bf16, where EnforceInferencePrecision honors the tag and
        // leaves the angle chain in f32 instead of lowering it. element::f16 names the marker, not a target.
        for (const auto& node : {mul, add_constant, add, transpose, reshape, sin_or_cos}) {
            ov::disable_conversion(pattern_map.at(node).get_node_shared_ptr(), element::f16);
        }
        return false;
    };

    auto m = std::make_shared<Matcher>(sin_or_cos, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass

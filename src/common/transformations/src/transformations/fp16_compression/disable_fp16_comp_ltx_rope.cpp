// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/disable_fp16_comp_ltx_rope.hpp"

#include "itt.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace ov::pass {

DisableFP16CompForLtxVideoRopePattern::DisableFP16CompForLtxVideoRopePattern() {
    MATCHER_SCOPE(DisableFP16CompForLtxVideoRopePattern);
    using namespace ov::pass::pattern;

    // grid values are small; the large magnitude appears only from the frequency Multiply onwards
    auto mul = wrap_type<v1::Multiply>({any_input(), any_input()});
    auto add_constant = wrap_type<v0::Constant>();
    auto add = wrap_type<v1::Add>({mul, add_constant});
    auto transpose = wrap_type<v1::Transpose>({add, any_input()});
    auto reshape = wrap_type<v1::Reshape>({transpose, any_input()});
    auto sin = wrap_type<v0::Sin>({reshape});
    auto cos = wrap_type<v0::Cos>({reshape});
    auto sin_or_cos = std::make_shared<pattern::op::Or>(OutputVector{sin, cos});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        for (const auto& node : {mul, add_constant, add, transpose, reshape}) {
            ov::disable_conversion(pattern_map.at(node).get_node_shared_ptr(), element::f16);
        }
        ov::disable_conversion(m.get_match_root(), element::f16);
        return false;
    };

    auto m = std::make_shared<Matcher>(sin_or_cos, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass

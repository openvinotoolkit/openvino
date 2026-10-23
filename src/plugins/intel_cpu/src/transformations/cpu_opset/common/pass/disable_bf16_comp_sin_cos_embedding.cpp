// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_bf16_comp_sin_cos_embedding.hpp"

#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {

DisableBF16CompSinCosEmbedding::DisableBF16CompSinCosEmbedding() {
    MATCHER_SCOPE(DisableBF16CompSinCosEmbedding);
    using namespace ov::pass::pattern;

    auto freq_m = wrap_type<ov::op::v0::Constant>();
    auto mul_m = wrap_type<ov::op::v1::Multiply>({any_input(), freq_m});
    auto sin_or_cos_m = wrap_type<ov::op::v0::Sin, ov::op::v0::Cos>({mul_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sin_or_cos = pattern_map.at(sin_or_cos_m).get_node_shared_ptr();
        if (transformation_callback(sin_or_cos)) {
            return false;
        }

        ov::disable_conversion(pattern_map.at(mul_m).get_node_shared_ptr(), ov::element::f32, ov::element::bf16);
        ov::disable_conversion(sin_or_cos, ov::element::f32, ov::element::bf16);
        return false;
    };

    auto m = std::make_shared<Matcher>(sin_or_cos_m, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_cpu

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "clamp_bf16_scalar_constants.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"

namespace ov::intel_cpu {

ClampBF16ScalarConstants::ClampBF16ScalarConstants() {
    MATCHER_SCOPE(ClampBF16ScalarConstants);
    using namespace ov::pass::pattern;

    auto binary_m = wrap_type<ov::op::v1::Multiply, ov::op::v1::Add, ov::op::v1::Subtract>({any_input(), any_input()});
    const auto bf16_max = static_cast<float>(std::numeric_limits<ov::bfloat16>::max());

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto binary = m.get_match_root();
        bool changed = false;
        for (auto& input : binary->inputs()) {
            const auto constant =
                ov::as_type_ptr<ov::op::v0::Constant>(input.get_source_output().get_node_shared_ptr());
            if (!constant || constant->get_element_type() != ov::element::f32 ||
                ov::shape_size(constant->get_shape()) != 1) {
                continue;
            }
            const float value = constant->cast_vector<float>()[0];
            // inf and NaN are kept as is: only finite values that would overflow to inf in bf16 are saturated
            if (!std::isfinite(value) || std::abs(value) <= bf16_max) {
                continue;
            }

            auto clamped = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                  constant->get_shape(),
                                                                  std::clamp(value, -bf16_max, bf16_max));
            clamped->set_friendly_name(constant->get_friendly_name());
            ov::copy_runtime_info(constant, clamped);
            // Only this consumer is updated: the constant may be shared with ops that are not affected by bf16
            input.replace_source_output(clamped);
            changed = true;
        }
        return changed;
    };

    auto m = std::make_shared<Matcher>(binary_m, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_cpu

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/mark_math_before_discontinuous_ops_to_keep_f16_rounding.hpp"

#include "itt.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/selu.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::pass {

MarkMathBeforeDiscontinuousOpsToKeepF16Rounding::MarkMathBeforeDiscontinuousOpsToKeepF16Rounding() {
    MATCHER_SCOPE(MarkMathBeforeDiscontinuousOpsToKeepF16Rounding);
    using namespace ov::pass::pattern;

    auto discontinuous_op = wrap_type<ov::op::v0::Floor,
                                      ov::op::v0::Ceiling,
                                      ov::op::v5::Round,
                                      ov::op::v0::Sign,
                                      ov::op::v1::Mod,
                                      ov::op::v1::FloorMod>();

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& root = m.get_match_root();
        // Mod and FloorMod are binary, so any of the inputs may be produced by a Math op.
        for (const auto& input : root->input_values()) {
            const auto& producer = input.get_node_shared_ptr();
            if (ov::is_type_any_of<ov::op::v0::Cos,
                                   ov::op::v0::Cosh,
                                   ov::op::v0::Sin,
                                   ov::op::v0::Sinh,
                                   ov::op::v0::Acos,
                                   ov::op::v3::Acosh,
                                   ov::op::v0::Asin,
                                   ov::op::v3::Asinh,
                                   ov::op::v0::Atan,
                                   ov::op::v3::Atanh,
                                   ov::op::v0::Tan,
                                   ov::op::v0::Sign,
                                   ov::op::v4::SoftPlus,
                                   ov::op::v9::SoftSign,
                                   ov::op::v0::Selu,
                                   ov::op::v0::HardSigmoid>(producer)) {
                ov::disable_conversion(producer, ov::element::f16, ov::element::f32);
            }
        }
        return false;
    };

    auto m = std::make_shared<Matcher>(discontinuous_op, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass

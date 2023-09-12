// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/validate.hpp"

#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/convert_truncation.hpp"
#include "snippets/itt.hpp"

#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/core/validation_util.hpp"


namespace ov {
namespace snippets {
namespace pass {

namespace {
#define VALIDATE(validator, op)    \
    OPENVINO_ASSERT(validator(op), "Snippets validation of OV body has been failed: " + \
                                   std::string(op->get_type_name()) + " op " + op->get_friendly_name() + " is not supported");

bool is_support_constant(const std::shared_ptr<const ov::op::v0::Constant>& constant) {
    const auto consumers = constant->get_output_target_inputs(0);
    return ov::shape_size(constant->get_output_shape(0)) == 1 ||
           (consumers.size() == 1 &&
            (ov::is_type<const ov::op::v1::Transpose>(consumers.begin()->get_node()) ||
             ov::is_type<const ov::op::v1::Broadcast>(consumers.begin()->get_node()) ||
             ov::is_type<const ov::op::v3::Broadcast>(consumers.begin()->get_node())));
}

bool is_support_convert(const std::shared_ptr<const ov::op::v0::Convert>& convert) {
    return ov::is_type<op::ConvertTruncation>(convert) || ov::is_type<op::ConvertSaturation>(convert);
}

bool is_supported_matmul(const std::shared_ptr<const ov::op::v0::MatMul>& matmul) {
    return !matmul->get_transpose_a() && !matmul->get_transpose_b();
}

bool is_support_softmax(const std::shared_ptr<const ov::Node>& softmax) {
    const auto softmax_rank = softmax->get_input_partial_shape(0).rank();
    int64_t axis = 0;
    if (const auto softmax_v8 = ov::as_type_ptr<const ov::op::v8::Softmax>(softmax)) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        axis = ov::normalize_axis(softmax_v8->get_friendly_name(), softmax_v8->get_axis(), softmax_rank);
        OPENVINO_SUPPRESS_DEPRECATED_END
    } else if (const auto softmax_v1 = ov::as_type_ptr<const ov::op::v1::Softmax>(softmax)) {
        axis = softmax_v1->get_axis();
    } else {
        return false;
    }
    return axis == softmax_rank.get_length() - 1;
}

bool is_supported(const std::shared_ptr<const ov::Node>& node) {
    return false;
}
}  // namespace

bool Validate::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(Validate);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::Validate")

    for (const auto& op : m->get_ordered_ops()) {
        if (const auto constant = as_type_ptr<ov::op::v0::Constant>(op)) {
            VALIDATE(is_support_constant, constant);
        } else if (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(op)) {
            VALIDATE(is_support_convert, convert);
        } else if (const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(op)) {
            VALIDATE(is_supported_matmul, matmul);
        } else if (ov::is_type<ov::op::v8::Softmax>(op) ||
                   ov::is_type<ov::op::v1::Softmax>(op)) {
            VALIDATE(is_support_softmax, op);
        } else if (ov::is_type<ov::op::v0::FakeQuantize>(op) ||
                   ov::is_type<ov::op::v1::Reshape>(op)) {
            VALIDATE(is_supported, op);
        }
    }
    return true;
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov

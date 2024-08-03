// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/convert_truncation.hpp"
#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"
#include "snippets/pass/fq_decomposition.hpp"
#include "snippets/pass/validate.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace pass {

namespace {
#define VALIDATE(op, op_type, validator) \
    if (ov::is_type<op_type>(op)) \
        OPENVINO_ASSERT(validator(op), "Snippets validation of OV body has been failed: " + \
                        std::string(op->get_type_name()) + " op " + op->get_friendly_name() + " is not supported"); \
    else

} // namespace

bool Validate::is_supported_constant(const std::shared_ptr<const ov::Node>& op) {
    const auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(op);
    const auto consumers = op->get_output_target_inputs(0);
    return constant &&
           (ov::shape_size(constant->get_output_shape(0)) == 1 ||
            std::all_of(consumers.cbegin(), consumers.cend(),
                        [](const ov::Input<ov::Node>& in) {
                            return ov::is_type<const ov::op::v1::Transpose>(in.get_node()) ||
                                   ov::is_type<const ov::op::v1::Broadcast>(in.get_node()) ||
                                   ov::is_type<const ov::op::v3::Broadcast>(in.get_node());
                        }));
}

bool Validate::is_supported_convert(const std::shared_ptr<const ov::Node>& op) {
    return ov::is_type<const op::ConvertTruncation>(op) || ov::is_type<const op::ConvertSaturation>(op);
}

bool Validate::is_supported_matmul(const std::shared_ptr<const ov::Node>& op) {
    // If ExplicitTransposeMatMulInputs pass is enabled, MatMul should have not transposed inputs
    const auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(op);
    return matmul && utils::implication(m_pass_config->is_enabled<ov::snippets::pass::ExplicitTransposeMatMulInputs>(),
                                        !matmul->get_transpose_a() && !matmul->get_transpose_b());
}

bool Validate::is_supported_softmax(const std::shared_ptr<const ov::Node>& op) {
    // Softmax is supported only with axis by last dim
    const auto softmax_rank = op->get_input_partial_shape(0).rank();
    int64_t axis = 0;
    if (const auto softmax_v8 = ov::as_type_ptr<const ov::op::v8::Softmax>(op)) {
        axis = ov::util::try_normalize_axis(softmax_v8->get_axis(), softmax_rank, *softmax_v8);
    } else if (const auto softmax_v1 = ov::as_type_ptr<const ov::op::v1::Softmax>(op)) {
        axis = softmax_v1->get_axis();
    } else {
        return false;
    }
    return axis == softmax_rank.get_length() - 1;
}

bool Validate::is_supported_fq(const std::shared_ptr<const ov::Node>& node) {
    // FQ is decomposed into ops in CommonFakeQuantizeDecomposition pass
    return m_pass_config->is_disabled<ov::snippets::pass::CommonFakeQuantizeDecomposition>();
}

bool Validate::is_supported_transpose(const std::shared_ptr<const ov::Node>& node) {
    // Transpose is supported only on Inputs or Outputs of body
    const auto consumers = node->get_output_target_inputs(0);
    return (ov::is_type<ov::op::v0::Parameter>(node->get_input_node_shared_ptr(0))) ||
           (consumers.size() == 1 && ov::is_type<ov::op::v0::Result>(consumers.cbegin()->get_node()));
}

bool Validate::is_supported_op(const std::shared_ptr<const ov::Node>& node) {
    return false;
}

bool Validate::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(Validate);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::Validate")

    for (const auto& op : m->get_ordered_ops()) {
        VALIDATE(op, ov::op::v0::Constant, is_supported_constant)
        VALIDATE(op, ov::op::v0::Convert, is_supported_convert)
        VALIDATE(op, ov::op::v0::MatMul, is_supported_matmul)
        VALIDATE(op, ov::op::v1::Softmax, is_supported_softmax)
        VALIDATE(op, ov::op::v8::Softmax, is_supported_softmax)
        VALIDATE(op, ov::op::v0::FakeQuantize, is_supported_fq)
        VALIDATE(op, ov::op::v1::Transpose, is_supported_transpose)
        VALIDATE(op, ov::op::v1::Reshape, is_supported_op);
    }
    return true;
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov

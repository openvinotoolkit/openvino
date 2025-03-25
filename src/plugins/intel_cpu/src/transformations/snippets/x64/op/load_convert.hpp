// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/op/load.hpp"

namespace ov::intel_cpu {

/**
 * @interface LoadConvertSaturation
 * @brief Fused operation to represent computations equal to consecutive Load and ConvertSaturation operations.
 *        The operation is used for peephole optimization during subgraph lowering.
 * @ingroup snippets
 */
class LoadConvertSaturation : public snippets::op::Load {
public:
    OPENVINO_OP("LoadConvertSaturation", "SnippetsOpset", snippets::op::Load);

    LoadConvertSaturation(const Output<Node>& x,
                          const ov::element::Type& destination_type,
                          const size_t count = 1lu,
                          const size_t offset = 0lu);
    LoadConvertSaturation() = default;

    ov::element::Type get_destination_type() const {
        return m_destination_type;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override {
        return false;
    }

protected:
    ov::element::Type m_destination_type;
};

/**
 * @interface LoadConvertTruncation
 * @brief Fused operation to represent computations equal to consecutive Load and ConvertTruncation operations.
 *        The operation is used for peephole optimization during subgraph lowering.
 * @ingroup snippets
 */
class LoadConvertTruncation : public snippets::op::Load {
public:
    OPENVINO_OP("LoadConvertTruncation", "SnippetsOpset", snippets::op::Load);

    LoadConvertTruncation(const Output<Node>& x,
                          const ov::element::Type& destination_type,
                          const size_t count = 1lu,
                          const size_t offset = 0lu);
    LoadConvertTruncation() = default;

    ov::element::Type get_destination_type() const {
        return m_destination_type;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override {
        return false;
    }

protected:
    ov::element::Type m_destination_type;
};

}  // namespace ov::intel_cpu

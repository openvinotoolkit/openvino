// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "snippets/op/load.hpp"

#include "emitters/jit_load_store_emitters.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @interface LoadConvert
 * @brief Fused operation to represent computations equal to consecutive Load and Convert operations.
 *        The operation is used for peephole optimization during subgraph lowering.
 * @ingroup snippets
 */

class LoadConvert : public ngraph::snippets::op::Load {
public:
    OPENVINO_OP("LoadConvert", "SnippetsOpset", ngraph::snippets::op::Load);

    LoadConvert(const Output<Node>& x, const ov::element::Type& destination_type, arithmetic_mode mode, const size_t count = 1lu);
    LoadConvert() = default;

    ov::element::Type get_destination_type() const { return m_destination_type; }
    arithmetic_mode get_arithmetic_mode() const { return m_mode; }

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override { return false; }

protected:
    arithmetic_mode m_mode;
    ov::element::Type m_destination_type;
};

/**
 * @interface StoreConvert
 * @brief Fused operation to represent computations equal to consecutive Store and Convert operations.
 *        The operation is used for peephole optimization during subgraph lowering.
 * @ingroup snippets
 */
class StoreConvert : public ngraph::snippets::op::Store {
public:
    OPENVINO_OP("StoreConvert", "SnippetsOpset", ngraph::snippets::op::Store);

    StoreConvert(const Output<Node>& x, const ov::element::Type& destination_type, arithmetic_mode mode, const size_t count = 1lu);
    StoreConvert() = default;

    ov::element::Type get_destination_type() const { return m_destination_type; }
    arithmetic_mode get_arithmetic_mode() const { return m_mode; }

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool has_evaluate() const override { return false; }

protected:
    arithmetic_mode m_mode;
    ov::element::Type m_destination_type;
};


} // namespace intel_cpu
} // namespace ov

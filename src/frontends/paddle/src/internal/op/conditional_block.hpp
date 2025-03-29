// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
class ConditionalBlock : public Op {
public:
    OPENVINO_OP("ConditionalBlock", "internal");

    ConditionalBlock() = default;

    ConditionalBlock(const OutputVector& inputs,
                     const Output<Node>& cond,
                     bool is_scalar_condition,
                     int32_t sub_block_index,
                     const std::vector<std::pair<ov::element::Type, ov::PartialShape>>& output_infos);
    ConditionalBlock(const Output<Node>& cond,
                     bool is_scalar_condition,
                     int32_t sub_block_index,
                     const std::vector<std::pair<ov::element::Type, ov::PartialShape>>& output_infos);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return A vector containing the values for each input except "cond".
    const OutputVector get_inputs_from_parent() const;

    const int32_t get_subblock_index() const {
        return m_sub_block_index;
    }

private:
    bool m_is_scalar_condition;
    int32_t m_sub_block_index;
    std::vector<std::pair<ov::element::Type, ov::PartialShape>> m_output_infos;
};

}  // namespace internal
}  // namespace op
}  // namespace ov

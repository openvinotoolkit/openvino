// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
class While : public Op {
public:
    OPENVINO_OP("While", "internal");

    While() = default;

    While(const OutputVector& inputs,
          int32_t sub_block,
          const std::vector<std::pair<ov::element::Type, ov::PartialShape>>& output_infos);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const int32_t get_subblock_index() const {
        return m_sub_block;
    }

private:
    int32_t m_sub_block = 0;

    std::vector<std::pair<ov::element::Type, ov::PartialShape>> m_output_infos;
};

}  // namespace internal
}  // namespace op
}  // namespace ov

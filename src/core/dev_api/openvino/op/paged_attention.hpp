// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {

// This is an experimental operation that is implemented in the plugins.
// Do not use in user applications, backward compatibility is not guaranteed in future releases.
class OPENVINO_API PagedAttentionExtension : public ov::op::Op {
public:
    OPENVINO_OP("PagedAttentionExtension");

    PagedAttentionExtension() = default;

    PagedAttentionExtension(const ov::OutputVector& args, bool write_kv_cache = true);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void set_out_type(int index, const ov::element::Type& output_type);

    bool get_write_kv_cache() const {
        return m_write_kv_cache;
    }

protected:
    std::vector<ov::element::Type> m_output_type = {ov::element::dynamic, ov::element::dynamic, ov::element::dynamic};
    bool m_write_kv_cache = true;
};

}  // namespace op
}  // namespace ov

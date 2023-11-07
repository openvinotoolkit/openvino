// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v13 {
/// \ingroup ov_ops_cpp_api
class OPENVINO_API FakeConvert : public Op {
public:
    OPENVINO_OP("FakeConvert", "opset13");

    FakeConvert() = default;
    FakeConvert(const ov::Output<ov::Node>& arg,
                const ov::Output<ov::Node>& scale,
                const ov::Output<ov::Node>& shift,
                const std::string& destination_type = "F8E4M3",
                bool apply_scale = true);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool has_evaluate() const override;

    bool get_apply_scale() const {
        return m_apply_scale;
    }

    const std::string& destination_type() const {
        return m_destination_type;
    }

private:
    void validate() const;
    std::string m_destination_type = "F8E4M3";
    bool m_apply_scale = false;
    static const std::vector<std::string> m_valid_types;
};
}  // namespace v13
}  // namespace op
}  // namespace ov

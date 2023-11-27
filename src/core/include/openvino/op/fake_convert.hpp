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
                std::string destination_type = "f8e4m3");

    FakeConvert(const ov::Output<ov::Node>& arg,
                const ov::Output<ov::Node>& scale,
                const ov::Output<ov::Node>& shift,
                std::string destination_type = "f8e4m3");

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    const std::string& get_destination_type() const;

private:
    void validate_destination_type() const;

    std::string m_destination_type = "f8e4m3";
};
}  // namespace v13
}  // namespace op
}  // namespace ov

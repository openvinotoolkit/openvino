// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/grid_sample.hpp"

namespace ov {
namespace op {
namespace v10 {
class OPENVINO_API IsInf : public Op {
public:
    OPENVINO_OP("IsInf", "opset10");

    struct Attributes {
        bool detect_negative = true;
        bool detect_positive = true;

        Attributes() = default;
        Attributes(bool detect_negative, bool detect_positive)
            : detect_negative{detect_negative},
              detect_positive{detect_positive} {}
    };

    IsInf() = default;

    IsInf(const Output<Node>& data, const Attributes& attributes);
    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Attributes& get_attributes() const {
        return m_attributes;
    }

private:
    Attributes m_attributes = {};
};
}  // namespace v10
}  //namespace op
}  //namespace ov
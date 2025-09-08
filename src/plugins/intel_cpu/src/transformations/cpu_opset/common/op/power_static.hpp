// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_cpu {

class PowerStaticNode : public ov::op::Op {
public:
    OPENVINO_OP("PowerStatic", "cpu_plugin_opset");

    PowerStaticNode() = default;

    PowerStaticNode(const ov::Output<ov::Node>& data,
                    const float& power,
                    const float& scale,
                    const float& shift,
                    ov::element::Type output_type = ov::element::dynamic);

    void validate_and_infer_types() override;

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    float get_power() const {
        return power;
    }
    float get_scale() const {
        return scale;
    }
    float get_shift() const {
        return shift;
    }

private:
    float scale = 0.0F, power = 0.0F, shift = 0.0F;
    ov::element::Type m_output_type;
};

}  // namespace ov::intel_cpu

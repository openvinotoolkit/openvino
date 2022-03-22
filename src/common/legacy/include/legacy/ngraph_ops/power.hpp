// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class PowerIE : public Op {
public:
    OPENVINO_OP("PowerIE", "legacy");
    BWDCMP_RTTI_DECLARATION;
    PowerIE() = default;
    PowerIE(const Output<Node>& data_batch,
            const float power, const float scale, const float shift, const element::Type output_type = element::undefined);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    float scale, power, shift;

private:
    element::Type m_output_type;
};

}  // namespace op
}  // namespace ngraph

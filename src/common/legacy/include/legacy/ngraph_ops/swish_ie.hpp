// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
class SwishIE : public Op {
public:
    OPENVINO_OP("SwishIE", "legacy");
    BWDCMP_RTTI_DECLARATION;

    explicit SwishIE(const Output<Node> &input, float alpha = 1.0);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector &new_args) const override;

    void set_alpha(float alpha);
    float get_alpha() const;
protected:
    float m_alpha;
};
}  // namespace op
}  // namespace ngraph

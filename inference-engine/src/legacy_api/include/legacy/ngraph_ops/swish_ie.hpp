// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
class INFERENCE_ENGINE_API_CLASS(SwishIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"SwishIE", 1};
    const NodeTypeInfo &get_type_info() const override { return type_info; }

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

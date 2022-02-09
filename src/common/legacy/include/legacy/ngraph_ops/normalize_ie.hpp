// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class NormalizeIE : public Op {
public:
    OPENVINO_OP("NormalizeIE", "legacy");
    BWDCMP_RTTI_DECLARATION;

    NormalizeIE() = default;

    NormalizeIE(const Output<Node>& data,
                const Output<Node>& weights,
                float eps,
                bool across_spatial,
                bool channel_shared,
                const ngraph::element::Type output_type);

    float get_eps() const { return m_eps; }
    bool get_channel_shared() const  { return m_channel_shared;}
    bool get_across_spatial() const  { return m_across_spatial;}

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor &visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector &new_args) const override;

protected:
    float m_eps;
    bool m_across_spatial;
    bool m_channel_shared;
    ngraph::element::Type m_output_type;
};

}  // namespace op
}  // namespace ngraph

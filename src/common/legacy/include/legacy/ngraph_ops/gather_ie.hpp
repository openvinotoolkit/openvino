// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class GatherIE : public Op {
public:
    OPENVINO_OP("GatherIE", "legacy");
    BWDCMP_RTTI_DECLARATION;
    GatherIE() = default;

    GatherIE(const Output<Node>& params, const Output<Node>& indices, int64_t axis);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    int64_t get_axis() const { return m_axis; }
    void set_axis(int64_t axis) { m_axis = axis; }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

protected:
    int64_t m_axis;
};

}  // namespace op
}  // namespace ngraph

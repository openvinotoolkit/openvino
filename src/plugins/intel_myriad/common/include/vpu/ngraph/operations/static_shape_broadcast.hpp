// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/broadcast_base.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/broadcast.hpp"

#include <memory>
#include <vector>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeBroadcast : public ::ngraph::op::v3::Broadcast {
public:
    OPENVINO_OP("StaticShapeBroadcast", "VPUOpset");

    StaticShapeBroadcast(const Output<Node>& arg,
                         const Output<Node>& targetShape,
                         const Output<Node>& axesMapping,
                         const ngraph::op::BroadcastModeSpec& broadcastSpec = ngraph::op::BroadcastType::EXPLICIT);

    StaticShapeBroadcast(const Output<Node>& arg,
                         const Output<Node>& targetShape,
                         const ngraph::op::BroadcastModeSpec& broadcastSpec = ngraph::op::BroadcastType::NUMPY);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& newInputs) const override;

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END

protected:
    ngraph::PartialShape m_evaluatedOutputShape;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph

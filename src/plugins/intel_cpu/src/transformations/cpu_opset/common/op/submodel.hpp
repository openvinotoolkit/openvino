// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

namespace ov::intel_cpu {

/**
 * @interface Subgraph
 * @brief An operation that is implemented by a model
 */
class SubModel : public ov::op::util::SubGraphOp {
public:
    OPENVINO_OP("SubModel", "cpu_plugin_opset", ov::op::util::SubGraphOp);

    SubModel() = default;

    SubModel(const std::shared_ptr<ov::Model>& body);

    SubModel(const OutputVector& args, const std::shared_ptr<ov::Model>& body);

    SubModel(const NodeVector& args, const std::shared_ptr<ov::Model>& body);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    const ov::Model& body() const {
        return *m_bodies[0];
    }
    const std::shared_ptr<ov::Model>& body_ptr() const {
        return m_bodies[0];
    }

private:
    ov::Model& body() {
        return *m_bodies[0];
    }
    std::shared_ptr<ov::Model>& body_ptr() {
        return m_bodies[0];
    }
};

}  // namespace ov::intel_cpu

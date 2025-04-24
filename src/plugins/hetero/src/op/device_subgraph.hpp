// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

namespace ov {
namespace hetero {
namespace op {

/**
 * @interface DeviceSubgraph
 * @brief An operation that is implemented by a model
 */
class DeviceSubgraph : public ov::op::util::SubGraphOp {
public:
    OPENVINO_OP("DeviceSubgraph", "hetero", ov::op::util::SubGraphOp);

    DeviceSubgraph() = default;

    DeviceSubgraph(const OutputVector& args, const std::shared_ptr<ov::Model>& body, const std::string& affinity);

    DeviceSubgraph(const NodeVector& args, const std::shared_ptr<ov::Model>& body, const std::string& affinity);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    std::string get_affinity() {
        return _affinity;
    }

private:
    const ov::Model& body() const {
        return *m_bodies[0];
    }
    ov::Model& body() {
        return *m_bodies[0];
    }
    const std::shared_ptr<ov::Model>& body_ptr() const {
        return m_bodies[0];
    }
    std::shared_ptr<ov::Model>& body_ptr() {
        return m_bodies[0];
    }

    std::string _affinity;
};
using DeviceSubgraphVector = std::vector<std::shared_ptr<ov::hetero::op::DeviceSubgraph>>;

}  // namespace op
}  // namespace hetero
}  // namespace ov

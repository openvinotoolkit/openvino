// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GatedDeltaNet : public ov::op::Op {
public:
    OPENVINO_OP("GatedDeltaNet");

    GatedDeltaNet() = default;
    GatedDeltaNet(const Output<Node>& query,
                  const Output<Node>& key,
                  const Output<Node>& value,
                  const Output<Node>& recurrent_state,
                  const Output<Node>& gate,
                  const Output<Node>& beta);
    struct Config {
        bool fuse_qk_l2norm = false;
        float q_l2_norm_eps = 1e-6F;
        float k_l2_norm_eps = 1e-6F;
    };
    GatedDeltaNet(const ov::OutputVector& args);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    const Config& get_config() const {
        return m_config;
    }
    void set_config(const Config& config) {
        m_config = config;
    }

protected:
    Config m_config;
};

}  // namespace ov::op::internal

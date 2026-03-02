// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {

// This is an experimental operation that is implemented in the plugins.
// Do not use in user applications, backward compatibility is not guaranteed in future releases.
class OPENVINO_API GatedDeltaNet : public ov::op::Op {
public:
    OPENVINO_OP("GatedDeltaNet");

    GatedDeltaNet() = default;
    struct Config {
        bool fuse_qk_l2norm = false;
        bool fuse_q_scale = false;
    };
    GatedDeltaNet(const ov::OutputVector& args);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    const Config& get_config() const {
        return m_config;
    }
    void set_config(const Config& config) {
        m_config = config;
    }
    void set_out_type(int index, const ov::element::Type& output_type);

protected:
    std::vector<ov::element::Type> m_output_type = {ov::element::dynamic, ov::element::dynamic, ov::element::dynamic};
    Config m_config;
};

}  // namespace op
}  // namespace ov

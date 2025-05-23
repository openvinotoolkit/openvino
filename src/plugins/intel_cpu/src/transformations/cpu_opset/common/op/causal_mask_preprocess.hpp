// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::intel_cpu {

class CausalMaskPreprocessNode : public ov::op::Op {
public:
    OPENVINO_OP("CausalMaskPreprocess", "cpu_plugin_opset");

    CausalMaskPreprocessNode() = default;

    struct Config {
        std::string type;
    };

    CausalMaskPreprocessNode(const OutputVector& args, Config cfg);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }

    Config& get_config() {
        return m_config;
    }

private:
    Config m_config;
};

}  // namespace ov::intel_cpu

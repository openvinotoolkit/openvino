// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/framework_node.hpp"
#include "quant/gguf.hpp"

namespace ov::frontend::gguf {

class GGUFConstant : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("GGUFConstant", "gguf", ov::op::util::FrameworkNode);
    explicit GGUFConstant(const GgufWeight& source);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override;
    std::shared_ptr<ov::Node> materialize() const;
    GgufTensorType qtype() const;
    const GgufWeight& source() const;

private:
    GgufWeight m_source;
};

}  // namespace ov::frontend::gguf

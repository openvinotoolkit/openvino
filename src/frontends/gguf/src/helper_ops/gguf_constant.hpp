// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "openvino/op/op.hpp"
#include "quant/weights.hpp"

namespace ov::frontend::gguf {

class GGUFConstant : public ov::op::Op {
public:
    OPENVINO_OP("GGUFConstant", "gguf");

    GGUFConstant(const WeightTensors& tensors, GgufTensorType qtype, const std::string& source_name);
    GGUFConstant(const ov::Shape& shape,
                 const ov::element::Type& type,
                 GgufTensorType qtype,
                 const std::string& source_name,
                 std::function<std::shared_ptr<ov::Node>()> materialize);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override;
    std::shared_ptr<ov::Node> materialize() const;
    bool is_u2() const;
    bool can_concat(const GGUFConstant& other) const;
    const WeightTensors& tensors() const;
    GgufTensorType qtype() const;
    const std::string& source_name() const;

private:
    WeightTensors m_tensors;
    GgufTensorType m_qtype;
    std::string m_source_name;
    ov::Shape m_shape;
    ov::element::Type m_type;
    std::function<std::shared_ptr<ov::Node>()> m_materialize;
};

}  // namespace ov::frontend::gguf

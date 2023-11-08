// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "modifiers.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"

#include "libxsmm_typedefs.h"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {
using AutoBroadcastSpec = ov::op::AutoBroadcastSpec;
using AutoBroadcastType = ov::op::AutoBroadcastType;
class BinaryEltwiseTPP : public TensorProcessingPrimitive {
public:
    static bool is_supported(const std::shared_ptr<ov::Node>& node);
    virtual libxsmm_meltw_binary_type get_op_type() const = 0;
};

class Add : public BinaryEltwiseTPP, public ov::op::v1::Add {
public:
    OPENVINO_OP("Add", "TppOpset", ov::op::v1::Add);
    Add(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    libxsmm_meltw_binary_type get_op_type() const override {
        return libxsmm_meltw_binary_type::LIBXSMM_MELTW_TYPE_BINARY_ADD;
    }
};

class Subtract : public BinaryEltwiseTPP, public ov::op::v1::Subtract {
public:
    OPENVINO_OP("Subtract", "TppOpset", ov::op::v1::Subtract);
    Subtract(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    libxsmm_meltw_binary_type get_op_type() const override {
        return libxsmm_meltw_binary_type::LIBXSMM_MELTW_TYPE_BINARY_SUB;
    }
};

class Multiply : public BinaryEltwiseTPP, public ov::op::v1::Multiply {
public:
    OPENVINO_OP("Multiply", "TppOpset", ov::op::v1::Multiply);
    Multiply(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    libxsmm_meltw_binary_type get_op_type() const override {
        return libxsmm_meltw_binary_type::LIBXSMM_MELTW_TYPE_BINARY_MUL;
    }
};

class Divide : public BinaryEltwiseTPP, public ov::op::v1::Divide {
public:
    OPENVINO_OP("Divide", "TppOpset", ov::op::v1::Divide);
    Divide(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    libxsmm_meltw_binary_type get_op_type() const override {
        return libxsmm_meltw_binary_type::LIBXSMM_MELTW_TYPE_BINARY_DIV;
    }
};

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov

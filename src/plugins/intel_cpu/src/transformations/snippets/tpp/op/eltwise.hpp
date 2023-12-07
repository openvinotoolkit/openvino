// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "modifiers.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include <variant>

#include "libxsmm_typedefs.h"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {
using AutoBroadcastSpec = ov::op::AutoBroadcastSpec;
using AutoBroadcastType = ov::op::AutoBroadcastType;
class EltwiseTPP : public modifier::TensorProcessingPrimitive {
public:
    class libxsmm_op_type {
    public:
        libxsmm_op_type(libxsmm_meltw_binary_type type) : 
            op_type(LIBXSMM_MELTW_OPERATION_BINARY), binary_type(type) {
        }
        libxsmm_op_type(libxsmm_meltw_unary_type type) : 
            op_type(LIBXSMM_MELTW_OPERATION_UNARY), unary_type(type) {
        }
        libxsmm_meltw_operation get_op_type() {
            return op_type;
        }
        libxsmm_meltw_binary_type get_binary_type() {
            OPENVINO_ASSERT(op_type == LIBXSMM_MELTW_OPERATION_BINARY);
            return binary_type;
        }
        libxsmm_meltw_unary_type get_unnary_type() {
            OPENVINO_ASSERT(op_type == LIBXSMM_MELTW_OPERATION_UNARY);
            return unary_type;
        }
    
    private:
        libxsmm_meltw_operation op_type;
        union {
            libxsmm_meltw_binary_type binary_type;
            libxsmm_meltw_unary_type unary_type;
        };
    };
    EltwiseTPP(libxsmm_op_type op_type);
    static bool is_supported(const std::shared_ptr<ov::Node>& node);

    const libxsmm_op_type& get_libxsmm_op_type() const { return op_type; }
    bool visit_attributes(AttributeVisitor& visitor);

protected:
    libxsmm_op_type op_type;
};

class Add : public EltwiseTPP, public ov::op::v1::Add {
public:
    OPENVINO_OP("Add", "TppOpset", ov::op::v1::Add);
    Add(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

class Subtract : public EltwiseTPP, public ov::op::v1::Subtract {
public:
    OPENVINO_OP("Subtract", "TppOpset", ov::op::v1::Subtract);
    Subtract(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

class Multiply : public EltwiseTPP, public ov::op::v1::Multiply {
public:
    OPENVINO_OP("Multiply", "TppOpset", ov::op::v1::Multiply);
    Multiply(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

class Divide : public EltwiseTPP, public ov::op::v1::Divide {
public:
    OPENVINO_OP("Divide", "TppOpset", ov::op::v1::Divide);
    Divide(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

// class Divide : public EltwiseTPP, public ov::op::v0::Exp {
// public:
//     OPENVINO_OP("Divide", "TppOpset", ov::op::v1::Divide);
//     Divide(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
//     std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
//     bool visit_attributes(AttributeVisitor& visitor) override;
//     libxsmm_meltw_binary_type get_op_type() const override {
//         return libxsmm_meltw_binary_type::LIBXSMM_MELTW_TYPE_UNARY_EXP;
//     }
// };

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov

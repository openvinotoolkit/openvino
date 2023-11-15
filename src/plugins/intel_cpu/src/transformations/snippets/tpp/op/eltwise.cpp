// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "eltwise.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

BinaryEltwiseTPP::BinaryEltwiseTPP() {
    // Initialize input/output ports as memory access ports
    ctor_initialize(std::set<size_t>{0, 1}, std::set<size_t>{0});
}

bool BinaryEltwiseTPP::is_supported(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v1::Add>(node) ||
           ov::is_type<ov::op::v1::Add>(node);
}

Add::Add(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
: BinaryEltwiseTPP(), ov::op::v1::Add(arg0, arg1, auto_broadcast) {
}

bool Add::visit_attributes(AttributeVisitor& visitor) {
    // todo: this is for debug purposes. remove before merge
    std::string tmp = "TPP";
    visitor.on_attribute("type", tmp);
    return MemoryAccess::visit_attributes(visitor);
}

std::shared_ptr<Node> Add::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Add>(new_args.at(0), new_args.at(1), this->get_autob());
}

Subtract::Subtract(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
        : BinaryEltwiseTPP(), ov::op::v1::Subtract(arg0, arg1, auto_broadcast) {
}

std::shared_ptr<Node> Subtract::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Subtract>(new_args.at(0), new_args.at(1), this->get_autob());
}

Multiply::Multiply(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
        : BinaryEltwiseTPP(), ov::op::v1::Multiply(arg0, arg1, auto_broadcast) {
}

std::shared_ptr<Node> Multiply::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Multiply>(new_args.at(0), new_args.at(1), this->get_autob());
}

Divide::Divide(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
        : BinaryEltwiseTPP(), ov::op::v1::Divide(arg0, arg1, auto_broadcast) {
}

std::shared_ptr<Node> Divide::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Divide>(new_args.at(0), new_args.at(1), this->get_autob());
}

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov

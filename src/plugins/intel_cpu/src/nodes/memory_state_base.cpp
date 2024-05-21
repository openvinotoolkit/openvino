// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_state_base.h"
#include <openvino/core/type.hpp>
#include <openvino/op/util/assign_base.hpp>
#include <openvino/op/util/read_value_base.hpp>

using namespace ov::intel_cpu::node;

MemoryNode::MemoryNode(const std::shared_ptr<ov::Node>& op) {
    if (auto assignOp = ov::as_type_ptr<ov::op::util::AssignBase>(op)) {
        m_id = assignOp->get_variable_id();
    } else if (auto readValueOp = ov::as_type_ptr<ov::op::util::ReadValueBase>(op)) {
        m_id = readValueOp->get_variable_id();
    } else {
        OPENVINO_THROW("Unexpected ov::Node type: ", op->get_type_info().name, " in MemoryNode");
    }
}
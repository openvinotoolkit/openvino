// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/kernel.hpp"

#include <memory>
#include <utility>

#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::op {

Kernel::Kernel(lowered::LinearIR nested) : region(std::make_shared<lowered::LinearIR>(std::move(nested))) {}

KernelStatic::KernelStatic(lowered::LinearIR nested) : Kernel(std::move(nested)) {}

KernelDynamic::KernelDynamic(lowered::LinearIR nested) : Kernel(std::move(nested)) {}

std::shared_ptr<Node> KernelStatic::clone_with_new_inputs(const OutputVector& /*inputs*/) const {
    return std::make_shared<KernelStatic>(*region);
}

std::shared_ptr<Node> KernelDynamic::clone_with_new_inputs(const OutputVector& /*inputs*/) const {
    return std::make_shared<KernelDynamic>(*region);
}

}  // namespace ov::snippets::op

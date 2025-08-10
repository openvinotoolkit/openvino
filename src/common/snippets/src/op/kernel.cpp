// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/kernel.hpp"

#include <memory>
#include <utility>

#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::op {

Kernel::Kernel(lowered::LinearIR region) : region(std::make_shared<lowered::LinearIR>(std::move(region))) {}

KernelStatic::KernelStatic(lowered::LinearIR region) : Kernel(std::move(region)) {}

KernelDynamic::KernelDynamic(lowered::LinearIR region) : Kernel(std::move(region)) {}

std::shared_ptr<Node> KernelStatic::clone_with_new_inputs([[maybe_unused]] const OutputVector& inputs) const {
    return std::make_shared<KernelStatic>(*region);
}

std::shared_ptr<Node> KernelDynamic::clone_with_new_inputs([[maybe_unused]] const OutputVector& inputs) const {
    return std::make_shared<KernelDynamic>(*region);
}

}  // namespace ov::snippets::op

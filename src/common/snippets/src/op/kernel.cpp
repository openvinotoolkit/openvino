// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/kernel.hpp"

#include "snippets/op/loop.hpp"

namespace ov {
namespace snippets {
namespace op {

Kernel::Kernel(lowered::LinearIR nested) : Op(), region(std::make_shared<lowered::LinearIR>(std::move(nested))) {}

KernelStatic::KernelStatic(lowered::LinearIR nested) : Kernel(std::move(nested)) {}

KernelDynamic::KernelDynamic(lowered::LinearIR nested) : Kernel(std::move(nested)) {}

std::shared_ptr<Node> KernelStatic::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<KernelStatic>(*region.get());
}

std::shared_ptr<Node> KernelDynamic::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<KernelDynamic>(*region.get());
}

} // namespace op
} // namespace snippets
} // namespace ov

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/kernel.hpp"

namespace ov {
namespace snippets {
namespace op {

Kernel::Kernel(lowered::LinearIR nested) : Op(), region(std::move(nested)) {}

KernelStatic::KernelStatic(lowered::LinearIR nested) : Kernel(std::move(nested)) {}

KernelDynamic::KernelDynamic(lowered::LinearIR nested) : Kernel(std::move(nested)) {}

std::shared_ptr<Node> KernelStatic::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<KernelStatic>(region);
}

std::shared_ptr<Node> KernelDynamic::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<KernelDynamic>(region);
}

} // namespace op
} // namespace snippets
} // namespace ov

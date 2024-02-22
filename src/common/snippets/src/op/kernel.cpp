// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/kernel.hpp"

#include "snippets/op/loop.hpp"

namespace ov {
namespace snippets {
namespace op {

Kernel::Kernel(lowered::LinearIR nested) : Op(), region(std::move(nested)) {}

std::shared_ptr<Kernel> Kernel::make_kernel(const lowered::LinearIR& region) {
    // TODO: In general the decision should be based on LinearIR.is_dynamic().
    //       Need to add this condition when full dynamic pipeline support is completed
    auto is_dynamic_loop = [](const lowered::ExpressionPtr& expr) {
        return ov::is_type<op::LoopBeginDynamic>(expr->get_node()) || ov::is_type<op::LoopEndDynamic>(expr->get_node());
    };

    if (std::any_of(region.cbegin(), region.cend(), is_dynamic_loop)) {
        return std::make_shared<KernelDynamic>(region);
    } else {
        return std::make_shared<KernelStatic>(region);
    }
}

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

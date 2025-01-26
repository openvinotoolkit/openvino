// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"

#include "intel_gpu/op/placeholder.hpp"

namespace ov {
namespace op {
namespace internal {
using Placeholder = ov::intel_gpu::op::Placeholder;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreatePlaceholderOp(ProgramBuilder&, const std::shared_ptr<ov::intel_gpu::op::Placeholder>&) { }

REGISTER_FACTORY_IMPL(internal, Placeholder);

}  // namespace ov::intel_gpu

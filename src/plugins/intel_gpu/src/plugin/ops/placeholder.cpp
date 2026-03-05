// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"

#include "intel_gpu/op/placeholder.hpp"
#include "openvino/op/placeholder_extension.hpp"

namespace ov {
namespace op {
namespace internal {
using Placeholder = ov::intel_gpu::op::Placeholder;
}  // namespace internal
}  // namespace op
}  // namespace ov

using PlaceholderExtension = ov::op::internal::PlaceholderExtension;

namespace ov::intel_gpu {

static void CreatePlaceholderOp(ProgramBuilder&, const std::shared_ptr<ov::intel_gpu::op::Placeholder>&) { }

static void CreatePlaceholderExtensionOp(ProgramBuilder&, const std::shared_ptr<ov::op::internal::PlaceholderExtension>&) { }

REGISTER_FACTORY_IMPL(internal, Placeholder);
REGISTER_FACTORY_IMPL(internal, PlaceholderExtension);

}  // namespace ov::intel_gpu

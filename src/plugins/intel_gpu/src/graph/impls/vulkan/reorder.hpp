// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "registry/implementation_manager.hpp"
#include "reorder_inst.h"

namespace cldnn {
namespace vulkan {

struct ReorderImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("vulkan::reorder")

    explicit ReorderImplementationManager(shape_types shape_type, ValidateFunc validate = nullptr)
        : ImplementationManager(impl_types::vulkan, shape_type, std::move(validate)) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    bool validate_impl(const program_node& node) const override;
    in_out_fmts_t query_formats(const program_node&) const override {
        return {};
    }
};

}  // namespace vulkan
}  // namespace cldnn

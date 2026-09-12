// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "eltwise_inst.h"
#include "registry/implementation_manager.hpp"

namespace cldnn {
namespace vulkan {

struct EltwiseImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("vulkan::eltwise")

    explicit EltwiseImplementationManager(shape_types shape_type, ValidateFunc validate = nullptr)
        : ImplementationManager(impl_types::vulkan, shape_type, std::move(validate)) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    bool validate_impl(const program_node& node) const override;
    in_out_fmts_t query_formats(const program_node&) const override {
        return {};
    }
};

}  // namespace vulkan
}  // namespace cldnn

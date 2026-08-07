// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "program_node.h"
#include "registry/implementation_manager.hpp"

namespace cldnn::common {

struct MLIRPrimitiveImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("common::mlir_primitive")
    MLIRPrimitiveImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::common, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                const kernel_impl_params& params) const override;
};

}  // namespace cldnn::common

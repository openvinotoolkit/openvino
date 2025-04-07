// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry/implementation_manager.hpp"
#include "program_node.h"

#include <memory>
namespace cldnn {
namespace ocl {

struct DetectionOutputImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::detection_output")
    DetectionOutputImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
};

}  // namespace ocl
}  // namespace cldnn

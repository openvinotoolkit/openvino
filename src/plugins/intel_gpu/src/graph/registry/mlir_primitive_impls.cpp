// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/mlir_primitive.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_COMMON
#    include "impls/common/mlir_primitive.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<mlir_primitive>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_COMMON(common::MLIRPrimitiveImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_COMMON(common::MLIRPrimitiveImplementationManager, shape_types::dynamic_shape)
    };
    return impls;
}

}  // namespace ov::intel_gpu

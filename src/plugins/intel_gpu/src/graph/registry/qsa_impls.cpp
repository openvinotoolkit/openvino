// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/qsa.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_CM
#    include "impls/cm/qsa/qsa.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<qsa>::get_implementations() {
    // QSA has no OpenCL / oneDNN implementation by design: the dedicated path is CM-only
    // and never falls back to paged_attention or scaled_dot_product_attention.
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
#if OV_GPU_WITH_CM
        OV_GPU_CREATE_INSTANCE_CM(cm::QSACMImplementationManager, shape_types::static_shape)
#endif
    };

    return impls;
}

}  // namespace ov::intel_gpu

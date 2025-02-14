// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "registry.hpp"
#include "intel_gpu/primitives/rnn.hpp"

#if OV_GPU_WITH_OCL
    #include "impls/ocl/lstm_cell.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<lstm_cell>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::LSTMCellImplementationManager, shape_types::static_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu

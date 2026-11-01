// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental/primitives/block_sparse_attention.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_CM
#    include "experimental/graph/impls/cm/block_sparse_attention/block_sparse_attention.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<block_sparse_attention>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_CM(cm::BlockSparseAttentionImplementationManager, shape_types::any)};

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov

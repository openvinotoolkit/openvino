// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental/primitives/draft_block_mask.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_CM
#    include "experimental/graph/impls/cm/draft_block_mask/draft_block_mask.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<draft_block_mask>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_CM(cm::DraftBlockMaskImplementationManager, shape_types::any)};

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov

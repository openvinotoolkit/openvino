// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_depth.hpp"
#include <common/primitive_hashing_utils.hpp>

namespace ov {
namespace intel_cpu {

size_t SpaceToDepthAttrs::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, layoutType);
    seed = hash_combine(seed, mode);
    seed = hash_combine(seed, blockSize);
    seed = hash_combine(seed, blockStep);
    seed = hash_combine(seed, dataSize);
    seed = hash_combine(seed, nSpatialDims);
    seed = get_vector_hash(seed, srcBlockedDims);
    seed = get_vector_hash(seed, destBlockedDims);

    return seed;
}

bool SpaceToDepthAttrs::operator==(const SpaceToDepthAttrs& rhs) const {
    bool result = layoutType == rhs.layoutType && mode == rhs.mode &&
                  blockSize == rhs.blockSize && blockStep == rhs.blockStep &&
                  dataSize == rhs.dataSize && nSpatialDims == rhs.nSpatialDims &&
                  srcBlockedDims == rhs.srcBlockedDims && destBlockedDims == rhs.destBlockedDims;

    return result;
}

SpaceToDepthExecutor::SpaceToDepthExecutor(const ExecutorContext::CPtr context) : spaceToDepthContext(context) {}

}   // namespace intel_cpu
}   // namespace ov
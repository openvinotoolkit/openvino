// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shuffle_channels.hpp"
#include <common/primitive_hashing_utils.hpp>

namespace ov {
namespace intel_cpu {

size_t ShuffleChannelsAttributes::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, layoutType);
    seed = hash_combine(seed, dataRank);
    seed = hash_combine(seed, axis);
    seed = hash_combine(seed, spatialRank);
    seed = hash_combine(seed, group);
    seed = hash_combine(seed, dataSize);
    seed = get_vector_hash(seed, srcDims);
    seed = get_vector_hash(seed, srcBlockedDims);

    return seed;
}

bool ShuffleChannelsAttributes::operator==(const ShuffleChannelsAttributes& rhs) const {
    bool result = layoutType == rhs.layoutType && dataRank == rhs.dataRank &&
                  axis == rhs.axis && spatialRank == rhs.spatialRank &&
                  group == rhs.group && dataSize == rhs.dataSize && srcDims == rhs.srcDims &&
                  srcBlockedDims == rhs.srcBlockedDims;
    return result;
}

ShuffleChannelsExecutor::ShuffleChannelsExecutor(const ExecutorContext::CPtr context) : shuffleChannelsContext(context) {}

}   // namespace intel_cpu
}   // namespace ov
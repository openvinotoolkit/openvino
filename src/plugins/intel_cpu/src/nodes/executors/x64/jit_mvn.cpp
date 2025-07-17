// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_mvn.hpp"

#include <any>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>
#include <vector>

#include "common/primitive_attr.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mvn_config.hpp"
#include "nodes/kernels/x64/mlp_utils.hpp"

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov::intel_cpu {

namespace {

struct MVNKey {
    MVNAttrs mvnAttrs;
    dnnl::primitive_attr attr;

    [[nodiscard]] size_t hash() const;
    bool operator==(const MVNKey& rhs) const;
};

size_t MVNKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, mvnAttrs.initAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.execAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.normalizeVariance_);
    seed = hash_combine(seed, mvnAttrs.epsValue_);
    seed = hash_combine(seed, mvnAttrs.epsMode_);
    seed = hash_combine(seed, mvnAttrs.src_prc.hash());
    seed = hash_combine(seed, mvnAttrs.dst_prc.hash());
    seed = hash_combine(seed, mvnAttrs.layout);
    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    return seed;
}

bool MVNKey::operator==(const MVNKey& rhs) const {
    bool retVal = true;
    retVal = retVal && mvnAttrs.initAcrossChannels_ == rhs.mvnAttrs.initAcrossChannels_ &&
             mvnAttrs.execAcrossChannels_ == rhs.mvnAttrs.execAcrossChannels_ &&
             mvnAttrs.normalizeVariance_ == rhs.mvnAttrs.normalizeVariance_ &&
             mvnAttrs.epsValue_ == rhs.mvnAttrs.epsValue_ && mvnAttrs.epsMode_ == rhs.mvnAttrs.epsMode_ &&
             mvnAttrs.src_prc == rhs.mvnAttrs.src_prc && mvnAttrs.dst_prc == rhs.mvnAttrs.dst_prc &&
             mvnAttrs.layout == rhs.mvnAttrs.layout;
    retVal = retVal && *attr.get() == *rhs.attr.get();
    return retVal;
}

}  // namespace

MVNJitExecutor::MVNJitExecutor(const MVNAttrs& mvnAttrs, MemoryArgs memory, ExecutorContext::CPtr contextPtr)
    : attrs(mvnAttrs),
      memoryArgs(std::move(memory)),
      context(std::move(contextPtr)),
      shape5D(mvnAttrs.shape5D) {
    // Extract post-ops data from MVNAttrs
    postOpsDataPtrs.clear();

    // MVNAttrs.postOps contains the post-ops data pointers
    // Each element is a std::any containing a const void* pointer
    for (const auto& postOp : attrs.postOps) {
        // Extract the data pointer from std::any
        if (postOp.type() == typeid(const void*)) {
            postOpsDataPtrs.push_back(std::any_cast<const void*>(postOp));
        }
    }

    // Create primitive attribute with default values
    dnnl::primitive_attr attr;

    // Create a key for caching
    MVNKey key{attrs, attr};

    auto builder = [&](const MVNKey& key) -> std::shared_ptr<legacy::MVNJitExecutorLagacy> {
        return std::make_shared<legacy::MVNJitExecutorLagacy>(key.mvnAttrs, key.attr);
    };

    // Use context's cache if available
    if (context) {
        auto cache = context->getRuntimeCache();
        auto result = cache->getOrCreate(key, builder);
        legacyJitExecutor = result.first;
    } else {
        // Fallback if no context available
        legacyJitExecutor = builder(key);
    }
}

void MVNJitExecutor::executeImpl(const MemoryArgs& memory) {
    // Extract memory pointers from MemoryArgs
    const auto* src_data = memory.at(ARG_SRC)->getDataAs<const uint8_t>();
    auto* dst_data = memory.at(ARG_DST)->getDataAs<uint8_t>();

    // Pass post-ops data to the legacy executor
    const void* post_ops_data = nullptr;
    if (!postOpsDataPtrs.empty()) {
        post_ops_data = static_cast<const void*>(static_cast<const void* const*>(postOpsDataPtrs.data()));
    }

    // Call legacy executor with proper parameters
    legacyJitExecutor->exec(src_data, dst_data, post_ops_data, shape5D);
}

bool MVNJitExecutor::canReuseShapeAgnosticKernel(const VectorDims& newShape5D) const {
    // Shape-agnostic kernel optimization
    // Reuses kernel if the shape is the same or only batch size changed
    if (shape5D[0] != newShape5D[0]) {
        if (shape5D[1] == newShape5D[1] && shape5D[2] == newShape5D[2] && shape5D[3] == newShape5D[3] &&
            shape5D[4] == newShape5D[4]) {
            shape5D = newShape5D;
            return true;
        }
    }
    return false;
}

bool MVNJitExecutor::supports(const MVNConfig& /*config*/) {
    // JIT implementation supports all precisions
    // The legacy implementation handles precision conversions internally
    return true;
}

}  // namespace ov::intel_cpu

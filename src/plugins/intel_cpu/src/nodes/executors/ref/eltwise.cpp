// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"

#include <algorithm>
#include <cassert>
#include <common/float16.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu_types.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/eltwise_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/jit/eltwise.h"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

size_t EltwiseRefKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;
    size_t seed = 0;
    auto hash_combine_eltwiseData = [](size_t seed, const EltwiseData& eltwiseData) {
        seed = hash_combine(seed, eltwiseData.algo);
        seed = hash_combine(seed, eltwiseData.onednnAlgorithm);
        seed = hash_combine(seed, eltwiseData.alpha);
        seed = hash_combine(seed, eltwiseData.beta);
        seed = hash_combine(seed, eltwiseData.gamma);
        return seed;
    };
    std::for_each(eltwise_data.begin(), eltwise_data.end(), [&](const EltwiseData& item) {
        seed = hash_combine_eltwiseData(seed, item);
    });

    seed = get_vector_hash(seed, outBlkDims);
    seed = hash_combine(seed, outPrc.hash());
    for (auto&& item : inpDims) {
        seed = get_vector_hash(seed, item);
    }
    return seed;
}

bool EltwiseRefKey::operator==(const EltwiseRefKey& rhs) const {
    if (inpDims.size() != rhs.inpDims.size()) {
        return false;
    }

    bool result = eltwise_data == rhs.eltwise_data && outPrc == rhs.outPrc;

    if (result) {
        result = result && outBlkDims == rhs.outBlkDims;
        for (size_t i = 0; i < inpDims.size() && result; ++i) {
            result = result && (inpDims[i] == rhs.inpDims[i]);
        }
    }

    return result;
}

static EltwiseExecutorPtr createRefExecutorByPrecision(const EltwiseRefKey& key) {
    switch (key.outPrc) {
    case ov::element::i8:
        return std::make_shared<BitwiseRefExecutor<int8_t>>(key);
    case ov::element::u8:
        return std::make_shared<BitwiseRefExecutor<uint8_t>>(key);
    case ov::element::i16:
        return std::make_shared<BitwiseRefExecutor<int16_t>>(key);
    case ov::element::u16:
        return std::make_shared<BitwiseRefExecutor<uint16_t>>(key);
    case ov::element::i32:
        return std::make_shared<BitwiseRefExecutor<int32_t>>(key);
    case ov::element::f16:
        return std::make_shared<EltwiseRefExecutor<dnnl::impl::float16_t>>(key);
    default:
        // Use float reference executor for any other precision
        return std::make_shared<EltwiseRefExecutor<float>>(key);
    }
}

EltwiseExecutorPtr create(const std::vector<VectorDims>& inDims,
                          const VectorDims& outBlkDims,
                          const ov::element::Type& outPrc,
                          const ExecutorContext::CPtr& context,
                          const EltwiseShapeAgnosticData& shapeAgnosticData) {
    EltwiseRefKey key = {inDims, outBlkDims, outPrc, shapeAgnosticData.eltwise_data};

    auto builder = [&](const EltwiseRefKey& key) {
        return createRefExecutorByPrecision(key);
    };

    auto runtimeCache = context->getRuntimeCache();
    const auto result = runtimeCache->getOrCreate(key, builder);
    const auto& executor = result.first;
    assert(executor);

    return executor;
}

}  // namespace ov::intel_cpu

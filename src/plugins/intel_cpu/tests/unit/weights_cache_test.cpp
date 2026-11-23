// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_cache.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>

#include "config.h"
#include "cpu_memory.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/reorder.h"

namespace ov::intel_cpu {
namespace {

constexpr size_t O = 3;
constexpr size_t I = 3;
constexpr size_t H = 3;
constexpr size_t W = 3;
constexpr size_t spatialSize = H * W;
constexpr size_t elementCount = O * I * spatialSize;

TEST(WeightsCacheTest, KeyIncludesSourceDestinationAndBuffer) {
    const dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    const Shape shape{O, I, H, W};
    using DataType = dnnl::memory::data_type;
    using Format = dnnl::memory::format_tag;
    auto source = std::make_shared<DnnlBlockedMemoryDesc>(shape, DataType::f32, Format::abcd);
    auto equivalentSource = std::make_shared<DnnlBlockedMemoryDesc>(shape, DataType::f32, Format::abcd);
    auto transposedSource = std::make_shared<DnnlBlockedMemoryDesc>(shape, DataType::f32, Format::bacd);
    auto destination = std::make_shared<DnnlBlockedMemoryDesc>(shape, DataType::f32, Format::Acdb16a);
    auto weights = std::make_shared<Memory>(engine, source);
    auto otherWeights = std::make_shared<Memory>(engine, source);

    const auto key = DnnlExtensionUtils::computeWeightsStringHash(weights, source, destination);
    EXPECT_EQ(key, DnnlExtensionUtils::computeWeightsStringHash(weights, equivalentSource, destination));
    EXPECT_NE(key, DnnlExtensionUtils::computeWeightsStringHash(weights, transposedSource, destination));
    EXPECT_NE(key, DnnlExtensionUtils::computeWeightsStringHash(weights, source, source));
    EXPECT_NE(key, DnnlExtensionUtils::computeWeightsStringHash(otherWeights, source, destination));
}

TEST(WeightsCacheTest, SharedWeightsWithDifferentSourceLayouts) {
    auto cache = std::make_shared<WeightsSharing>();
    auto context = std::make_shared<GraphContext>(Config{}, cache, false);
    const auto& engine = context->getEngine();
    const Shape shape{O, I, H, W};
    using DataType = dnnl::memory::data_type;
    using Format = dnnl::memory::format_tag;
    auto convSource = std::make_shared<DnnlBlockedMemoryDesc>(shape, DataType::f32, Format::abcd);
    auto deconvSource = std::make_shared<DnnlBlockedMemoryDesc>(shape, DataType::f32, Format::bacd);
    auto destination = std::make_shared<DnnlBlockedMemoryDesc>(shape, DataType::f32, Format::Acdb16a);
    auto weights = std::make_shared<Memory>(engine, convSource);
    std::iota(weights->getDataAs<float>(), weights->getDataAs<float>() + elementCount, 1.0F);

    const auto prepare = [&](const DnnlMemoryDescPtr& source) {
        // The source is an effective view of the same buffer, not necessarily its stored descriptor.
        return utils::prepareWeightsMemory(source,
                                           destination,
                                           weights,
                                           engine,
                                           context->getParamsCache(),
                                           cache,
                                           nullptr,
                                           context->getCpuParallel()->get_thread_pool());
    };

    const auto convWeights = prepare(convSource);
    const auto deconvWeights = prepare(deconvSource);
    EXPECT_NE(convWeights->getData(), deconvWeights->getData());
    EXPECT_EQ(convWeights->getData(), prepare(convSource)->getData());
    EXPECT_EQ(deconvWeights->getData(), prepare(deconvSource)->getData());

    Memory unpackedConv(engine, convSource);
    Memory unpackedDeconv(engine, convSource);
    node::Reorder::reorderData(*convWeights,
                               unpackedConv,
                               context->getParamsCache(),
                               context->getCpuParallel()->get_thread_pool());
    node::Reorder::reorderData(*deconvWeights,
                               unpackedDeconv,
                               context->getParamsCache(),
                               context->getCpuParallel()->get_thread_pool());
    // The deconv view swaps the first two dims, so it relies on O == I.
    static_assert(O == I, "the transposed view requires a square O x I weight");
    for (size_t outputChannel = 0; outputChannel < O; ++outputChannel) {
        for (size_t inputChannel = 0; inputChannel < I; ++inputChannel) {
            for (size_t spatial = 0; spatial < spatialSize; ++spatial) {
                const auto index = (outputChannel * I + inputChannel) * spatialSize + spatial;
                const auto transposedIndex = (inputChannel * O + outputChannel) * spatialSize + spatial;
                EXPECT_FLOAT_EQ(unpackedConv.getDataAs<float>()[index], weights->getDataAs<float>()[index]);
                EXPECT_FLOAT_EQ(unpackedDeconv.getDataAs<float>()[index], weights->getDataAs<float>()[transposedIndex]);
            }
        }
    }
}

TEST(WeightsCacheTest, SignedToUnsignedShiftIsPartOfKey) {
    auto cache = std::make_shared<WeightsSharing>();
    auto context = std::make_shared<GraphContext>(Config{}, cache, false);
    const auto& engine = context->getEngine();
    const Shape shape{O, I, H, W};
    using DataType = dnnl::memory::data_type;
    using Format = dnnl::memory::format_tag;
    auto source = std::make_shared<DnnlBlockedMemoryDesc>(shape, DataType::s8, Format::abcd);
    auto destination = std::make_shared<DnnlBlockedMemoryDesc>(shape, DataType::u8, Format::abcd);
    auto weights = std::make_shared<Memory>(engine, source);
    auto* src = weights->getDataAs<int8_t>();
    // Mix negative and non-negative values so shifted and plain results differ for every element.
    std::iota(src, src + elementCount, static_cast<int8_t>(-static_cast<int>(elementCount / 2)));

    const auto prepare = [&](bool shift) {
        return utils::prepareWeightsMemory(source,
                                           destination,
                                           weights,
                                           engine,
                                           context->getParamsCache(),
                                           cache,
                                           nullptr,
                                           context->getCpuParallel()->get_thread_pool(),
                                           shift);
    };

    // Populate the cache with the shifted version first: a plain request must not hit it.
    const auto shifted = prepare(true);
    const auto plain = prepare(false);
    EXPECT_NE(shifted->getData(), plain->getData());
    EXPECT_EQ(shifted->getData(), prepare(true)->getData());
    EXPECT_EQ(plain->getData(), prepare(false)->getData());

    const auto* shiftedData = shifted->getDataAs<const uint8_t>();
    const auto* plainData = plain->getDataAs<const uint8_t>();
    for (size_t i = 0; i < elementCount; ++i) {
        EXPECT_EQ(shiftedData[i], static_cast<uint8_t>(src[i] + 128));
        if (src[i] >= 0) {
            EXPECT_EQ(plainData[i], static_cast<uint8_t>(src[i]));
        }
        EXPECT_NE(plainData[i], shiftedData[i]);
    }
}

}  // namespace
}  // namespace ov::intel_cpu
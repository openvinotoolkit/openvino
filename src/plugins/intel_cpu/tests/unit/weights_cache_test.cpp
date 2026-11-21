// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_cache.hpp"

#include <gtest/gtest.h>

#include <cstddef>
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

TEST(WeightsCacheTest, KeyIncludesSourceDestinationAndBuffer) {
    const dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    const Shape shape{3, 3, 3, 3};
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
    const Shape shape{3, 3, 3, 3};
    using DataType = dnnl::memory::data_type;
    using Format = dnnl::memory::format_tag;
    auto convSource = std::make_shared<DnnlBlockedMemoryDesc>(shape, DataType::f32, Format::abcd);
    auto deconvSource = std::make_shared<DnnlBlockedMemoryDesc>(shape, DataType::f32, Format::bacd);
    auto destination = std::make_shared<DnnlBlockedMemoryDesc>(shape, DataType::f32, Format::Acdb16a);
    auto weights = std::make_shared<Memory>(engine, convSource);
    std::iota(weights->getDataAs<float>(), weights->getDataAs<float>() + 81, 1.0F);

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
    for (size_t outputChannel = 0; outputChannel < 3; ++outputChannel) {
        for (size_t inputChannel = 0; inputChannel < 3; ++inputChannel) {
            for (size_t spatial = 0; spatial < 9; ++spatial) {
                const auto index = (outputChannel * 3 + inputChannel) * 9 + spatial;
                const auto transposedIndex = (inputChannel * 3 + outputChannel) * 9 + spatial;
                EXPECT_FLOAT_EQ(unpackedConv.getDataAs<float>()[index], weights->getDataAs<float>()[index]);
                EXPECT_FLOAT_EQ(unpackedDeconv.getDataAs<float>()[index], weights->getDataAs<float>()[transposedIndex]);
            }
        }
    }
}

}  // namespace
}  // namespace ov::intel_cpu
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_profiling.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "intel_npu/common/igraph.hpp"
#include "openvino/core/except.hpp"

namespace {

ov::ProfilingInfo makeProfilingInfo(const std::string& name) {
    ov::ProfilingInfo info;
    info.node_name = name;
    return info;
}

class TestGraph final : public intel_npu::IGraph {
public:
    std::optional<bool> is_profiling_blob() const override {
        return true;
    }

    bool has_profiling_decoder() const override {
        return decoderAvailable;
    }

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& data) const override {
        receivedData = data;
        if (!decoderAvailable) {
            return intel_npu::IGraph::process_profiling_output(data);
        }
        if (throwOnDecode) {
            OPENVINO_THROW("synthetic graph decoder failure");
        }
        return graphStatistics;
    }

    bool decoderAvailable = true;
    bool throwOnDecode = false;
    mutable std::vector<uint8_t> receivedData;
    std::vector<ov::ProfilingInfo> graphStatistics;
};

struct TestQuery {
    template <typename ProfilingData>
    std::vector<ProfilingData> getData() const {
        ++rawDataCalls;
        return std::vector<ProfilingData>(rawData.begin(), rawData.end());
    }

    intel_npu::zeroProfiling::LayerStatistics getLayerStatistics() const {
        ++layerStatisticsCalls;
        return layerStatistics;
    }

    std::vector<uint8_t> rawData{4, 5, 6};
    intel_npu::zeroProfiling::LayerStatistics layerStatistics{makeProfilingInfo("level-zero")};
    mutable size_t rawDataCalls = 0;
    mutable size_t layerStatisticsCalls = 0;
};

TEST(ZeroProfilingTest, PrefersGraphDecoderWhenItSucceeds) {
    TestGraph graph;
    graph.graphStatistics = {makeProfilingInfo("graph-decoded")};
    TestQuery query;

    const auto result = intel_npu::zeroProfiling::getModelLayerStatistics(graph, query);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().node_name, "graph-decoded");
    EXPECT_EQ(graph.receivedData, query.rawData);
    EXPECT_EQ(query.rawDataCalls, 1u);
    EXPECT_EQ(query.layerStatisticsCalls, 0u);
}

TEST(ZeroProfilingTest, FallsBackToLevelZeroStatisticsWhenGraphHasNoDecoder) {
    TestGraph graph;
    graph.decoderAvailable = false;
    TestQuery query;

    const auto result = intel_npu::zeroProfiling::getModelLayerStatistics(graph, query);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().node_name, "level-zero");
    // No decoder means the raw profiling buffer is never worth fetching.
    EXPECT_TRUE(graph.receivedData.empty());
    EXPECT_EQ(query.rawDataCalls, 0u);
    EXPECT_EQ(query.layerStatisticsCalls, 1u);
}

TEST(ZeroProfilingTest, FallsBackToLevelZeroStatisticsWhenGraphDecoderThrows) {
    TestGraph graph;
    graph.throwOnDecode = true;
    TestQuery query;

    const auto result = intel_npu::zeroProfiling::getModelLayerStatistics(graph, query);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().node_name, "level-zero");
    EXPECT_EQ(query.layerStatisticsCalls, 1u);
}

}  // namespace

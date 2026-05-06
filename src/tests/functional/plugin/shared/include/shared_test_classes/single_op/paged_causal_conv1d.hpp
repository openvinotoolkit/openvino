// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

struct PagedCausalConv1DLayerParams {
    int32_t hidden_size;
    int32_t kernel_size;
    bool has_bias;
    // Multiple sets of seq_lengths/cache_intervals to exercise dynamic shapes.
    // Each inner vector represents one inference iteration with different token counts.
    std::vector<std::vector<int32_t>> seq_lengths_sets;
    std::vector<std::vector<int32_t>> cache_intervals_sets;
    ov::element::Type element_type;
    std::string target_device;
};

class PagedCausalConv1DLayerTest : public testing::WithParamInterface<PagedCausalConv1DLayerParams>,
                                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedCausalConv1DLayerParams>& obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    std::vector<ov::Tensor> calculate_refs() override;
    std::vector<ov::Tensor> get_plugin_outputs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void SetUp() override;

private:
    struct IterationData {
        std::vector<int32_t> subsequence_begins;
        std::vector<int32_t> block_indices;
        std::vector<int32_t> block_indices_begins;
        std::vector<int32_t> past_lens;
        std::vector<int32_t> cache_interval;
    };

    std::map<std::shared_ptr<ov::Node>, ov::Tensor> host_inputs;
    ov::element::Type data_type;
    std::vector<IterationData> m_iteration_data;
    size_t m_current_iteration = 0;
};

}  // namespace ov::test

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

using PagedSelectiveSSMLayerParams = std::tuple<int32_t,
                                                int32_t,
                                                int32_t,
                                                int32_t,
                                                std::vector<int32_t>,
                                                std::vector<int32_t>,
                                                std::vector<int32_t>,
                                                ov::element::Type,
                                                ov::element::Type,
                                                ov::element::Type,
                                                std::string>;  // num_heads, num_groups, head_dim, state_size,
                                                               // seq_lengths, num_processed_tokens, cache_intervals,
                                                               // data_type, state_type, index_type, target_device

class PagedSelectiveSSMLayerTest : public testing::WithParamInterface<PagedSelectiveSSMLayerParams>,
                                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedSelectiveSSMLayerParams>& obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    std::vector<ov::Tensor> calculate_refs() override;
    std::vector<ov::Tensor> get_plugin_outputs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void SetUp() override;

private:
    std::map<std::shared_ptr<ov::Node>, ov::Tensor> host_inputs;
    ov::element::Type data_type;
    ov::element::Type state_type;
};

}  // namespace ov::test

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <optional>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

struct PagedAttentionRotationStruct {
    std::vector<int> rotated_block_indices;
    ov::Shape rotation_deltas;
    ov::Shape rotation_trig_lut;
    };

struct PagedAttentionMiscInpStruct {
    std::vector<float> scale;
    std::optional<int> sliding_window;
    std::vector<float> alibi_slopes;
    int max_context_len;
    };

struct PagedAttentionIntVectorsStruct {
        std::vector<int> past_lens;
        std::vector<int> subsequence_begins;
        std::vector<int> block_indices;
        std::vector<int> block_indices_begins;
        };

using PagedAttentionParamsTuple = typename std::tuple<
                                std::vector<InputShape>,
                                PagedAttentionIntVectorsStruct,
                                PagedAttentionMiscInpStruct,
                                std::optional<PagedAttentionRotationStruct>,
                                ov::element::Type,         // Model type
                                std::string>;              // Device name

class PagedAttentionLayerTest : public testing::WithParamInterface<PagedAttentionParamsTuple>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttentionParamsTuple> &obj);

protected:
    void SetUp() override;
};
} //  namespace ov::test

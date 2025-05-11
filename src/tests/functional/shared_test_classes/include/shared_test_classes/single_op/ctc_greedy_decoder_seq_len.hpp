// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<InputShape>,   // Input shape
        int,                       // Sequence lengths
        ov::element::Type,         // Probabilities precision
        ov::element::Type,         // Indices precision
        int,                       // Blank index
        bool,                      // Merge repeated
        std::string                // Device name
    > ctcGreedyDecoderSeqLenParams;

class CTCGreedyDecoderSeqLenLayerTest
    :  public testing::WithParamInterface<ctcGreedyDecoderSeqLenParams>,
       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ctcGreedyDecoderSeqLenParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

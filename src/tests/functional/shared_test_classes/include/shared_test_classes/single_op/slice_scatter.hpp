// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2018-2024 Intel Corporation
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
struct SliceScatterSpecificParams {
        std::vector<InputShape> shapes;
        std::vector<int64_t> start;
        std::vector<int64_t> stop;
        std::vector<int64_t> step;
        std::vector<int64_t> axes;
};

using SliceScatterParams = std::tuple<
        SliceScatterSpecificParams,        // SliceScatter specific parameters
        ov::element::Type,                 // Model type
        ov::test::TargetDevice             // Device name
>;

class SliceScatterLayerTest : public testing::WithParamInterface<SliceScatterParams>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SliceScatterParams> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov

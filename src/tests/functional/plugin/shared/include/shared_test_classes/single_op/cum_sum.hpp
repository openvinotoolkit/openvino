// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<
        std::vector<InputShape>,     // Input shapes
        ov::element::Type,           // Model type
        int64_t,                     // Axis
        bool,                        // Exclusive
        bool,                        // Reverse
        std::string> cumSumParams;   // Device name

class CumSumLayerTest : public testing::WithParamInterface<cumSumParams>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<cumSumParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

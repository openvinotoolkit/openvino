// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using ConvertColorNV12ParamsTuple = std::tuple<
        std::vector<InputShape>,                       // Input Shape
        ov::element::Type,                             // Element type
        bool,                                          // Conversion type
        bool,                                          // 1 or 2 planes
        std::string>;                                  // Device name

class ConvertColorNV12LayerTest : public testing::WithParamInterface<ConvertColorNV12ParamsTuple>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertColorNV12ParamsTuple> &obj);

protected:
    void SetUp() override;
};
} // namespace test
} // namespace ov

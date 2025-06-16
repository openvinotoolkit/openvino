// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include "gtest/gtest.h"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
using ConversionParamsTuple = typename std::tuple<ov::test::utils::ConversionTypes,  // Convertion op type
                                                  std::vector<InputShape>,           // Input shapes
                                                  ov::element::Type,                 // Input type
                                                  ov::element::Type,                 // Convert type
                                                  std::string>;                      // Device name

class ConversionLayerTest : public testing::WithParamInterface<ConversionParamsTuple>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConversionParamsTuple>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov

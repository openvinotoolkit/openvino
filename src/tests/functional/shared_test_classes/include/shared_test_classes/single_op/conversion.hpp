// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include "gtest/gtest.h"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace ov {
namespace test {
using ConversionParamsTuple = typename std::tuple<ngraph::helpers::ConversionTypes,  // Convertion op type
                                                  std::vector<InputShape>,  // Input1 shapes
                                                  ov::element::Type,        // Input1 precision
                                                  ov::element::Type,        // Input2 precision
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

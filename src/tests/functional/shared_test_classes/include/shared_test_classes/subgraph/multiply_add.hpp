// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using MultiplyAddParamsTuple = typename std::tuple<ov::Shape,          // input shapes
                                                   ov::element::Type,  // Input precision
                                                   std::string>;       // Device name

class MultiplyAddLayerTest : public testing::WithParamInterface<MultiplyAddParamsTuple>,
                             virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultiplyAddParamsTuple>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

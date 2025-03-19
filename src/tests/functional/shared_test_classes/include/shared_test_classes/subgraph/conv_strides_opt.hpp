// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<ov::Shape,  // input shape
                   ov::op::PadType,
                   std::string  // Device name
                   >
    ConvStridesOptParams;

class ConvStridesOpt : public testing::WithParamInterface<ConvStridesOptParams>,
                       virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvStridesOptParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

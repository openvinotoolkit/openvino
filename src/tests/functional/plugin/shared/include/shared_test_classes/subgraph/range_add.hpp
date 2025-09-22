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

typedef std::tuple<float,              // start
                   float,              // stop
                   float,              // step
                   ov::element::Type,  // Input type
                   std::string         // Target device name
                   >
    RangeParams;

// ------------------------------ V0 ------------------------------

class RangeAddSubgraphTest : public testing::WithParamInterface<RangeParams>,
                             virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RangeParams>& obj);

protected:
    void SetUp() override;
};

// ------------------------------ V4 ------------------------------

class RangeNumpyAddSubgraphTest : public testing::WithParamInterface<RangeParams>,
                                  virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RangeParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

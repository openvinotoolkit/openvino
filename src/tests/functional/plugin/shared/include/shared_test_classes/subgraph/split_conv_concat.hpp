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

class SplitConvConcatBase : public ov::test::SubgraphBaseStaticTest {
protected:
    void configure_test(const ov::test::BasicParams& param);
};

class SplitConvConcat : public testing::WithParamInterface<ov::test::BasicParams>, virtual public SplitConvConcatBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::test::BasicParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

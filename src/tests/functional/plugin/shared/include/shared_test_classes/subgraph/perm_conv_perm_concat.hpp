// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<ov::element::Type,  // Network Precision
                   std::string,        // Target Device
                   ov::Shape,          // Input shape
                   ov::Shape,          // Kernel shape
                   size_t,             // Output channels
                   ov::AnyMap          // Configuration
                   >
    PermConvPermConcatParams;

class PermConvPermConcat : public testing::WithParamInterface<PermConvPermConcatParams>,
                           virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PermConvPermConcatParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

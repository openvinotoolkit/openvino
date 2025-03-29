// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using bucketizeParamsTuple = std::tuple<
    std::vector<InputShape>,        // data shape, bucket shape
    bool,                           // Right edge of interval
    ov::element::Type,              // Data input precision
    ov::element::Type,              // Buckets input precision
    ov::element::Type,              // Output precision
    std::string>;                   // Device name

class BucketizeLayerTest : public testing::WithParamInterface<bucketizeParamsTuple>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<bucketizeParamsTuple>& obj);
protected:
    void SetUp() override;
};


} // namespace test
} // namespace ov
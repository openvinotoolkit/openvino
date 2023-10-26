// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        int64_t,                        // keepK
        int64_t,                        // axis
        ov::op::v1::TopK::Mode,         // mode
        ov::op::v1::TopK::SortType,     // sort
        ov::element::Type,              // Model type
        std::vector<InputShape>,        // Input shape
        std::string                     // Target device name
> TopKParams;

class TopKLayerTest : public testing::WithParamInterface<TopKParams>,
                      virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TopKParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov

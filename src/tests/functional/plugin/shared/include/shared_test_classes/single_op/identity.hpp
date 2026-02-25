// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {
typedef std::tuple<
        ov::element::Type,              // Model type
        std::vector<InputShape>,        // Input shapes
        std::string                     // Target device name
> identityParams;

class IdentityLayerTest : public testing::WithParamInterface<identityParams>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<identityParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace ov::test

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<size_t>,            // Input order
        ov::element::Type,              // Model type
        std::vector<InputShape>,        // Input shapes
        std::string                     // Target device name
> transposeParams;

class TransposeLayerTest : public testing::WithParamInterface<transposeParams>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<transposeParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov

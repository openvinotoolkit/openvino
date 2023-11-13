// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using IsInfParams = std::tuple<
                        std::vector<InputShape>,             // Data shape
                        bool,                                // Detect negative
                        bool,                                // Detect positive
                        ov::element::Type,                   // Model type
                        std::string,                         // Device name
                        std::map<std::string, std::string>   // Additional config
>;

class IsInfLayerTest : public testing::WithParamInterface<IsInfParams>,
                        virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<IsInfParams>& obj);
protected:
    void SetUp() override;
};
} // namespace test
} // namespace ov

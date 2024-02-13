// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using InverseTestParams = typename std::tuple<ov::Shape,         // input shape
                                              ov::element::Type, // element type
                                              bool,              // adjoint
                                              std::string        // device_name
                                              >;

class InverseLayerTest : public testing::WithParamInterface<InverseTestParams>, virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InverseTestParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov

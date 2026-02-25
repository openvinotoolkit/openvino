// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using StatefulSdpaBoolMaskParams = ov::element::Type;

class StatefulSdpaBoolMaskTest : public testing::WithParamInterface<StatefulSdpaBoolMaskParams>,
                                 virtual public ov::test::SubgraphBaseTest,
                                 public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StatefulSdpaBoolMaskParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

}  // namespace test
}  // namespace ov

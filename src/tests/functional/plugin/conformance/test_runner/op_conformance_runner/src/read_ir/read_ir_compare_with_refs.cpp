// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"

#include "read_ir_test/read_ir_compare_with_refs.hpp"
#include "conformance.hpp"

namespace ov {
namespace test {
namespace conformance {
namespace op {

using namespace ov::test::subgraph;

namespace {

class ThresholdReadIRTest : public ReadIRTest {
protected:
    void SetUp() override {
        ReadIRTest::SetUp();
        abs_threshold = 1.0001;
    }
};

TEST_P(ThresholdReadIRTest, CpuReadIR) {
    run();
}

INSTANTIATE_TEST_SUITE_P(conformance,
                         ThresholdReadIRTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(getModelPaths(IRFolderPaths)),
                                 ::testing::Values(targetDevice),
                                 ::testing::Values(pluginConfig)),
                         ThresholdReadIRTest::getTestCaseName);

}  // namespace

}  // namespace op
}  // namespace conformance
}  // namespace test
}  // namespace ov

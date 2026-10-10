// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "shared_test_classes/single_op/grouped_matmul.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

using CPUTestUtils::CPUSpecificParams;
using CPUTestUtils::CPUTestsBase;

// CPU counterparts of the shared GroupedMatMul suites: on top of numerics they check the executable
// graph - which primitive got built, and for the compressed flavour that the weights reach it still
// compressed. Model / offsets / input generation is reused from GroupedMatMulTestBase.

using GroupedMatMulCPUTestParams = std::tuple<GroupedMatMulShapeParams,  // shape bundle + routing
                                              ov::element::Type,         // activation precision
                                              ov::AnyMap,                // additional plugin config
                                              CPUSpecificParams          // expected impl type / layouts
                                              >;

class GroupedMatMulLayerCPUTest : public testing::WithParamInterface<GroupedMatMulCPUTestParams>,
                                  virtual public GroupedMatMulTestBase,
                                  public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupedMatMulCPUTestParams>& obj);

protected:
    void SetUp() override;
    std::shared_ptr<ov::Node> build_weights() override;
    void check_results();
};

using GroupedMatMulCompressedCPUTestParams = std::tuple<GroupedMatMulShapeParams,
                                                        ov::element::Type,  // activation precision
                                                        ov::element::Type,  // weights (compressed) precision
                                                        ov::element::Type,  // decompression precision
                                                        ov::element::Type,  // scale precision
                                                        ov::test::utils::DecompressionType,  // multiply type
                                                        ov::test::utils::DecompressionType,  // subtract type
                                                        bool,               // reshape on decompression constants
                                                        int,                // decompression group size (-1 = per-OC)
                                                        bool,               // expect the compressed primitive
                                                        ov::AnyMap,         // additional plugin config
                                                        CPUSpecificParams>  // expected impl type / layouts
    ;

class GroupedMatMulCompressedLayerCPUTest : public testing::WithParamInterface<GroupedMatMulCompressedCPUTestParams>,
                                            virtual public GroupedMatMulTestBase,
                                            public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupedMatMulCompressedCPUTestParams>& obj);

protected:
    void SetUp() override;
    std::shared_ptr<ov::Node> build_weights() override;
    void check_results();

private:
    ov::element::Type weights_prec_;
    ov::element::Type decomp_prec_;
    ov::element::Type scale_prec_;
    ov::test::utils::DecompressionType multiply_type_;
    ov::test::utils::DecompressionType subtract_type_;
    bool reshape_on_decomp_ = false;
    int group_size_ = -1;
    bool expect_compressed_ = true;
};

}  // namespace test
}  // namespace ov

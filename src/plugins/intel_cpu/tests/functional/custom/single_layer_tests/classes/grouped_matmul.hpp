// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "shared_test_classes/single_op/grouped_matmul.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov::test {

using GroupedMatMulCPUTestParams = std::tuple<GroupedMatMulShapeParams,          // shape bundle + routing
                                              ov::element::Type,                 // activation precision
                                              ov::AnyMap,                        // additional plugin config
                                              CPUTestUtils::CPUSpecificParams>;  // expected impl type / layouts

class GroupedMatMulLayerCPUTest : public testing::WithParamInterface<GroupedMatMulCPUTestParams>,
                                  virtual public GroupedMatMulTestBase,
                                  public CPUTestUtils::CPUTestsBase {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<GroupedMatMulCPUTestParams>& obj);

protected:
    void SetUp() override;
    std::shared_ptr<ov::Node> build_weights() override;
    void check_results();
};

using GroupedMatMulCompressedCPUTestParams = std::tuple<GroupedMatMulShapeParams,
                                                        ov::element::Type,                           // activation precision
                                                        ov::element::Type,                           // weights (compressed) precision
                                                        ov::element::Type,                           // decompression precision
                                                        ov::element::Type,                           // scale precision
                                                        ov::test::utils::DecompressionType,          // multiply type
                                                        ov::test::utils::DecompressionType,          // subtract type
                                                        bool,                                        // reshape on decompression constants
                                                        int,                                         // decompression group size (-1 = per-OC)
                                                        bool,                                        // expect the compressed primitive
                                                        ov::AnyMap,                                  // additional plugin config
                                                        CPUTestUtils::CPUSpecificParams>;            // expected impl type / layouts

class GroupedMatMulCompressedLayerCPUTest : public testing::WithParamInterface<GroupedMatMulCompressedCPUTestParams>,
                                            virtual public GroupedMatMulTestBase,
                                            public CPUTestUtils::CPUTestsBase {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<GroupedMatMulCompressedCPUTestParams>& obj);

protected:
    void SetUp() override;
    std::shared_ptr<ov::Node> build_weights() override;
    void check_results();

private:
    ov::element::Type m_weights_prec;
    ov::element::Type m_decomp_prec;
    ov::element::Type m_scale_prec;
    ov::test::utils::DecompressionType m_multiply_type;
    ov::test::utils::DecompressionType m_subtract_type;
    bool m_reshape_on_decomp = false;
    int m_group_size = -1;
    bool m_expect_compressed = true;
};

}  // namespace ov::test

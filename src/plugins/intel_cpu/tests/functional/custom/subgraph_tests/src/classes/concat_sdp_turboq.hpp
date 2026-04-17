// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

// TurboQuant KV cache integration test.
// Builds the same ReadValue/Gather/Concat/SDPA/Assign subgraph as ConcatSDPTest
// but sets KEY_CACHE_PRECISION and VALUE_CACHE_PRECISION to "tbq3" or "tbq4".
// Requires head_dim == 128 (TBQ v1 constraint).

typedef std::tuple<ElementType,
                   std::vector<InputShape>,
                   std::string,   // kCacheMode: "none", "u8", "tbq3", "tbq4", "polar3", "polar4"
                   std::string,   // vCacheMode: "none", "u8", "tbq3", "tbq4", "polar3", "polar4"
                   std::string,   // rotationMode: "wht", "dense"
                   bool>          // is_causal
    ConcatSDPTurboQTestParams;

class ConcatSDPTurboQTest : public testing::WithParamInterface<ConcatSDPTurboQTestParams>,
                            virtual public ov::test::SubgraphBaseTest,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatSDPTurboQTestParams>& obj);
    void generate(int idx, const std::vector<ov::Shape>& targetInputStaticShapes);
    void prepare();
    void reset();
    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model);

protected:
    void SetUp() override;

private:
    std::string m_kCacheMode;
    std::string m_vCacheMode;
    std::string m_rotationMode;
};

}  // namespace test
}  // namespace ov

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/runtime_weights_update.hpp"

#include <sstream>
#include <tuple>

namespace ov::test::gpu {
namespace {
using RuntimeWeightsUpdateParams = std::tuple<bool, bool, bool, RuntimeWeightsShapeMode>;

class RuntimeWeightsUpdateTest : public RuntimeWeightsUpdateTestBase, public testing::WithParamInterface<RuntimeWeightsUpdateParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RuntimeWeightsUpdateParams>& obj) {
        const auto& [matmul, compressed, constant_weights, shape_mode] = obj.param;
        std::ostringstream result;
        result << (matmul ? "MatMul" : "Convolution") << (compressed ? "_Compressed" : "_F16") << (constant_weights ? "_Constant" : "_Runtime") << "_"
               << shape_mode;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [matmul, compressed, constant_weights, shape_mode] = GetParam();
        setup_model(matmul, compressed, constant_weights, shape_mode);
    }
};

TEST_P(RuntimeWeightsUpdateTest, uses_current_weights) {
    run();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    RuntimeWeightsUpdateTest,
    testing::Combine(testing::Bool(),
                     testing::Bool(),
                     testing::Bool(),
                     testing::Values(RuntimeWeightsShapeMode::Static, RuntimeWeightsShapeMode::DynamicSameShape, RuntimeWeightsShapeMode::DynamicShapeChange)),
    RuntimeWeightsUpdateTest::getTestCaseName);

}  // namespace
}  // namespace ov::test::gpu

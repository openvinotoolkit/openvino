// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/runtime_weights_update.hpp"

#include <sstream>
#include <tuple>

namespace ov::test::gpu {
namespace {
using RuntimeWeightsUpdateParams = std::tuple<bool, bool, bool, InputShape>;

class RuntimeWeightsUpdateTest : public RuntimeWeightsUpdateTestBase, public testing::WithParamInterface<RuntimeWeightsUpdateParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RuntimeWeightsUpdateParams>& obj) {
        const auto& [matmul, compressed, constant_weights, input_shape] = obj.param;
        std::ostringstream result;
        result << (matmul ? "MatMul" : "Convolution") << (compressed ? "_Compressed" : "_F16") << (constant_weights ? "_Constant" : "_Runtime") << "_"
               << "IS=" << ov::test::utils::partialShape2str({input_shape.first});
        for (const auto& shape : input_shape.second) {
            result << "_TS=" << ov::test::utils::vec2str(shape);
        }
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [matmul, compressed, constant_weights, input_shape] = GetParam();
        setup_model(matmul, compressed, constant_weights, input_shape);
    }
};

TEST_P(RuntimeWeightsUpdateTest, uses_current_weights) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_MatMul,
                         RuntimeWeightsUpdateTest,
                         testing::Combine(testing::Values(true), testing::Bool(), testing::Bool(), testing::ValuesIn(runtime_weights_input_shapes(true))),
                         RuntimeWeightsUpdateTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution,
                         RuntimeWeightsUpdateTest,
                         testing::Combine(testing::Values(false), testing::Bool(), testing::Bool(), testing::ValuesIn(runtime_weights_input_shapes(false))),
                         RuntimeWeightsUpdateTest::getTestCaseName);
}  // namespace
}  // namespace ov::test::gpu

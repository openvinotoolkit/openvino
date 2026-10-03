// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/block_sparse_attention.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace {

// End-to-end CPU-plugin correctness test: `SubgraphBaseTest::run()` compiles and infers this
// graph on the real intel_cpu plugin, then automatically cross-checks the result against the
// same graph run through the Template plugin's interpreter (`calculate_refs()` in
// ov_subgraph.cpp calls `ov::test::utils::infer_on_template(...)`) -- i.e. against the very
// backend already validated numerically against ScaledDotProductAttention in
// src/plugins/template/tests/functional/op_reference/block_sparse_attention.cpp. No separate,
// hand-derived oracle is needed here: this test exists to prove the CPU node's registration,
// primitive-descriptor negotiation, and executor wiring produce the same result as the
// already-trusted reference path, not to re-validate the attention math itself.
struct BlockSparseAttentionCPUTestParams {
    ov::Shape queryShape;
    ov::Shape keyShape;
    ov::Shape valueShape;
    ov::Shape blockIndicesShape;
    int64_t blockSize;
    bool causal;
    bool withMask;
    std::string name;
};

using BlockSparseAttentionCPUTestParamsTuple = std::tuple<BlockSparseAttentionCPUTestParams, ElementType>;

class BlockSparseAttentionCPUTest : public testing::WithParamInterface<BlockSparseAttentionCPUTestParamsTuple>,
                                     virtual public SubgraphBaseTest,
                                     public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BlockSparseAttentionCPUTestParamsTuple>& obj) {
        const auto& [params, prc] = obj.param;
        std::ostringstream result;
        result << params.name << "_prc=" << prc;
        return result.str();
    }

protected:
    BlockSparseAttentionCPUTestParams m_params;

    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        const auto& [params, prc] = this->GetParam();
        m_params = params;

        const auto query = std::make_shared<ov::op::v0::Parameter>(prc, params.queryShape);
        const auto key = std::make_shared<ov::op::v0::Parameter>(prc, params.keyShape);
        const auto value = std::make_shared<ov::op::v0::Parameter>(prc, params.valueShape);
        const auto blockIndices =
            std::make_shared<ov::op::v0::Parameter>(ov::element::i64, params.blockIndicesShape);

        ov::OutputVector graphInputs{query, key, value, blockIndices};
        ov::ParameterVector paramVec{query, key, value, blockIndices};
        std::vector<ov::Shape> shapesInOrder{params.queryShape, params.keyShape, params.valueShape,
                                             params.blockIndicesShape};

        if (params.withMask) {
            const auto mask =
                std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, params.blockIndicesShape);
            graphInputs.push_back(mask);
            paramVec.push_back(mask);
            shapesInOrder.push_back(params.blockIndicesShape);
        }

        const auto op =
            std::make_shared<ov::op::v17::BlockSparseAttention>(graphInputs, params.blockSize, params.causal);
        function = std::make_shared<ov::Model>(ov::OutputVector{op}, paramVec, "BlockSparseAttention");

        init_input_shapes(ov::test::static_shapes_to_test_representation(shapesInOrder));
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        const int64_t numKvBlocks = static_cast<int64_t>(m_params.keyShape[2]) / m_params.blockSize;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            const auto& shape = targetInputStaticShapes[i];
            ov::Tensor tensor;

            if (i == 3) {
                // block_indices: round-robin over the valid [0, numKvBlocks) range instead of
                // arbitrary random data, which would produce out-of-range indices.
                std::vector<int64_t> data(shape_size(shape));
                for (size_t idx = 0; idx < data.size(); ++idx) {
                    data[idx] = static_cast<int64_t>(idx % static_cast<size_t>(numKvBlocks));
                }
                tensor = ov::Tensor(ov::element::i64, shape);
                std::memcpy(tensor.data(), data.data(), data.size() * sizeof(int64_t));
            } else if (i == 4) {
                // block_indices_mask: all-true -- every generated index above is meant to be used.
                std::vector<char> data(shape_size(shape), 1);
                tensor = ov::Tensor(ov::element::boolean, shape);
                std::memcpy(tensor.data(), data.data(), data.size());
            } else {
                tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), shape, 2, -1, 100);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(BlockSparseAttentionCPUTest, CompareWithRefs) {
    run();
}

const std::vector<BlockSparseAttentionCPUTestParams> params = {
    {{1, 2, 8, 4}, {1, 2, 8, 4}, {1, 2, 8, 4}, {1, 2, 4, 4}, 2, false, false, "DenseEquivalent"},
    {{1, 2, 8, 4}, {1, 2, 8, 4}, {1, 2, 8, 4}, {1, 2, 4, 4}, 2, true, false, "Causal"},
    {{1, 2, 8, 4}, {1, 2, 8, 4}, {1, 2, 8, 4}, {1, 2, 4, 2}, 2, false, false, "SparseUniformSelection"},
    {{1, 2, 8, 4}, {1, 2, 8, 4}, {1, 2, 8, 4}, {1, 2, 4, 3}, 2, false, true, "PaddingMask"},
    {{1, 4, 8, 4}, {1, 1, 8, 4}, {1, 1, 8, 4}, {1, 1, 4, 4}, 2, false, false, "HeadBroadcast"},
    {{2, 2, 16, 8}, {2, 2, 16, 8}, {2, 2, 16, 8}, {2, 2, 8, 3}, 2, false, true, "MultiBatchLargerBlocks"},
};

INSTANTIATE_TEST_SUITE_P(smoke_BlockSparseAttention,
                         BlockSparseAttentionCPUTest,
                         ::testing::Combine(::testing::ValuesIn(params),
                                            ::testing::Values(ElementType::f32, ElementType::bf16)),
                         BlockSparseAttentionCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov

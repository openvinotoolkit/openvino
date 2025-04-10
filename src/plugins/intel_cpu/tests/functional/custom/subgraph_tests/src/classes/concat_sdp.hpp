// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

// Subgraph:
/*                            Parameter
 *                                |
 *       Parameter    ReadValue   |    ReadValue  Parameter
 *           \           /        |       \          /
 *         Gather       /               Gather      /
 *             \       /          |         \      /
 *               Concat           |          Concat
 *                / \             |            / \
 *               /   \            |           /   \
 *              /     \           |          /     \
 *          Assign     ScaledDotProductAttention  Assign
 *                                |
 *                               Add
 *                                |
 *                              Result
 */
template<typename IT, typename T>
void strided_iota(IT first, size_t n, T value, T stride);

typedef std::tuple<ElementType, std::vector<InputShape>, bool, bool, bool> ConcatSDPTestParams;

class ConcatSDPTest :
        public testing::WithParamInterface<ConcatSDPTestParams>,
        virtual public ov::test::SubgraphBaseTest,
        public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatSDPTestParams>& obj);
    void generate(int idx, const std::vector<ov::Shape>& targetInputStaticShapes);
    void prepare();
    void reset();
    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model);
    bool m_forceKVU8;
    bool m_hasShapeOf;
    bool m_isDiffKVHeadSize;
protected:
    void SetUp() override;

    static constexpr size_t m_diffKVHeadSize = 16;
};

}  // namespace test
}  // namespace ov

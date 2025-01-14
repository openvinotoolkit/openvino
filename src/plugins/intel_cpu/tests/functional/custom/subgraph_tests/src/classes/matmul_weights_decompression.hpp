// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

/*
 * WP - weights precision
 * DP - decompression precision
 * IP - input precision
 * SP - scale precision
 * Opt - optional
 *                        Subtract_const(WP)
 *                           /
 *    Weights(WP)     Convert(DP)
 *       |               /           Multiply_const(SP)
 *    Convert(DP)   Reshape (Opt)      /
 *            \        /          Convert(if SP != DP)
 *            Subtract(Opt)       /
 *                  \         Reshape (Opt)
 *                   \         /
 *                    Multiply
 *                      |
 *                   Reshape (in case of group decompression)
 *                      |
 *                   Convert (if IP != DP)
 *                      |
 *      Data(IP)   Transpose(Opt)
 *            \     /
 *             Matmul
 *               |
 *              Bias
 */
typedef std::tuple<MatMulDecompressionShapeParams,
                   ov::test::ElementType,      // weights precision
                   ov::test::ElementType,      // decompression precision
                   ov::test::ElementType,      // scale precision
                   bool,                       // transpose on weights
                   DecompressionType,          // decompression multiply type
                   DecompressionType,          // decompression subtract type
                   bool,                       // reshape on decompression constants
                   ov::AnyMap,                 // additional config
                   fusingSpecificParams,
                   bool>  // should use decompression implementation
    MatmulWeightsDecompressionParams;

class MatmulWeightsDecompression : public testing::WithParamInterface<MatmulWeightsDecompressionParams>,
                                   virtual public SubgraphBaseTest,
                                   public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatmulWeightsDecompressionParams> obj);

protected:
    std::shared_ptr<ov::Model> initSubgraph(const ov::PartialShape& data_shape,
                                            const ov::Shape& weights_shape,
                                            const int group_size,
                                            const ov::element::Type data_precision,
                                            const ov::element::Type weights_precision,
                                            const ov::element::Type decompression_precision,
                                            const ov::element::Type scale_precision,
                                            const bool transpose_weights,
                                            const DecompressionType decompression_multiply_type,
                                            const DecompressionType decompression_subtract_type,
                                            const bool reshape_on_decompression);

    void SetUp() override;

    void check_results();
};

}  // namespace test
}  // namespace ov

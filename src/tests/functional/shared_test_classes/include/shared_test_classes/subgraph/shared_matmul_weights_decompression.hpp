// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"

namespace ov {
namespace test {
/*
 * WP - weights precision
 * DP - decompression precision
 * IP - input precision
 * Opt - optional
 *                 Weights(WP)     Subtract_const(WP)
 *                    |               /
 *                 Convert(DP)    Convert(DP)
 *                         \        /
 *                         Subtract(Opt)
 *                               \          Multiply_const(DP)
 *                                \         /
 *                                 Multiply
 *                                   |
 *                                Reshape (in case of group decompression)
 *                                   |
 *                                Convert (if IP != DP)
 *                               /                    \
 *      Data(IP)   Transpose(Opt)                      Transpose(Opt)     Data2(IP)
 *            \     /                                               \     /
 *             Matmul                                                Matmul
 */

using MatmulSharedWeightsDecompressionParams = std::tuple<std::string,                 // target device
                                                          MatMulDecompressionShapeParams,
                                                          ElementType,                 // weights precision
                                                          ElementType,                 // decompression precision
                                                          bool,                        // transpose on weights
                                                          DecompressionType,           // decompression subtract type
                                                          bool>;                       // use matmul decompression implementation

class SharedMatmulWeightsDecompression : public testing::WithParamInterface<MatmulSharedWeightsDecompressionParams>,
                                         virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatmulSharedWeightsDecompressionParams> obj);

protected:
    std::shared_ptr<ov::Model> initSubgraph(const ov::PartialShape& data_shape,
                                            const ov::Shape& weights_shape,
                                            const int group_size,
                                            const ov::element::Type data_precision,
                                            const ov::element::Type weights_precision,
                                            const ov::element::Type decompression_precision,
                                            const bool transpose_weights,
                                            const DecompressionType decompression_subtract_type);
    void SetUp() override;
    void check_results();
};

}  // namespace test
}  // namespace ov
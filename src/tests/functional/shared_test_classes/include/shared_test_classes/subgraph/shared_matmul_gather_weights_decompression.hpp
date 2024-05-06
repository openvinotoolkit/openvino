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
 *                Data(IP)        Convert (if IP != DP)        Indices(I32)
 *                        \      /                     \      /
 *               Matmul(transpose_b = true)             Gather
*/

using SharedMatmulAndGatherWeightsDecompressionParams = std::tuple<std::string,                 // target device
                                                                   GatherDecompressionShapeParams,
                                                                   ElementType,                 // weights precision
                                                                   ElementType,                 // decompression precision
                                                                   bool,                        // decompression subtract
                                                                   bool>;                       // use matmul decompression implementation

class SharedMatmulAndGatherWeightsDecompression : public testing::WithParamInterface<SharedMatmulAndGatherWeightsDecompressionParams>,
                                                  virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SharedMatmulAndGatherWeightsDecompressionParams> obj);

protected:
    std::shared_ptr<ov::Model> initSubgraph(const ov::Shape& data_shape,
                                            const ov::PartialShape& indices_shape,
                                            const int axis,
                                            const int64_t batch_dims,
                                            const int group_size,
                                            const ov::element::Type data_precision,
                                            const ov::element::Type output_precision,
                                            const bool add_subtract);
    void SetUp() override;
    void check_results();
};


}  // namespace test
}  // namespace ov
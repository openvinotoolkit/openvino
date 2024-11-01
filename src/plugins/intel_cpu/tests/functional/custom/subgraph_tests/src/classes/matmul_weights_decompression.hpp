// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// #include "openvino/opsets/opset13.hpp"
// #include "openvino/pass/manager.hpp"
// #include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<MatMulDecompressionShapeParams,
        ov::test::ElementType,      // weights precision
        ov::test::ElementType,      // decompression precision
        ov::test::ElementType,      // scale precision
        bool,                       // transpose on weights
        DecompressionSubtractType,  // decompression subtract type
        bool,                       // reshape on decompression constants
        ov::AnyMap,                 // additional config
        fusingSpecificParams,
        bool>                      // should use decompression implementation
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
                                            const DecompressionSubtractType decompression_subtract_type,
                                            const bool reshape_on_decompression);

    void SetUp() override;

    void check_results();
};

}  // namespace test
}  // namespace ov

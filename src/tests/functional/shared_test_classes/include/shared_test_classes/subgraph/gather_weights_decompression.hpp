// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"

namespace ov {
namespace test {

class GatherWeightsDecompressionBase : virtual public ov::test::SubgraphBaseTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override;
    void check_results(const ov::element::Type& weights_precision, const size_t& num_exec_ops_expect);
};

/*
 *                        Subtract_const(U8/NF4/U4/I4)
 *                             /
 *    Weights(U8/NF4/U4/I4)  Convert(F32)
 *       |                 /
 *    Convert(F32)   Reshape(optional)
 *            \        /       Multiply_const(F32)
 *            Subtract(optional)     /
 *                  \       Reshape(optional)
 *                   \       /
 *    Indices(I32)    Multiply
 *            \     /
 *             Gather
 */

using GatherWeightsDecompressionParams = std::tuple<std::string,        // Device name
                                                    GatherDecompressionShapeParams,
                                                    ov::element::Type,  // data type
                                                    ov::element::Type,  // output type
                                                    bool,               // decompression subtract
                                                    bool,               // reshape on decompression constants
                                                    bool,               // per-tensor scale
                                                    bool>;              // per-tensor zero-point

class GatherWeightsDecompression : public testing::WithParamInterface<GatherWeightsDecompressionParams>,
                                   virtual public ov::test::GatherWeightsDecompressionBase {
public:
    static std::string get_test_case_name(testing::TestParamInfo<GatherWeightsDecompressionParams> obj);

protected:
    std::shared_ptr<ov::Model> init_subgraph(const ov::Shape& data_shape,
                                             const ov::PartialShape& indices_shape,
                                             const int axis,
                                             const int64_t batch_dims,
                                             const int group_size,
                                             const ov::element::Type data_precision,
                                             const ov::element::Type output_precision,
                                             const bool add_subtract,
                                             const bool reshape_on_decompression,
                                             const bool per_tensor_zp,
                                             const bool per_tensor_scale);
    void check_results();
    void SetUp() override;
};

/*
 *    Weights(FP16/BF16)
 *          |
 *    Convert(F32) Indices(I32)
 *          \      /
 *           Gather
 */
using GatherWeightsDecompressionWithoutScaleParams = std::tuple<std::string,  // Device name
                                                                GatherDecompressionShapeParams,
                                                                ov::element::Type,   // data type
                                                                ov::element::Type>;  // output type

class GatherWeightsDecompressionWithoutScale
    : public testing::WithParamInterface<GatherWeightsDecompressionWithoutScaleParams>,
      virtual public ov::test::GatherWeightsDecompressionBase {
public:
    static std::string get_test_case_name(testing::TestParamInfo<GatherWeightsDecompressionWithoutScaleParams> obj);

protected:
    std::shared_ptr<ov::Model> init_subgraph(const ov::Shape& data_shape,
                                             const ov::PartialShape& indices_shape,
                                             const int axis,
                                             const int64_t batch_dims,
                                             const ov::element::Type data_precision,
                                             const ov::element::Type output_precision);
    void check_results();
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
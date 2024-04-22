// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

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

struct GWDShapeParams {
    GWDShapeParams() = default;
    GWDShapeParams(ov::Shape data_shape,
                InputShape indices_shape,
                int axis,
                int64_t batch_dims,
                int decompression_group_size = -1)
        : data_shape(std::move(data_shape)),
          indices_shape(std::move(indices_shape)),
          axis(axis),
          batch_dims(batch_dims),
          decompression_group_size(decompression_group_size) {}

    ov::Shape data_shape;
    InputShape indices_shape;
    int axis;
    int64_t batch_dims;
    // Decompression group size. If the value is equal to -1, ordinary decompression is used
    int decompression_group_size;
};

using GatherWeightsDecompressionParams = std::tuple<std::string,        // Device name
                                                    GWDShapeParams,     // input shapes
                                                    ov::element::Type,  // data type
                                                    ov::element::Type,  // output type
                                                    bool,               // decompression subtract
                                                    bool,               // reshape on decompression constants
                                                    bool>;              // per-tensor zero-point

class GatherWeightsDecompression : public testing::WithParamInterface<GatherWeightsDecompressionParams>,
                                   virtual public ov::test::SubgraphBaseTest {
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
                                             const bool per_tensor_zp);
    std::shared_ptr<ov::Node> init_compressed_weights_subgraph(const ov::Shape& data_shape,
                                                               const int group_size,
                                                               const ov::element::Type data_precision,
                                                               const ov::element::Type output_precision,
                                                               const bool add_subtract,
                                                               const bool reshape_on_decompression_constant,
                                                               const bool per_tensor_zp);
    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override;
    void check_results();
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
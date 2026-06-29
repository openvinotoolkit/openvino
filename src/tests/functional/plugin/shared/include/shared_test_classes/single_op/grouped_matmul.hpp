// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"

namespace ov {
namespace test {

// Per-iteration token allocation for the 2D×3D case.
// tokens_per_expert[iter][g] = tokens routed to expert g in iteration iter.
// Sum over g must equal T for that iteration. Empty vector → even distribution.
using TokensPerExpert = std::vector<std::vector<size_t>>;

/// Bundles the shape-related parameters that are always specified together.
/// rank of a_input_shape.first selects the case:
///   rank 2 → 2D×3D: A:[T,K] x B:[G,N,K] → [T,N]    (with runtime offsets)
///   rank 3 → 3D×3D: A:[G,M,K] x B:[G,N,K] → [G,M,N] (no offsets)
struct GroupedMatMulShapeParams {
    ov::test::InputShape a_input_shape;  ///< Dynamic shape + static test shapes
    ov::Shape            b_shape;        ///< Pre-transposed weight shape [G, N, K]
    TokensPerExpert      tokens_per_expert;  ///< Per-iter routing; empty = even
};

class GroupedMatMulTestBase : virtual public ov::test::SubgraphBaseTest {
protected:
    GroupedMatMulShapeParams shape_params_;
    ov::element::Type act_type_;
    std::string model_name_;

    void SetUp() override;
    virtual std::shared_ptr<ov::Node> build_weights() = 0;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

using GroupedMatMulParams = std::tuple<
    GroupedMatMulShapeParams,  // shape bundle
    ov::element::Type,         // element type
    std::string                // target device
>;

class GroupedMatMulLayerTest : public testing::WithParamInterface<GroupedMatMulParams>,
                               virtual public GroupedMatMulTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupedMatMulParams>& obj);

protected:
    void SetUp() override;
    std::shared_ptr<ov::Node> build_weights() override;
};

using GroupedMatMulCompressedParams = std::tuple<
    GroupedMatMulShapeParams,                // shape bundle
    ov::element::Type,                       // activation element type
    ov::element::Type,                       // weights (compressed) precision
    ov::element::Type,                       // decompression precision
    ov::element::Type,                       // scale precision
    ov::test::utils::DecompressionType,      // multiply type (full/empty/scalar)
    ov::test::utils::DecompressionType,      // subtract type (full/empty/scalar)
    bool,                                    // reshape_on_decompression
    int,                                     // decompression_group_size (-1 = per-OC)
    std::string                              // target device
>;

class GroupedMatMulCompressedLayerTest
    : public testing::WithParamInterface<GroupedMatMulCompressedParams>,
      virtual public GroupedMatMulTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupedMatMulCompressedParams>& obj);

protected:
    void SetUp() override;
    std::shared_ptr<ov::Node> build_weights() override;

private:
    ov::element::Type weights_prec_;
    ov::element::Type decomp_prec_;
    ov::element::Type scale_prec_;
    ov::test::utils::DecompressionType multiply_type_;
    ov::test::utils::DecompressionType subtract_type_;
    bool reshape_on_decomp_;
    int group_size_;
};

}  // namespace test
}  // namespace ov

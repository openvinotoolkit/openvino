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
// tokens_per_expert[iter][g] = number of rows (tokens) assigned to expert g in iteration iter.
// The values are converted to cumulative end-offsets passed to GroupedMatMul:
//   offsets[g] = sum(tokens_per_expert[iter][0..g])  (exclusive row end for expert g).
// Sum over g must equal T (total token count) for that iteration.
// Zeros mean the expert is inactive (gets no tokens) — exercises sparse MoE routing.
// Empty vector → even distribution across experts.
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
    // ov op type name expected to appear in the executed network (via ProfilingInfo::node_type).
    // Not a plugin specific primitive.
    // Empty = skip the check.
    std::string expected_primitive_;

    void SetUp() override;
    void validate() override;
    virtual std::shared_ptr<ov::Node> build_weights() = 0;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

/// Builds plain (uncompressed) weights in the activation precision. Not parameterized, so that
/// both the shared and the plugin specific tests can reuse it.
class GroupedMatMulPlainBase : virtual public GroupedMatMulTestBase {
protected:
    std::shared_ptr<ov::Node> build_weights() override;
};

/// Builds weights as a decompression subgraph. Not parameterized, so that both the shared and the
/// plugin specific tests can reuse it. Derived classes fill the m_* members in their SetUp().
class GroupedMatMulCompressedBase : virtual public GroupedMatMulTestBase {
protected:
    std::shared_ptr<ov::Node> build_weights() override;
    // Dequantization adds rounding error on top of the uncompressed path. Call after
    // GroupedMatMulTestBase::SetUp().
    void set_compressed_thresholds();

    ov::element::Type m_weights_prec;
    ov::element::Type m_decomp_prec;
    ov::element::Type m_scale_prec;
    ov::test::utils::DecompressionType m_multiply_type = ov::test::utils::DecompressionType::full;
    ov::test::utils::DecompressionType m_subtract_type = ov::test::utils::DecompressionType::full;
    bool m_reshape_on_decomp = false;
    int m_group_size = -1;
};

using GroupedMatMulParams = std::tuple<
    GroupedMatMulShapeParams,  // shape bundle
    ov::element::Type,         // element type
    std::string,               // target device
    std::string                // expected ov op type in the executed network (empty = skip check)
>;

class GroupedMatMulLayerTest : public testing::WithParamInterface<GroupedMatMulParams>,
                               virtual public GroupedMatMulPlainBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupedMatMulParams>& obj);

protected:
    void SetUp() override;
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
    std::string,                             // target device
    std::string,                             // expected ov op type in the executed network (empty = skip check)
    ov::AnyMap                               // additional plugin configuration (e.g. dynamic_quantization_group_size)
>;

class GroupedMatMulCompressedLayerTest
    : public testing::WithParamInterface<GroupedMatMulCompressedParams>,
      virtual public GroupedMatMulCompressedBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupedMatMulCompressedParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

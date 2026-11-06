// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/gated_residual.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

// Describes how w_down/w_up/(w_inject) are compressed. `compressed == false` ignores every other
// field and the weights are generated directly at `GatedResidualLayerParams`'s precision. Scales are [G, N]
// floating-point tensors (f16 / bf16 / f32); zero points, when present, are [G, N] tensors of the weights' type.
struct GatedResidualWeightCompression {
    bool compressed = false;
    ov::element::Type weights_precision = ov::element::dynamic;
    bool has_zero_points = false;
    size_t group_size = 0;  // 0 == per-output-channel (G = 1, K/G = K)
    ov::element::Type scale_precision = ov::element::f16;
};

// Bundles the shape dimensions with precision/compression because they aren't freely combinable:
// a compressed weight's group size must evenly divide hc_count*hidden_size and lowrank (see
// isSupportedOperation()'s group-size check in gated_residual.cpp), and each compression spec below
// is deliberately paired with one precision for coverage diversity rather than tested against all of
// them. Keeping them together lets instance files describe a test case as a single ValuesIn axis
// instead of a ::testing::Combine that would otherwise generate invalid combinations.
struct GatedResidualShapeParams {
    size_t hc_count;     // C
    size_t hidden_size;  // D
    size_t lowrank;      // L
    ov::element::Type precision;
    GatedResidualWeightCompression compression{};
};

using GatedResidualLayerParams = std::tuple<ov::op::internal::GatedResidual::Mode,
                                            GatedResidualShapeParams,
                                            bool,  // produce_write_gate
                                            bool,  // norm_gain_plus_one
                                            // Leading [B, T] dims of every data-path input (residual, sublayer_y,
                                            // write_gate): `first` is the dynamic partial shape (must be dynamic),
                                            // `second` the static [B, T] used for each successive inference. The
                                            // fixture appends each input's last dim (C*D, D or C).
                                            ov::test::InputShape,
                                            std::string>;  // target device

class GatedResidualLayerCPUTest : public testing::WithParamInterface<GatedResidualLayerParams>,
                                  virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatedResidualLayerParams>& obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    std::vector<ov::Tensor> calculate_refs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void SetUp() override;

private:
    ov::op::internal::GatedResidual::Mode m_mode = ov::op::internal::GatedResidual::Mode::READ;
    size_t m_hc_count = 0;
    size_t m_hidden_size = 0;
    size_t m_lowrank = 0;
    bool m_produce_write_gate = false;
    bool m_norm_gain_plus_one = true;
    float m_epsilon = 1e-6F;
    ov::element::Type m_precision = ov::element::f32;
    GatedResidualWeightCompression m_compression{};

    // Dequantized (for the compressed case) or precision-upconverted (for the uncompressed case)
    // float weight matrices, cached at generate_inputs() time (compressed: already cached by SetUp(),
    // see below) so calculate_refs() never needs to re-read/unpack the actual graph input tensors
    // (which may be packed sub-byte storage).
    std::vector<float> m_down_weight_f32;    // [L, CD]
    std::vector<float> m_up_weight_f32;      // [CD, L]
    std::vector<float> m_inject_weight_f32;  // [C, CD], only when m_produce_write_gate
};

}  // namespace ov::test

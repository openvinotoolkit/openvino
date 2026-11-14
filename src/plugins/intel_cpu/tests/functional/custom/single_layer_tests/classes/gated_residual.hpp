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

// Bundles the shape dimensions with precision and compression because they aren't freely combinable: a compressed
// weight's group size must divide hc_count*hidden_size and lowrank, and each compression spec is paired with one
// precision for coverage diversity rather than tested against all of them. A test case is then a single ValuesIn axis
// instead of a ::testing::Combine that would generate invalid combinations.
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

    // Dequantized (compressed) or upconverted (uncompressed) f32 weight matrices, cached by SetUp() so calculate_refs()
    // never has to unpack the graph's (possibly sub-byte) constants.
    std::vector<float> m_down_weight_f32;    // [L, CD]
    std::vector<float> m_up_weight_f32;      // [CD, L]
    std::vector<float> m_inject_weight_f32;  // [C, CD], only when m_produce_write_gate
};

}  // namespace ov::test

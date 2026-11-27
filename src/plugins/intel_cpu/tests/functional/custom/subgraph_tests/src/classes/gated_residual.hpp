// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "openvino/core/any.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/gated_residual.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

// How w_down/w_up/(w_inject) are compressed. `compressed == false` ignores the other fields (plain constants of the
// case's precision). Compressed weights are NNCF-style decompression subgraphs: scales of `scale_precision`, zero
// points of the weight type, one group per `group_size` input channels.
struct GatedResidualWeightCompression {
    bool compressed = false;
    ov::element::Type weights_precision = ov::element::dynamic;
    bool has_zero_points = false;
    size_t group_size = 0;  // 0 == per-output-channel (G = 1, K/G = K)
    ov::element::Type scale_precision = ov::element::f16;
    // One zero point for the whole tensor (needs has_zero_points); the pass keeps it as it is.
    bool scalar_zero_point = false;
};

// Shape, precision and compression are bundled because they are not freely combinable: a group size must divide C*D and
// L, and each compression spec is paired with one precision for coverage. A case is then one ValuesIn axis.
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
                                            // Leading [B, T] dims of the data-path inputs: `first` is the dynamic
                                            // partial shape, `second` the static [B, T] per inference. The fixture
                                            // appends each input's last dim (C*D, D or C).
                                            ov::test::InputShape,
                                            ov::AnyMap,    // plugin configuration
                                            std::string>;  // target device

class GatedResidualLayerCPUTest : public testing::WithParamInterface<GatedResidualLayerParams>,
                                  virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatedResidualLayerParams>& obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void SetUp() override;

private:
    ov::op::internal::GatedResidual::Mode m_mode = ov::op::internal::GatedResidual::Mode::READ;
    size_t m_hc_count = 0;
    size_t m_hidden_size = 0;
    size_t m_lowrank = 0;
    bool m_produce_write_gate = false;
    float m_epsilon = 1e-6F;
    ov::element::Type m_precision = ov::element::f32;
    GatedResidualWeightCompression m_compression{};
};

}  // namespace ov::test

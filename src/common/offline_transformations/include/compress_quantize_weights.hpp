// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace pass {

class CompressQuantizeWeights;
class CompressWeightsWithFakeQuantize;
class CompressWeightsWithFakeConvert;

}  // namespace pass
}  // namespace ov

/*
    CompressWeightsWithFakeQuantize transformation goal is to pre-quantize data to minimize runtime calculations with
   constant data. To achieve this goal we perform FakeQuantize decomposition to separate quantization from
   dequantization in it.

    Initial graph (FakeQuantize where all inputs are Constants):

                                   |  |  |  |  |
                                   |  |  |  |  |
                                   v  v  v  v  v
                                  +------------+
                                  |FakeQuantize|
                                  +------------+
                                        |
                                        v

    is replaced to:
                                +-----------------+
                                |    Constant     |
                                | (low precision) |
                                +-----------------+
                                        |
                                        v
                                +------------------+
                                |     Convert      |
                                |  (to high prec)  |
                                +------------------+
                                        |
                                        v
                  +----------+    +------------+
                  |zero point|--->|  Subtract  |
                  +----------+    +-----+------+
                                        |
                                        v
                   +---------+    +------------+
                   |  scale  |--->|  Multiply  |
                   +---------+    +-----+------+
                                        |
                                        v

    Transformation prepares quantized constant data for Low Precision pipeline.
    Such constant data packing reduces IR size (.bin file size) in offline transformations.
    With that we can skip same calculations in the runtime and make loading of such sub-graphs to the plugin faster.
    Additionally zero point can be fused to weights if it doesn't affect accuracy.
*/
class ov::pass::CompressWeightsWithFakeQuantize : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("CompressWeightsWithFakeQuantize");

    CompressWeightsWithFakeQuantize();
};

/*
    CompressWeightsWithFakeConvert replaces FakeConvert node with constant inputs to the following subgraph:

            +----------+
            | Constant |
            | (float8) }
            +----+-----+
                 |
                 v
            +----------+
            | Convert  |
            | (float32)|
            +----+-----+
                 |
                 v
            +----------+     +--------+
            | Subtract |<----| -shift |
            +----+-----+     +--------+
                 |
                 v
            +----------+     +---------+
            | Multiply |<----| 1/scale |
            +----------+     +---------+

*/
class ov::pass::CompressWeightsWithFakeConvert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("CompressWeightsWithFakeConvert");

    CompressWeightsWithFakeConvert();
};

class ov::pass::CompressQuantizeWeights : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("CompressQuantizeWeights");
    CompressQuantizeWeights();
};

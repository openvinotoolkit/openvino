// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/node.hpp"
#include "tp_transformations/annotations.hpp"

namespace ov {
namespace tp {
namespace blocks {

/// Ops a projection's result may pass through on its way to attention or to the next projection.
bool is_transparent(const ov::Node* node);

/// A MatMul whose second input is entirely constant-derived: a linear layer.
bool is_linear_layer(const ov::Node* node);

/// The linear layers reached walking back from `output` through transparent ops.
std::vector<std::shared_ptr<ov::Node>> nearest_producer_matmuls(const ov::Output<ov::Node>& output);

/// The closest linear layer reached walking forward from `start` through transparent ops.
std::shared_ptr<ov::Node> nearest_consumer_matmul(const std::shared_ptr<ov::Node>& start);

struct BiasAdd {
    std::shared_ptr<ov::Node> node;  ///< nullptr when the projection has no bias
    size_t port = 0;                 ///< input index of the bias on `node`
};

/// `Add(matmul, constant subgraph)` on a projection, if there is one.
BiasAdd find_bias_add(const std::shared_ptr<ov::Node>& matmul);

/// The bias Add when there is one, the MatMul otherwise.
ov::Output<ov::Node> projection_output(const std::shared_ptr<ov::Node>& matmul);

struct WeightAxes {
    int64_t out_features;
    int64_t in_features;
};

/// Throws for a transposed activation or a weight that is not 2D.
WeightAxes weight_axes(const std::shared_ptr<ov::Node>& matmul);

/// Static size of a weight axis, 0 when dynamic.
int64_t weight_dim(const std::shared_ptr<ov::Node>& matmul, int64_t axis);

/// Head count and head size from the Reshape [.., heads, head_dim] after a projection output.
bool read_head_layout(const ov::Output<ov::Node>& projection, int64_t& heads, int64_t& head_dim);

/// Coarsest step a split along `axis` of the decompression chain at `output` has to respect
/// (quantization groups), or 0 when the chain cannot be cut at all.
int64_t chain_granularity(const ov::Output<ov::Node>& output, int64_t axis, int64_t full_size);

/// Why the chain feeding `input` cannot be cut at every rank's boundary along `axis` at compile
/// time; empty when it can.
std::string chain_cut_problem(ov::Input<ov::Node> input,
                              int64_t axis,
                              const std::vector<Segment>& segments,
                              size_t world_size);

/// Annotates the constants of the chain feeding `input` for a split of its `axis` (`full_size`
/// elements) into `segments`. Throws, saying why, when the chain cannot be cut at every rank's
/// boundary, and on a weight of 1, 2, 3 or 6 bits.
void annotate_chain(ov::Input<ov::Node> input,
                    int64_t axis,
                    const std::vector<Segment>& segments,
                    ShardStrategy strategy,
                    size_t world_size,
                    const std::string& block);

/// Annotates the role of a projection, its weight chain and, column-parallel, its bias.
void annotate_projection(const std::shared_ptr<ov::Node>& matmul,
                         ShardStrategy strategy,
                         const std::vector<Segment>& segments,
                         Collective collective,
                         size_t world_size,
                         const std::string& block);

/// Annotates the listed elements of the shape constant feeding `node`'s `port`, if it is one.
bool annotate_shape_input(const std::shared_ptr<ov::Node>& node,
                          size_t port,
                          std::vector<size_t> elements,
                          const std::vector<Segment>& segments,
                          const std::string& block,
                          ShardAnnotation::Elements mode = ShardAnnotation::Elements::SIZE);

/// The parts of a fused projection's output split along its last axis, in order: the outputs of a
/// Split / VariadicSplit, or of Slices with constant bounds. Empty when the output is not split so.
std::vector<ov::Output<ov::Node>> feature_parts(const ov::Output<ov::Node>& output);

/// Gives the split of `output` (see feature_parts) the rank's lengths or bounds of `segments`, one per part.
void annotate_feature_parts(const ov::Output<ov::Node>& output, const std::vector<Segment>& segments, const std::string& block);

}  // namespace blocks
}  // namespace tp
}  // namespace ov

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_replicated_branches.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {

// The matcher recognizes an N-way replicated camera pipeline: N static
// branches, each reducing a shared 5D [B, cameras, C, H, W]-style tensor via
// a single Gather, that feed into one Concat. Neither N, the spatial/channel
// sizes, nor the concat axis are fixed: they are derived from the matched
// graph so the pass generalizes across models instead of one fixed topology.
// `kBranchRank` reflects the NCHW-style 4D tensors the supported op set
// below (Convolution/Add/Relu/MaxPool/Convert) is validated against.
constexpr size_t kBranchRank = 4;
constexpr size_t kMinReplicaCount = 2;
constexpr char kRequiresPrecisionConversionKey[] = "requires_precision_conversion_0";

using NodePtr = std::shared_ptr<ov::Node>;
using NodeMap = std::unordered_map<const ov::Node*, NodePtr>;
using NodeSet = std::unordered_set<ov::Node*>;

bool is_static_node(const NodePtr& node) {
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (!node->get_output_partial_shape(i).is_static()) {
            return false;
        }
    }
    return true;
}

bool is_constant(const NodePtr& node) {
    return ov::is_type<ov::op::v0::Constant>(node);
}

bool is_constant_derived_convert(const NodePtr& node) {
    const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node);
    return convert && convert->get_input_size() == 1 &&
           is_constant(convert->input_value(0).get_node_shared_ptr());
}

bool is_batch_safe_add(const NodePtr& node, size_t replica_count) {
    const auto add = ov::as_type_ptr<ov::op::v1::Add>(node);
    if (!add || add->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY ||
        add->get_input_size() != 2 || add->get_output_size() == 0) {
        return false;
    }

    const auto output_shape = add->get_output_partial_shape(0);
    if (!output_shape.is_static() || output_shape.rank() != static_cast<int64_t>(kBranchRank) ||
        output_shape[0] != 1) {
        return false;
    }

    ov::Shape expected_batched_shape = output_shape.to_shape();
    expected_batched_shape[0] = replica_count;

    ov::PartialShape merged_shape;
    for (size_t i = 0; i < add->get_input_size(); ++i) {
        auto input_shape = add->get_input_partial_shape(i);
        if (!input_shape.is_static()) {
            return false;
        }

        // Only constants are shared by all replicas. Every other input must be
        // a branch value of the same rank whose original batch dimension is
        // the one changed by this pass. This deliberately excludes shapes for
        // which the new batch behavior cannot be proven from the original
        // graph.
        if (!is_constant(add->input_value(i).get_node_shared_ptr())) {
            if (input_shape.rank() != static_cast<int64_t>(kBranchRank) || input_shape[0] != 1) {
                return false;
            }
            auto batched_input_shape = input_shape.to_shape();
            batched_input_shape[0] = replica_count;
            input_shape = batched_input_shape;
        }

        if (i == 0) {
            merged_shape = input_shape;
        } else if (!ov::PartialShape::broadcast_merge_into(merged_shape,
                                                            input_shape,
                                                            ov::op::AutoBroadcastType::NUMPY)) {
            return false;
        }
    }

    return merged_shape == ov::PartialShape(expected_batched_shape);
}

bool constants_equal(const NodePtr& lhs, const NodePtr& rhs) {
    const auto lhs_constant = ov::as_type_ptr<ov::op::v0::Constant>(lhs);
    const auto rhs_constant = ov::as_type_ptr<ov::op::v0::Constant>(rhs);
    if (!lhs_constant || !rhs_constant || lhs_constant->get_element_type() != rhs_constant->get_element_type() ||
        lhs_constant->get_shape() != rhs_constant->get_shape() ||
        lhs_constant->get_byte_size() != rhs_constant->get_byte_size()) {
        return false;
    }

    return lhs_constant->get_byte_size() == 0 ||
           std::memcmp(lhs_constant->get_data_ptr(),
                       rhs_constant->get_data_ptr(),
                       lhs_constant->get_byte_size()) == 0;
}

bool convolution_attributes_equal(const ov::op::v1::Convolution& lhs, const ov::op::v1::Convolution& rhs) {
    return lhs.get_strides() == rhs.get_strides() && lhs.get_pads_begin() == rhs.get_pads_begin() &&
           lhs.get_pads_end() == rhs.get_pads_end() && lhs.get_dilations() == rhs.get_dilations() &&
           lhs.get_auto_pad() == rhs.get_auto_pad();
}

bool max_pool_attributes_equal(const ov::op::v1::MaxPool& lhs, const ov::op::v1::MaxPool& rhs) {
    return lhs.get_strides() == rhs.get_strides() && lhs.get_pads_begin() == rhs.get_pads_begin() &&
           lhs.get_pads_end() == rhs.get_pads_end() && lhs.get_kernel() == rhs.get_kernel() &&
           lhs.get_rounding_type() == rhs.get_rounding_type() && lhs.get_auto_pad() == rhs.get_auto_pad();
}

template <typename MaxPool>
bool max_pool_attributes_equal(const MaxPool& lhs, const MaxPool& rhs) {
    return lhs.get_strides() == rhs.get_strides() && lhs.get_pads_begin() == rhs.get_pads_begin() &&
           lhs.get_pads_end() == rhs.get_pads_end() && lhs.get_kernel() == rhs.get_kernel() &&
           lhs.get_rounding_type() == rhs.get_rounding_type() && lhs.get_auto_pad() == rhs.get_auto_pad() &&
           lhs.get_dilations() == rhs.get_dilations() &&
           lhs.get_index_element_type() == rhs.get_index_element_type() && lhs.get_axis() == rhs.get_axis();
}

bool operation_attributes_equal(const NodePtr& lhs, const NodePtr& rhs) {
    if (ov::is_type<ov::op::v1::Convolution>(lhs)) {
        const auto lhs_conv = ov::as_type_ptr<ov::op::v1::Convolution>(lhs);
        const auto rhs_conv = ov::as_type_ptr<ov::op::v1::Convolution>(rhs);
        return rhs_conv && convolution_attributes_equal(*lhs_conv, *rhs_conv);
    }
    if (ov::is_type<ov::op::v1::MaxPool>(lhs)) {
        const auto lhs_pool = ov::as_type_ptr<ov::op::v1::MaxPool>(lhs);
        const auto rhs_pool = ov::as_type_ptr<ov::op::v1::MaxPool>(rhs);
        return rhs_pool && max_pool_attributes_equal(*lhs_pool, *rhs_pool);
    }
    if (ov::is_type<ov::op::v8::MaxPool>(lhs)) {
        const auto lhs_pool = ov::as_type_ptr<ov::op::v8::MaxPool>(lhs);
        const auto rhs_pool = ov::as_type_ptr<ov::op::v8::MaxPool>(rhs);
        return rhs_pool && max_pool_attributes_equal(*lhs_pool, *rhs_pool);
    }
    if (ov::is_type<ov::op::v14::MaxPool>(lhs)) {
        const auto lhs_pool = ov::as_type_ptr<ov::op::v14::MaxPool>(lhs);
        const auto rhs_pool = ov::as_type_ptr<ov::op::v14::MaxPool>(rhs);
        return rhs_pool && max_pool_attributes_equal(*lhs_pool, *rhs_pool);
    }
    if (ov::is_type<ov::op::v0::Convert>(lhs)) {
        const auto lhs_convert = ov::as_type_ptr<ov::op::v0::Convert>(lhs);
        const auto rhs_convert = ov::as_type_ptr<ov::op::v0::Convert>(rhs);
        return rhs_convert && lhs_convert->get_destination_type() == rhs_convert->get_destination_type();
    }
    return true;
}

bool is_batch_safe_operation(const NodePtr& node, size_t replica_count) {
    return is_constant(node) || ov::is_type<ov::op::v1::Convolution>(node) ||
           is_batch_safe_add(node, replica_count) || ov::is_type<ov::op::v0::Relu>(node) ||
           ov::is_type<ov::op::v1::MaxPool>(node) || ov::is_type<ov::op::v8::MaxPool>(node) ||
           ov::is_type<ov::op::v14::MaxPool>(node) || ov::is_type<ov::op::v0::Convert>(node);
}

bool scalar_i64_constant(const ov::Output<ov::Node>& value, int64_t expected) {
    const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(value.get_node_shared_ptr());
    if (!constant || constant->get_element_type() != ov::element::i64 ||
        ov::shape_size(constant->get_shape()) != 1) {
        return false;
    }
    const auto values = constant->cast_vector<int64_t>();
    return values.size() == 1 && values[0] == expected;
}

bool output_shapes_equal(const NodePtr& lhs, const NodePtr& rhs) {
    if (lhs->get_output_size() != rhs->get_output_size()) {
        return false;
    }
    for (size_t i = 0; i < lhs->get_output_size(); ++i) {
        if (lhs->get_output_element_type(i) != rhs->get_output_element_type(i) ||
            lhs->get_output_partial_shape(i) != rhs->get_output_partial_shape(i)) {
            return false;
        }
    }
    return true;
}

bool gather_attributes_equal(const std::shared_ptr<ov::op::v8::Gather>& lhs,
                             const std::shared_ptr<ov::op::v8::Gather>& rhs) {
    // The index input is intentionally not compared here: its values are
    // checked against the final Concat input position by the caller. The
    // axis input only needs to be self-consistent with each Gather's own
    // normalized axis, not a global constant, so the pass is not tied to one
    // fixed camera-axis position.
    return lhs->get_axis() == rhs->get_axis() && lhs->get_batch_dims() == rhs->get_batch_dims() &&
           lhs->input_value(0) == rhs->input_value(0) &&
           scalar_i64_constant(lhs->input_value(2), lhs->get_axis()) &&
           scalar_i64_constant(rhs->input_value(2), rhs->get_axis()) && output_shapes_equal(lhs, rhs);
}

void collect_branch_nodes(const NodePtr& node, NodeSet& nodes, NodeSet& visiting) {
    if (!node || nodes.count(node.get()) != 0) {
        return;
    }
    OPENVINO_ASSERT(visiting.count(node.get()) == 0, "Camera branch unexpectedly contains a cycle");
    visiting.insert(node.get());
    nodes.insert(node.get());

    // Gather is the semantic boundary. Its data input is the shared camera
    // tensor and its index/axis inputs are not part of a replica branch.
    if (!ov::is_type<ov::op::v8::Gather>(node)) {
        for (const auto& input : node->input_values()) {
            collect_branch_nodes(input.get_node_shared_ptr(), nodes, visiting);
        }
    }
    visiting.erase(node.get());
}

bool equivalent_supported_rt_value(const std::string& key, const ov::Any& lhs, const ov::Any& rhs) {
    if (lhs.empty() || rhs.empty()) {
        return lhs.empty() && rhs.empty();
    }
    if (lhs.type_info() != rhs.type_info()) {
        return false;
    }

    // Decompression is a marker attribute with no payload.  It is
    // intentionally non-copyable through ov::copy_runtime_info(), so it
    // needs an explicit, type-safe equivalence rule here.
    if (lhs.is<ov::Decompression>()) {
        return rhs.is<ov::Decompression>();
    }
    if (lhs.is<ov::WeightlessCacheAttribute>()) {
        const auto& lhs_attr = lhs.as<ov::WeightlessCacheAttribute>();
        const auto& rhs_attr = rhs.as<ov::WeightlessCacheAttribute>();
        return lhs_attr.original_size == rhs_attr.original_size && lhs_attr.bin_offset == rhs_attr.bin_offset &&
               lhs_attr.original_dtype == rhs_attr.original_dtype;
    }
    if (key == kRequiresPrecisionConversionKey) {
        // ConstantFolding uses this payload-free marker internally. It has
        // no value to compare, but must survive cloning of an equivalent op.
        return true;
    }
    if (lhs.is<ov::FusedNames>()) {
        // FusedNames records provenance only. Replica names are expected to
        // differ, while the reference node's provenance is retained below.
        return rhs.is<ov::FusedNames>();
    }

    if (lhs.is<bool>()) {
        return lhs.as<bool>() == rhs.as<bool>();
    }
    if (lhs.is<int8_t>()) {
        return lhs.as<int8_t>() == rhs.as<int8_t>();
    }
    if (lhs.is<int16_t>()) {
        return lhs.as<int16_t>() == rhs.as<int16_t>();
    }
    if (lhs.is<int32_t>()) {
        return lhs.as<int32_t>() == rhs.as<int32_t>();
    }
    if (lhs.is<int64_t>()) {
        return lhs.as<int64_t>() == rhs.as<int64_t>();
    }
    if (lhs.is<uint8_t>()) {
        return lhs.as<uint8_t>() == rhs.as<uint8_t>();
    }
    if (lhs.is<uint16_t>()) {
        return lhs.as<uint16_t>() == rhs.as<uint16_t>();
    }
    if (lhs.is<uint32_t>()) {
        return lhs.as<uint32_t>() == rhs.as<uint32_t>();
    }
    if (lhs.is<uint64_t>()) {
        return lhs.as<uint64_t>() == rhs.as<uint64_t>();
    }
    if (lhs.is<float>()) {
        return lhs.as<float>() == rhs.as<float>();
    }
    if (lhs.is<double>()) {
        return lhs.as<double>() == rhs.as<double>();
    }
    if (lhs.is<std::string>()) {
        return lhs.as<std::string>() == rhs.as<std::string>();
    }

    // Runtime attributes and plugin-specific values do not have a uniform,
    // non-throwing equality contract. Reject these entries rather than
    // invoking ov::Any::operator==, which may throw for an unsupported type.
    return false;
}

bool equivalent_runtime_info(const ov::RTMap& lhs, const ov::RTMap& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (const auto& [key, lhs_value] : lhs) {
        const auto rhs_it = rhs.find(key);
        if (rhs_it == rhs.end() || !equivalent_supported_rt_value(key, lhs_value, rhs_it->second)) {
            return false;
        }
    }
    return true;
}

bool equivalent_node_metadata(const NodePtr& lhs, const NodePtr& rhs) {
    if (!equivalent_runtime_info(lhs->get_rt_info(), rhs->get_rt_info()) ||
        lhs->get_input_size() != rhs->get_input_size() ||
        lhs->get_output_size() != rhs->get_output_size()) {
        return false;
    }

    for (size_t i = 0; i < lhs->get_input_size(); ++i) {
        if (!equivalent_runtime_info(lhs->input(i).get_rt_info(), rhs->input(i).get_rt_info())) {
            return false;
        }
    }
    for (size_t i = 0; i < lhs->get_output_size(); ++i) {
        // The original branch nodes remain in place for any users other than
        // the replaced Concat. Branch-local names are therefore safe to
        // preserve on the cloned reference branch; the Concat names are
        // copied to the externally visible repacked result.
        if (!equivalent_runtime_info(lhs->output(i).get_rt_info(), rhs->output(i).get_rt_info())) {
            return false;
        }
    }
    return true;
}

bool equivalent_branch_nodes(const NodePtr& lhs,
                             const NodePtr& rhs,
                             const std::shared_ptr<ov::op::v8::Gather>& lhs_gather,
                             const std::shared_ptr<ov::op::v8::Gather>& rhs_gather,
                             size_t replica_count,
                             NodeMap& lhs_to_rhs,
                             NodeMap& rhs_to_lhs) {
    // The transformed graph retains the first branch's nodes and metadata.
    // Reject a match when a corresponding branch carries different metadata
    // instead of silently dropping it.
    if (!equivalent_node_metadata(lhs, rhs)) {
        return false;
    }

    if (lhs == lhs_gather || rhs == rhs_gather) {
        return lhs == lhs_gather && rhs == rhs_gather && gather_attributes_equal(lhs_gather, rhs_gather);
    }

    if (is_constant(lhs) || is_constant(rhs)) {
        return is_constant(lhs) && is_constant(rhs) && constants_equal(lhs, rhs);
    }
    if (!is_batch_safe_operation(lhs, replica_count) || !is_batch_safe_operation(rhs, replica_count)) {
        return false;
    }
    if (lhs->get_type_info() != rhs->get_type_info() || !is_static_node(lhs) || !is_static_node(rhs) ||
        !output_shapes_equal(lhs, rhs) || !operation_attributes_equal(lhs, rhs) ||
        lhs->get_input_size() != rhs->get_input_size()) {
        return false;
    }

    const auto lhs_it = lhs_to_rhs.find(lhs.get());
    const auto rhs_it = rhs_to_lhs.find(rhs.get());
    if (lhs_it != lhs_to_rhs.end() || rhs_it != rhs_to_lhs.end()) {
        return lhs_it != lhs_to_rhs.end() && rhs_it != rhs_to_lhs.end() && lhs_it->second == rhs &&
               rhs_it->second == lhs;
    }

    lhs_to_rhs.emplace(lhs.get(), rhs);
    rhs_to_lhs.emplace(rhs.get(), lhs);
    for (size_t i = 0; i < lhs->get_input_size(); ++i) {
        if (!equivalent_branch_nodes(lhs->input_value(i).get_node_shared_ptr(),
                                     rhs->input_value(i).get_node_shared_ptr(),
                                     lhs_gather,
                                     rhs_gather,
                                     replica_count,
                                     lhs_to_rhs,
                                     rhs_to_lhs)) {
            return false;
        }
    }
    return true;
}

struct BranchInfo {
    std::shared_ptr<ov::op::v8::Gather> gather;
    NodeSet nodes;
};

bool get_branch_info(const ov::Output<ov::Node>& endpoint_output,
                     int64_t expected_index,
                     size_t replica_count,
                     BranchInfo& info) {
    if (endpoint_output.get_index() != 0) {
        return false;
    }
    const auto endpoint = endpoint_output.get_node_shared_ptr();
    if (!endpoint || endpoint->get_output_size() == 0 || !is_static_node(endpoint)) {
        return false;
    }

    // Every replica branch starts from an implicit batch of 1 so that the
    // replicas can later be stacked into a batch of `replica_count`. The rank
    // reflects the NCHW-style ops this pass supports, not one fixed model.
    const auto endpoint_shape = endpoint->get_output_partial_shape(0);
    if (endpoint_shape.rank() != static_cast<int64_t>(kBranchRank) || endpoint_shape[0] != 1) {
        return false;
    }

    NodeSet visiting;
    collect_branch_nodes(endpoint, info.nodes, visiting);

    std::vector<std::shared_ptr<ov::op::v8::Gather>> gathers;
    for (auto* raw_node : info.nodes) {
        const auto node = raw_node->shared_from_this();
        if (const auto gather = ov::as_type_ptr<ov::op::v8::Gather>(node)) {
            gathers.push_back(gather);
        } else if (!is_batch_safe_operation(node, replica_count)) {
            return false;
        }
    }
    if (gathers.size() != 1) {
        return false;
    }

    info.gather = gathers.front();
    if (info.gather->get_input_size() != 3 || info.gather->get_output_size() == 0 ||
        info.gather->get_batch_dims() != 0 || !scalar_i64_constant(info.gather->input_value(1), expected_index) ||
        !is_static_node(info.gather)) {
        return false;
    }

    // A scalar-indexed Gather removes exactly the axis dimension from its
    // data input. That dimension must hold exactly `replica_count` camera
    // slots (not a fixed literal), and removing it must reproduce the
    // Gather's own output shape dimension-for-dimension.
    const auto camera_input_shape = info.gather->get_input_partial_shape(0);
    const auto camera_output_shape = info.gather->get_output_partial_shape(0);
    const auto axis = info.gather->get_axis();
    if (!camera_input_shape.is_static() || !camera_output_shape.is_static() || axis < 0 ||
        camera_input_shape.rank().get_length() != camera_output_shape.rank().get_length() + 1 ||
        axis >= camera_input_shape.rank().get_length() ||
        static_cast<size_t>(camera_input_shape[axis].get_length()) != replica_count ||
        !scalar_i64_constant(info.gather->input_value(2), axis)) {
        return false;
    }
    std::vector<ov::Dimension> expected_output_dims;
    expected_output_dims.reserve(camera_output_shape.rank().get_length());
    for (int64_t i = 0; i < camera_input_shape.rank().get_length(); ++i) {
        if (i != axis) {
            expected_output_dims.push_back(camera_input_shape[i]);
        }
    }
    return ov::PartialShape(expected_output_dims) == camera_output_shape;
}

bool branch_nodes_are_disjoint(const std::vector<BranchInfo>& branches) {
    NodeSet seen;
    bool disjoint = true;
    for (const auto& branch : branches) {
        for (auto* node : branch.nodes) {
            // A conversion fed only by a constant is immutable branch data.
            // Weight sharing may legally use one such node for every replica.
            if (is_constant(node->shared_from_this()) || is_constant_derived_convert(node->shared_from_this()) ||
                node == branch.gather.get()) {
                continue;
            }
            if (!seen.insert(node).second) {
                disjoint = false;
            }
        }
    }
    return disjoint;
}

void copy_node_metadata(const NodePtr& original, const NodePtr& replacement) {
    replacement->set_friendly_name(original->get_friendly_name());
    ov::copy_runtime_info(original, replacement);
    // Decompression is non-copyable by design because it must not propagate
    // through arbitrary graph rewrites.  This pass has already established
    // that the corresponding replica operations are equivalent, so preserve
    // this known marker on the explicitly cloned operation.
    if (ov::is_decompression(original)) {
        ov::mark_as_decompression(replacement);
    }
    ov::copy_weightless_cache_attr(original, replacement);
    const auto precision_marker = original->get_rt_info().find(kRequiresPrecisionConversionKey);
    if (precision_marker != original->get_rt_info().end()) {
        replacement->get_rt_info()[kRequiresPrecisionConversionKey] = precision_marker->second;
    }
    for (size_t i = 0; i < std::min(original->get_output_size(), replacement->get_output_size()); ++i) {
        replacement->output(i).set_names(original->output(i).get_names());
    }
}

bool transform_camera_branches(const std::shared_ptr<ov::op::v0::Concat>& output_concat) {
    if (!output_concat || output_concat->get_output_size() == 0 || !is_static_node(output_concat)) {
        return false;
    }

    // The replica count, per-branch feature shape, and concat axis are all
    // derived from the matched graph below instead of being pinned to one
    // reference model's literal dimensions.
    const size_t replica_count = output_concat->get_input_size();
    if (replica_count < kMinReplicaCount) {
        return false;
    }

    // The rewrite replaces only the replica Concat. Original branch nodes
    // remain available to other consumers, so an endpoint name or an
    // intermediate branch value used elsewhere does not make this rewrite
    // unsafe.
    std::vector<BranchInfo> branches(replica_count);
    for (size_t i = 0; i < replica_count; ++i) {
        if (!get_branch_info(output_concat->input_value(i), static_cast<int64_t>(i), replica_count, branches[i])) {
            return false;
        }
    }
    if (!branch_nodes_are_disjoint(branches)) {
        return false;
    }

    const auto& reference = branches.front();
    const auto reference_endpoint = output_concat->input_value(0).get_node_shared_ptr();
    const auto reference_shape = reference_endpoint->get_output_partial_shape(0).to_shape();

    // The concat axis must select one of the genuine feature dimensions and
    // never the implicit per-branch batch dimension at axis 0, which is what
    // this pass repurposes to stack the replicas.
    const auto concat_axis = output_concat->get_axis();
    if (concat_axis <= 0 || static_cast<size_t>(concat_axis) >= reference_shape.size()) {
        return false;
    }

    for (size_t i = 1; i < replica_count; ++i) {
        NodeMap reference_to_branch;
        NodeMap branch_to_reference;
        if (!equivalent_branch_nodes(reference_endpoint,
                                     output_concat->input_value(i).get_node_shared_ptr(),
                                     reference.gather,
                                     branches[i].gather,
                                     replica_count,
                                     reference_to_branch,
                                     branch_to_reference)) {
            return false;
        }
    }

    const auto shared_camera_input = reference.gather->input_value(0);
    for (size_t i = 1; i < replica_count; ++i) {
        if (branches[i].gather->input_value(0) != shared_camera_input) {
            return false;
        }
    }

    ov::OutputVector camera_inputs;
    camera_inputs.reserve(replica_count);
    for (const auto& branch : branches) {
        camera_inputs.push_back(branch.gather->output(0));
    }
    auto batched_camera = std::make_shared<ov::op::v0::Concat>(camera_inputs, 0);
    batched_camera->set_friendly_name(output_concat->get_friendly_name() + "/camera_batch");
    ov::copy_runtime_info({output_concat, reference.gather}, batched_camera);

    NodeMap cloned_nodes;
    std::function<NodePtr(const NodePtr&)> clone_node = [&](const NodePtr& original) -> NodePtr {
        if (original == reference.gather) {
            return batched_camera;
        }
        if (is_constant(original)) {
            return original;
        }
        const auto found = cloned_nodes.find(original.get());
        if (found != cloned_nodes.end()) {
            return found->second;
        }

        ov::OutputVector new_inputs;
        new_inputs.reserve(original->get_input_size());
        for (const auto& input : original->input_values()) {
            const auto source = clone_node(input.get_node_shared_ptr());
            new_inputs.push_back(source->output(input.get_index()));
        }
        auto clone = original->clone_with_new_inputs(new_inputs);
        copy_node_metadata(original, clone);
        cloned_nodes.emplace(original.get(), clone);
        return clone;
    };

    const auto batched_branch = clone_node(reference_endpoint);
    ov::Shape expected_batched_shape = reference_shape;
    expected_batched_shape[0] = replica_count;
    if (batched_branch->get_output_partial_shape(0) != ov::PartialShape(expected_batched_shape)) {
        return false;
    }

    // Move the replica (batch) dimension to sit directly before the original
    // concat axis, then merge it with that axis. This reproduces the same
    // concatenation order the original per-branch Concat produced, for
    // whichever axis and rank the matched graph actually used.
    const size_t rank = reference_shape.size();
    std::vector<int64_t> transpose_order_values;
    transpose_order_values.reserve(rank);
    for (int64_t axis = 1; axis < concat_axis; ++axis) {
        transpose_order_values.push_back(axis);
    }
    transpose_order_values.push_back(0);
    for (int64_t axis = concat_axis; axis < static_cast<int64_t>(rank); ++axis) {
        transpose_order_values.push_back(axis);
    }

    auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{rank}, transpose_order_values);
    auto repack_transpose = std::make_shared<ov::op::v1::Transpose>(batched_branch, transpose_order);
    repack_transpose->set_friendly_name(output_concat->get_friendly_name() + "/camera_repack_transpose");
    ov::copy_runtime_info(output_concat, repack_transpose);

    std::vector<int64_t> reshape_shape_values;
    reshape_shape_values.reserve(rank);
    reshape_shape_values.push_back(1);
    for (int64_t axis = 1; axis < concat_axis; ++axis) {
        reshape_shape_values.push_back(static_cast<int64_t>(reference_shape[axis]));
    }
    reshape_shape_values.push_back(static_cast<int64_t>(replica_count * reference_shape[concat_axis]));
    for (int64_t axis = concat_axis + 1; axis < static_cast<int64_t>(rank); ++axis) {
        reshape_shape_values.push_back(static_cast<int64_t>(reference_shape[axis]));
    }

    auto reshape_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{rank}, reshape_shape_values);
    auto repacked_output = std::make_shared<ov::op::v1::Reshape>(repack_transpose, reshape_shape, false);
    copy_node_metadata(output_concat, repacked_output);
    ov::copy_runtime_info({output_concat, repack_transpose}, repacked_output);
    ov::replace_node(output_concat, repacked_output);
    return true;
}

}  // namespace

BatchReplicatedBranches::BatchReplicatedBranches() {
    using namespace ov::pass::pattern;

    auto output_concat = wrap_type<ov::op::v0::Concat>();
    auto callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& matcher) {
        const auto output_concat_node = ov::as_type_ptr<ov::op::v0::Concat>(matcher.get_match_root());
        if (!output_concat_node || transformation_callback(output_concat_node)) {
            return false;
        }
        return transform_camera_branches(output_concat_node);
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(output_concat, "BatchReplicatedBranches");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu

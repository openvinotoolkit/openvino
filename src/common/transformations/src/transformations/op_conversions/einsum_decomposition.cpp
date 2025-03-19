// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/einsum_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/einsum.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {
/// \brief      Compute einsum_path for a given Einsum node meaning that the (pseudo-)optimal
/// order of operands contraction in terms of performance and memory consumption
///
/// \param      einsum_node         An input Einsum node
///
/// \return     a vector of pairs with input indices assuming that the intermediate result is
/// appended in the tail
///
std::vector<std::pair<size_t, size_t>> compute_einsum_path(std::shared_ptr<const ov::op::v7::Einsum> einsum_node) {
    // TODO: implement algorithm for finding (pseudo-)optimal einsum_path
    std::vector<std::pair<size_t, size_t>> einsum_path;
    const size_t num_inputs = einsum_node->get_input_size();
    OPENVINO_ASSERT(num_inputs > 0);
    for (size_t input_ind = num_inputs - 1; input_ind > 0; --input_ind) {
        einsum_path.push_back(std::make_pair(0, input_ind));
    }
    return einsum_path;
}

/// \brief      Check if the dimension with a given label is reduced. The dimension is reduced
/// if the corresponding label is met in neither the output subscript nor the input subscripts
/// excluding ones specified by a vector excluded_indices
///
/// \param      input_subscripts         The vector of the input subscripts
/// \param      output_subscript         The output subscript
/// \param      label_to_check           A label that corresponds to dimension to check
/// \param      excluded_indices         A vector of input subscript indices to be excluded
///
/// \return     true - a dimension to reduce, false - otherwise
///
bool is_dimension_reduced(const std::vector<std::string>& input_subscripts,
                          const std::string& output_subscript,
                          const std::string label_to_check,
                          const std::vector<size_t>& excluded_indices) {
    for (size_t input_ind = 0; input_ind < input_subscripts.size(); ++input_ind) {
        const auto& input_subscript = input_subscripts[input_ind];
        // the subscript is checked only if its index is not in excluded indices list
        bool check_subscript =
            (std::find(excluded_indices.begin(), excluded_indices.end(), input_ind) == excluded_indices.end());
        if (check_subscript && input_subscript.find(label_to_check) != std::string::npos) {
            return false;
        }
    }
    return output_subscript.find(label_to_check) == std::string::npos;
}

/// \brief    Checks if input vector represents a range [0; n]
///
/// \param    labels_inds    Input vector to check
///
/// \return   true - the input vector is a range [0; n]; false - otherwise
///
bool is_range_0_to_n(const std::vector<int64_t>& labels_inds) {
    int64_t check_index = 0;
    for (auto index : labels_inds) {
        if (check_index != index) {
            return false;
        }
        ++check_index;
    }
    return true;
}

/// \brief      Generate an input subscript that provides to group dimensions into the common,
/// separate and reduced dimensions after transpose
///
/// \param      input_subscripts         A vector of the input subscripts
/// \param      common_labels_inds       A vector of indices of the common dimensions
/// \param      separate_labels_inds     A vector of indices of the separate dimensions
/// \param      reduced_labels_inds      A vector of indices of the reduced dimensions
/// \param      is_separate_first        A boolean flag. It is true if the separate dimensions
/// goes before the reduced dimensions
///
/// \return     An input subscript for grouping dimensions
///
std::string generate_grouping_subscript(const std::string& input_subscript,
                                        const std::vector<int64_t>& common_labels_inds,
                                        const std::vector<int64_t>& separate_labels_inds,
                                        const std::vector<int64_t>& reduced_labels_inds,
                                        bool& is_separate_first) {
    // transpose is not needed if common labels, reduced labels
    // and separate labels indices go concurrently
    std::vector<int64_t> labels_inds = common_labels_inds;
    labels_inds.insert(labels_inds.end(), reduced_labels_inds.begin(), reduced_labels_inds.end());
    labels_inds.insert(labels_inds.end(), separate_labels_inds.begin(), separate_labels_inds.end());
    if (is_range_0_to_n(labels_inds)) {
        is_separate_first = false;
        return input_subscript;
    }

    // transpose is not needed if common labels, separate labels
    // and reduced labels indices go concurrently
    labels_inds = common_labels_inds;
    labels_inds.insert(labels_inds.end(), separate_labels_inds.begin(), separate_labels_inds.end());
    labels_inds.insert(labels_inds.end(), reduced_labels_inds.begin(), reduced_labels_inds.end());
    if (is_range_0_to_n(labels_inds)) {
        is_separate_first = true;
        return input_subscript;
    }

    auto labels = ov::op::v7::Einsum::extract_labels(input_subscript);
    std::string required_subscript = "";
    for (auto index : labels_inds) {
        required_subscript += labels[index];
    }
    is_separate_first = true;
    return required_subscript;
}

/// \brief      Update a vector of input nodes and subscripts by removing items for operands
/// with indices input_ind1 and input_ind2 and inserted new input node and the corresponsing
/// subscript in the tail
///
/// \param      input_nodes         A vector of the input nodes to update
/// \param      input_subscripts    A vector of the input subscripts to update
/// \param      input_ind1          An index of item to be removed
/// \param      input_ind2          An index of item to be removed
/// \param      new_node            New input node to be inserted in the tail
/// \param      new_subscript       New input subscript to be inserted in the tail
///
void update_operands(ov::OutputVector& input_nodes,
                     std::vector<std::string>& input_subscripts,
                     size_t input_ind1,
                     size_t input_ind2,
                     const ov::Output<ov::Node>& new_node,
                     const std::string& new_subscript) {
    OPENVINO_ASSERT(input_ind1 < input_ind2);
    OPENVINO_ASSERT(input_ind2 < input_nodes.size());
    OPENVINO_ASSERT(input_ind2 < input_subscripts.size());
    input_nodes.erase(input_nodes.begin() + input_ind2);
    input_nodes.erase(input_nodes.begin() + input_ind1);
    input_nodes.push_back(new_node);
    input_subscripts.erase(input_subscripts.begin() + input_ind2);
    input_subscripts.erase(input_subscripts.begin() + input_ind1);
    input_subscripts.push_back(new_subscript);
}
using LabelDimMap = std::unordered_map<std::string, std::vector<size_t>>;

/// \brief Computes a mapping from labels to dimensions based on the input rank and subscript.
///
/// This function processes the input subscript to extract labels and maps them to the corresponding
/// dimensions of the input tensor. The function also considers the presence of ellipsis ("...") in
/// the labels and adjusts the dimension map accordingly.
///
/// \param input_rank The rank of the input tensor. It can be static or dynamic.
/// \param input_subscript The subscript string representing the labels of the input tensor dimensions.
/// \return A map where the keys are labels (strings) and the values are vectors of dimension indices.
///
LabelDimMap compute_label_dim_map(const ov::Rank& input_rank, const std::string& input_subscript) {
    static const std::string ellipsis = "...";
    const auto labels = ov::op::v7::Einsum::extract_labels(input_subscript);
    const auto static_input_rank = input_rank.is_static();
    OPENVINO_ASSERT(static_input_rank || (std::find(labels.begin(), labels.end(), ellipsis) == labels.end()),
                    "Input rank cannot be dynamic in case of ellipsis in input subscript");
    const size_t input_rank_length = static_input_rank ? input_rank.get_length() : labels.size();
    OPENVINO_ASSERT(input_rank_length >= labels.size());
    const size_t num_broadcasted_dims = input_rank_length - labels.size() + 1;
    OPENVINO_ASSERT(num_broadcasted_dims > 0);

    LabelDimMap resulted_map;
    size_t current_dim = 0;
    for (const auto& label : labels) {
        if (label == ellipsis) {
            std::vector<size_t> label_dims(num_broadcasted_dims);
            std::iota(label_dims.begin(), label_dims.end(), current_dim);
            resulted_map[label] = label_dims;
            current_dim += num_broadcasted_dims;
        } else if (resulted_map.find(label) != resulted_map.end()) {
            resulted_map[label].push_back(current_dim);
            ++current_dim;
        } else {
            std::vector<size_t> label_dims{current_dim};
            resulted_map[label] = label_dims;
            ++current_dim;
        }
    }

    return resulted_map;
}

/// \brief Computes the ranges for common, separated, and reduced labels in the input subscript.
///
/// This function calculates the start and end indices for common, separated, and reduced labels
/// based on the input rank and subscript. It also considers the presence of ellipsis ("...") in
/// the labels and adjusts the ranks accordingly.
///
/// \param input_rank The rank of the input tensor.
/// \param input_subscript The subscript string representing the input tensor dimensions.
/// \param common_labels A vector of strings representing the common labels.
/// \param sep_labels A vector of strings representing the separated labels.
/// \param reduced_labels A vector of strings representing the reduced labels.
/// \param common_begin Reference to a size_t variable to store the beginning index of common labels.
/// \param common_end Reference to a size_t variable to store the ending index of common labels.
/// \param sep_begin Reference to a size_t variable to store the beginning index of separated labels.
/// \param sep_end Reference to a size_t variable to store the ending index of separated labels.
/// \param reduced_begin Reference to a size_t variable to store the beginning index of reduced labels.
/// \param reduced_end Reference to a size_t variable to store the ending index of reduced labels.
/// \param is_separated_first Boolean flag indicating whether the separated labels should come before the reduced
/// labels.
///
void compute_ranges(const ov::Rank& input_rank,
                    const std::string& input_subscript,
                    const std::vector<std::string>& common_labels,
                    const std::vector<std::string>& sep_labels,
                    const std::vector<std::string>& reduced_labels,
                    size_t& common_begin,
                    size_t& common_end,
                    size_t& sep_begin,
                    size_t& sep_end,
                    size_t& reduced_begin,
                    size_t& reduced_end,
                    bool is_separated_first) {
    auto label_to_dim_map = compute_label_dim_map(input_rank, input_subscript);

    size_t common_rank = common_labels.size();
    size_t sep_rank = sep_labels.size();
    size_t reduced_rank = reduced_labels.size();

    static const std::string ellipsis = "...";
    // Adjust rank to include ellipsis dimensions.
    // Initial rank is the number of labels in the input subscript, so if the ellipsis is present, initial ellipsis rank
    // would be counted as 1. Adjust the rank to include actual ellipsis rank with accounting for existing "placeholder"
    // by subtracting by 1.
    if (label_to_dim_map.find(ellipsis) != label_to_dim_map.end()) {
        if (std::find(common_labels.begin(), common_labels.end(), ellipsis) != common_labels.end()) {
            common_rank += label_to_dim_map[ellipsis].size() - 1;
        }
        if (std::find(sep_labels.begin(), sep_labels.end(), ellipsis) != sep_labels.end()) {
            sep_rank += label_to_dim_map[ellipsis].size() - 1;
        }
        if (std::find(reduced_labels.begin(), reduced_labels.end(), ellipsis) != reduced_labels.end()) {
            reduced_rank += label_to_dim_map[ellipsis].size() - 1;
        }
    }

    common_begin = 0;
    common_end = common_begin + common_rank;
    if (is_separated_first) {
        sep_begin = common_end;
        sep_end = sep_begin + sep_rank;
        reduced_begin = sep_end;
        reduced_end = reduced_begin + reduced_rank;
    } else {
        reduced_begin = common_end;
        reduced_end = reduced_begin + reduced_rank;
        sep_begin = reduced_end;
        sep_end = sep_begin + sep_rank;
    }
}
/// \brief      Return input node with computed sub-shape defined by a range [s_begin;s_end)
///
/// \param      data_shape          Input node that contains some tensor shape
/// \param      s_begin             Start index of dimension
/// \param      s_end               End index of dimension
/// \param      subgraph_nodes      A vector of operation nodes where to add new ones
/// \param      is_product          A boolean flag that indicates if to compute a product of
/// dimension sizes in the computed sub-shape
///
/// \return     A vector of input nodes that can be empty (if s_end <= s_begin)
/// or contains just one input node with sub-shape or its product
///
ov::OutputVector compute_sub_shape(const ov::Output<ov::Node>& data_shape,
                                   size_t s_begin,
                                   size_t s_end,
                                   ov::NodeVector& subgraph_nodes,
                                   bool is_product = false) {
    int64_t begin = static_cast<int64_t>(s_begin);
    int64_t end = static_cast<int64_t>(s_end);
    ov::OutputVector sub_shape_vector;
    if (end <= begin) {
        return sub_shape_vector;
    }
    std::vector<int64_t> begin_mask(1, 0);
    std::vector<int64_t> end_mask(1, 0);
    auto begin_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape{1}, {begin});
    auto end_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape{1}, {end});
    auto stride_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape{1}, {1});
    auto sub_shape =
        std::make_shared<ov::op::v1::StridedSlice>(data_shape, begin_const, end_const, begin_mask, end_mask);

    if (is_product) {
        auto reduce_axis_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape{1}, {0});
        auto separate_shape_prod =
            std::make_shared<ov::op::v1::ReduceProd>(sub_shape->output(0), reduce_axis_const, true);
        sub_shape_vector.push_back(separate_shape_prod->output(0));
        subgraph_nodes.insert(subgraph_nodes.end(), {reduce_axis_const, separate_shape_prod});
    } else {
        sub_shape_vector.push_back(sub_shape->output(0));
    }
    subgraph_nodes.insert(subgraph_nodes.end(), {begin_const, end_const, stride_const, sub_shape});
    return sub_shape_vector;
}

/// \brief      Unsqueeze input node by given dimensions if a vector of unsqueezing dimensions
/// is not empty
///
/// \param      input_node          Input node to unsqueeze
/// \param      unsqueeze_axes      A vector of dimensions to be unsqueezed
/// \param      subgraph_nodes      A vector of operation nodes that is included into a
/// sub-graph decomposing Einsum that is needed for copy_runtime_info
///
/// \return     Unsqueezed input node if a vector of unsqueezing dimensions is not empty,
/// otherwise, the original input node
///
ov::Output<ov::Node> unsqueeze_input(const ov::Output<ov::Node>& input_node,
                                     const std::vector<int64_t>& unsqueeze_axes,
                                     ov::NodeVector& subgraph_nodes) {
    if (unsqueeze_axes.empty()) {
        return input_node;
    }
    auto unsqueeze_axes_const =
        ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape{unsqueeze_axes.size()}, unsqueeze_axes);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_node, unsqueeze_axes_const);
    subgraph_nodes.insert(subgraph_nodes.end(), {unsqueeze_axes_const, unsqueeze});
    return unsqueeze->output(0);
}

/// \brief Broadcasts and merges two shapes of the same rank.
///
/// This function takes two  shapes (shapes_lhs and shapes_rhs) of same rank and attempts to broadcast
/// and merge them into a single shape. The resulting broadcasted shape is returned as an OutputVector.
/// If one of the input vectors is empty, the other vector is returned as is.
///
/// \param shapes_lhs A single element vector containing the left-hand side shape to be broadcasted or empty.
/// \param shapes_rhs A single element vector containing the right-hand side shape to be broadcasted or empty.
/// \param subgraph_nodes A vector to which the nodes created during the broadcasting process are added.
/// \return An OutputVector containing the broadcasted and merged shape. If one of the input vectors is empty,
///         the other vector is returned.
///
ov::OutputVector broadcast_merge_shapes(const ov::OutputVector& shapes_lhs,
                                        const ov::OutputVector& shapes_rhs,
                                        ov::NodeVector& subgraph_nodes) {
    ov::OutputVector broadcasted_shape_nodes{};
    // OutputVector is either empty or contains a single shape
    if (shapes_lhs.size() == 1 && shapes_rhs.size() == 1) {
        // For common and reference subshapes, same rank should already be ensured by function
        // `unsqueeze_ellipses_to_same_rank`.
        const auto& maximum = std::make_shared<ov::op::v1::Maximum>(shapes_lhs[0], shapes_rhs[0]);
        subgraph_nodes.push_back(maximum);
        broadcasted_shape_nodes.push_back(maximum);
    } else if (shapes_lhs.size() == 0 && shapes_rhs.size() == 1) {
        return shapes_rhs;
    } else if (shapes_lhs.size() == 1 && shapes_rhs.size() == 0) {
        return shapes_lhs;
    }
    return broadcasted_shape_nodes;
}

/// \brief      Broadcast input node to the new shape specified by broadcasted sub-shapes of the common,
/// separate and reduced dimensions so that the broadcasted input has a format acceptable by Reshape MatMul
///
/// \param      input_node              Input node to reshape
/// \param      common_sub_shape        A sub-shape corresponding to the broadcasted common dimensions
/// \param      separate_sub_shape      A sub-shape corresponding to the broadcasted separate dimensions
/// \param      reduced_sub_shape_prod  A product of the broadcasted separate dimensions sizes
/// \param      is_separate_first       true - the separate dimensions placed before reduced
/// dimensions, otherwise, it is after them
/// \param      subgraph_nodes          A vector of operation nodes that is included into
/// a sub-graph decomposing Einsum that is needed for copy_runtime_info
///
/// \return     Broadcasted input node
///
ov::Output<ov::Node> broadcast_input(const ov::Output<ov::Node>& input_node,
                                     const ov::OutputVector& common_sub_shape,
                                     const ov::OutputVector& separate_sub_shape,
                                     const ov::OutputVector& reduced_sub_shape,
                                     bool is_separate_first,
                                     ov::NodeVector& subgraph_nodes) {
    ov::OutputVector new_shape_parts;
    new_shape_parts.insert(new_shape_parts.end(), common_sub_shape.begin(), common_sub_shape.end());
    // form a new shape for input so that collapsed dimensions corresponding
    // to the common, separate and reduced dimensions are placed in the correct order
    if (is_separate_first) {
        new_shape_parts.insert(new_shape_parts.end(), separate_sub_shape.begin(), separate_sub_shape.end());
        new_shape_parts.insert(new_shape_parts.end(), reduced_sub_shape.begin(), reduced_sub_shape.end());
    } else {
        new_shape_parts.insert(new_shape_parts.end(), reduced_sub_shape.begin(), reduced_sub_shape.end());
        new_shape_parts.insert(new_shape_parts.end(), separate_sub_shape.begin(), separate_sub_shape.end());
    }

    // in case of scalar reshape is not needed
    if (new_shape_parts.size() == 0) {
        return input_node;
    }
    auto new_shape_op = std::make_shared<ov::op::v0::Concat>(new_shape_parts, 0);
    // if new shape is possible to compute on the shape infer stage, insert Constant node immediately
    // in order to prevent repeated computing during constant-folding pass
    std::shared_ptr<ov::op::v3::Broadcast> reshaped_input_op;
    if (auto new_shape_const = ov::util::get_constant_from_source(new_shape_op)) {
        reshaped_input_op =
            std::make_shared<ov::op::v3::Broadcast>(input_node, new_shape_const, ov::op::BroadcastType::BIDIRECTIONAL);
        subgraph_nodes.insert(subgraph_nodes.end(), {new_shape_const});
    } else {
        reshaped_input_op = std::make_shared<ov::op::v3::Broadcast>(input_node,
                                                                    new_shape_op->output(0),
                                                                    ov::op::BroadcastType::BIDIRECTIONAL);
        subgraph_nodes.insert(subgraph_nodes.end(), {new_shape_op});
    }

    subgraph_nodes.insert(subgraph_nodes.end(), {reshaped_input_op});
    return reshaped_input_op->output(0);
}

/// \brief      Reshape input node to the new shape specified by sub-shapes of the common,
/// separate and reduced dimensions so that the reshaped input has a format acceptable by MatMul
///
/// \param      input_node              Input node to reshape
/// \param      common_sub_shape        A sub-shape corresponding to the common dimensions
/// \param      separate_sub_shape      A sub-shape corresponding to the separate dimensions
/// \param      reduced_sub_shape_prod  A product of the separate dimensions sizes
/// \param      is_separate_first       true - the separate dimensions placed before reduced
/// dimensions, otherwise, it is after them
/// \param      subgraph_nodes          A vector of operation nodes that is included into
/// a sub-graph decomposing Einsum that is needed for copy_runtime_info
///
/// \return     Reshaped input node
///
ov::Output<ov::Node> reshape_input_for_matmul(const ov::Output<ov::Node>& input_node,
                                              const ov::OutputVector& common_sub_shape,
                                              const ov::OutputVector& separate_sub_shape,
                                              const ov::OutputVector& reduced_sub_shape,
                                              bool is_separate_first,
                                              ov::NodeVector& subgraph_nodes) {
    ov::OutputVector new_shape_parts;
    new_shape_parts.insert(new_shape_parts.end(), common_sub_shape.begin(), common_sub_shape.end());

    // compute a product of a sub-shape for separate labels
    ov::OutputVector separate_parts;
    auto reduce_axis_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape{1}, {0});
    if (common_sub_shape.size() > 0 && separate_sub_shape.size() == 0) {
        // in this case new dimension corresponding to separate labels must be added
        // since MatMul operation is not possible to do without separate dimensions if the
        // common dimension presents
        auto separate_new_dim = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape{1}, {1});
        separate_parts.push_back(separate_new_dim);
        subgraph_nodes.insert(subgraph_nodes.end(), {separate_new_dim});
    } else if (separate_sub_shape.size() > 0) {
        // in this case compute a product of separate dimension sizes since they must be
        // presented with just one dimension for MatMul
        auto separate_shape_prod =
            std::make_shared<ov::op::v1::ReduceProd>(separate_sub_shape[0], reduce_axis_const, true);
        separate_parts.push_back(separate_shape_prod->output(0));
        subgraph_nodes.insert(subgraph_nodes.end(), {reduce_axis_const, separate_shape_prod});
    }
    ov::OutputVector reduced_sub_shape_prod;
    if (reduced_sub_shape.size() > 0) {
        auto const_0 = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
        auto product = std::make_shared<ov::op::v1::ReduceProd>(reduced_sub_shape[0], const_0, true);
        subgraph_nodes.insert(subgraph_nodes.end(), {reduce_axis_const, const_0, product});
        reduced_sub_shape_prod.push_back(product->output(0));
    }

    // form a new shape for input so that collapsed dimensions corresponding
    // to the common, separate and reduced dimensions are placed in the correct order

    if (is_separate_first) {
        new_shape_parts.insert(new_shape_parts.end(), separate_parts.begin(), separate_parts.end());
        new_shape_parts.insert(new_shape_parts.end(), reduced_sub_shape_prod.begin(), reduced_sub_shape_prod.end());
    } else {
        new_shape_parts.insert(new_shape_parts.end(), reduced_sub_shape_prod.begin(), reduced_sub_shape_prod.end());
        new_shape_parts.insert(new_shape_parts.end(), separate_parts.begin(), separate_parts.end());
    }

    // in case of scalar reshape is not needed
    if (new_shape_parts.size() == 0) {
        return input_node;
    }

    auto new_shape_op = std::make_shared<ov::op::v0::Concat>(new_shape_parts, 0);

    // if new shape is possible to compute on the shape infer stage, insert Constant node immediately
    // in order to prevent repeated computing during constant-folding pass
    std::shared_ptr<ov::op::v1::Reshape> reshaped_input_op;
    if (auto new_shape_const = ov::util::get_constant_from_source(new_shape_op)) {
        reshaped_input_op = std::make_shared<ov::op::v1::Reshape>(input_node, new_shape_const, false);
        subgraph_nodes.insert(subgraph_nodes.end(), {new_shape_const});
    } else {
        reshaped_input_op = std::make_shared<ov::op::v1::Reshape>(input_node, new_shape_op->output(0), false);
        subgraph_nodes.insert(subgraph_nodes.end(), {new_shape_op});
    }

    subgraph_nodes.insert(subgraph_nodes.end(), {reshaped_input_op});
    return reshaped_input_op->output(0);
}

/// \brief      Transpose one of the Einsum inputs to layout specified through the required
/// subscript
///
/// \param      input_nodes         A vector of input nodes to Einsum
/// \param      input_subscripts    A vector of corresponding subscripts for input nodes
/// \param      required_subscript  The required subscript that defines layout to which the
/// input is to transpose
/// \param      input_ind           An index of the input node to be transposed
/// \param      subgraph_nodes      A vector of operation nodes that is included into
/// a sub-graph decomposing Einsum that is needed for copy_runtime_info
///
void transpose_input(ov::OutputVector& input_nodes,
                     std::vector<std::string>& input_subscripts,
                     const std::string& required_subscript,
                     size_t input_ind,
                     ov::NodeVector& subgraph_nodes) {
    // perform sanity check for arguments
    const auto num_inputs = input_nodes.size();
    OPENVINO_ASSERT(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
    OPENVINO_ASSERT(input_ind < num_inputs, "Input index is out of range.");

    // generate permutation vector by searching for bijection between input_subscripts
    // and required_subscript
    std::vector<int64_t> permutation;
    const auto& input_subscript = input_subscripts[input_ind];

    // transpose is not needed since the input subscript is not going to be changed
    if (required_subscript == input_subscript) {
        return;
    }

    // find permutation that establishes bijection between the input subscript
    // and the required one
    const auto& input_node = input_nodes[input_ind];
    const auto labels = ov::op::v7::Einsum::extract_labels(input_subscript);
    const auto required_labels = ov::op::v7::Einsum::extract_labels(required_subscript);
    const auto label_dim_map = compute_label_dim_map(input_node.get_partial_shape().rank(), input_subscript);
    for (const auto& required_label : required_labels) {
        const auto label_dims_it = label_dim_map.find(required_label);
        OPENVINO_ASSERT(label_dims_it != label_dim_map.end());
        const auto& label_dims = label_dims_it->second;
        permutation.insert(permutation.end(), label_dims.begin(), label_dims.end());
    }

    // create a sub-graph for transposing into the required layout
    const auto permutation_const =
        ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape{permutation.size()}, permutation);
    const auto transpose = std::make_shared<ov::op::v1::Transpose>(input_node, permutation_const);

    // update a vector of inputs and input subscripts
    input_nodes[input_ind] = transpose->output(0);
    input_subscripts[input_ind] = required_subscript;

    // update a vector of nodes for copy_runtime_info
    subgraph_nodes.insert(subgraph_nodes.end(), {permutation_const, transpose});
}

/// \brief      Find labels (in a given input subscript) that are met once in the equation
/// and reduce dimensions corresponding to such labels
///
/// \param      input_nodes             A vector of input nodes to Einsum operation
/// \param      input_subscripts        A vector of corresponding subscripts for the input nodes
/// \param      output_subscript        The output subscript
/// \param      input_ind               An index of the input node for which it will check
/// dimensions to be reduced
/// \param      subgraph_nodes          A vector of operation nodes that is included into
/// a sub-graph decomposing Einsum that is needed for copy_runtime_info
///
void reduce_input(ov::OutputVector& input_nodes,
                  std::vector<std::string>& input_subscripts,
                  const std::string& output_subscript,
                  size_t input_ind,
                  ov::NodeVector& subgraph_nodes) {
    // perform sanity check for arguments
    const auto num_inputs = input_nodes.size();
    OPENVINO_ASSERT(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
    OPENVINO_ASSERT(input_ind < num_inputs, "Input index is out of range.");

    const auto& input_node = input_nodes[input_ind];
    const auto& input_subscript = input_subscripts[input_ind];

    // compute output shape and axes to reduce
    std::set<int64_t> reduced_axes;
    const auto labels = ov::op::v7::Einsum::extract_labels(input_subscripts[input_ind]);
    auto label_dim_map = compute_label_dim_map(input_node.get_partial_shape().rank(), input_subscript);
    std::string new_input_subscript = "";

    for (const auto& label : labels) {
        // check if the current label is met in the other input subscripts
        // or the output subscript
        const bool is_dim_reduced = is_dimension_reduced(input_subscripts, output_subscript, label, {input_ind});

        OPENVINO_ASSERT(label_dim_map.find(label) != label_dim_map.end());
        const auto& label_dims = label_dim_map[label];

        // if label is not met, dimension corresponding to the label is to reduce
        if (is_dim_reduced) {
            reduced_axes.insert(label_dims.begin(), label_dims.end());
        } else {
            new_input_subscript += label;
        }
    }

    if (reduced_axes.empty()) {
        // there is no axis to reduce
        return;
    }

    // reduce by summed up elements along dimension for which label is met just once
    const std::vector<int64_t> reduced_axes_vec{reduced_axes.cbegin(), reduced_axes.cend()};
    const auto axes_const =
        ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape{reduced_axes.size()}, reduced_axes_vec);
    const auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(input_node, axes_const, false);
    // update a vector of inputs and input subscripts
    input_nodes[input_ind] = reduce_sum->output(0);
    input_subscripts[input_ind] = new_input_subscript;

    // update a vector of nodes for copy_runtime_info
    subgraph_nodes.insert(subgraph_nodes.end(), {axes_const, reduce_sum});
}

/// \brief Prepares data for diagonal extraction in Einsum operation.
///
/// This function processes the input subscript and label-dimension map to identify repeated labels,
/// update the resultant subscript, and determine the axes to be reduced.
///
/// \param input_subscript The input subscript string representing the Einsum equation.
/// \param label_dim_map A map from labels to their corresponding dimensions.
/// \param resultant_subscript A reference to the resultant subscript string to be updated.
/// \param resultant_subscript_with_duplicates A reference to a resultant subscript string with duplicates.
/// \param repeated_labels A reference to a vector of strings to store repeated labels found in input subscript.
/// \param unrepeated_labels A reference to a vector where unrepeated labels will be stored.
/// \param reduced_axes A reference to an AxisVector to store the axes that need to be reduced.
///
void prepare_diagonal_extraction_data(const std::string& input_subscript,
                                      const LabelDimMap& label_dim_map,
                                      std::string& resultant_subscript,
                                      std::string& resultant_subscript_with_duplicates,
                                      std::vector<std::string>& repeated_labels,
                                      std::vector<std::string>& unrepeated_labels,
                                      ov::AxisVector& reduced_axes) {
    static const std::string ellipsis = "...";
    const auto labels = ov::op::v7::Einsum::extract_labels(input_subscript);
    std::vector<std::string> repeated_labels_with_duplicates;
    size_t reduced_dim = 1;
    for (const auto& label : labels) {
        if (resultant_subscript.find(label) != std::string::npos) {
            continue;
        }

        const auto dims_it = label_dim_map.find(label);
        OPENVINO_ASSERT(dims_it != label_dim_map.end());

        auto dims = dims_it->second;
        const auto dims_size = dims.size();
        OPENVINO_ASSERT(dims_size > 0);

        if (label != ellipsis && dims_size > 1) {
            // repeated label is found
            // save only the first dimension corresponding to the repeated label
            dims = {dims[0]};
            reduced_axes.push_back(reduced_dim);
            repeated_labels.push_back(label);
            repeated_labels_with_duplicates.insert(repeated_labels_with_duplicates.end(), dims_size, label);
            reduced_dim += 2;
            resultant_subscript += label;
        } else {
            unrepeated_labels.push_back(label);
        }
    }
    resultant_subscript = std::accumulate(unrepeated_labels.begin(), unrepeated_labels.end(), resultant_subscript);
    resultant_subscript_with_duplicates = std::accumulate(repeated_labels_with_duplicates.begin(),
                                                          repeated_labels_with_duplicates.end(),
                                                          std::string(""));
    resultant_subscript_with_duplicates =
        std::accumulate(unrepeated_labels.begin(), unrepeated_labels.end(), resultant_subscript_with_duplicates);
}

///
/// \brief Extracts the diagonal elements from the input tensor based on the provided subscripts.
///
/// This function modifies the input tensor by extracting its diagonal elements for repeated labels and updating the
/// corresponding subscript.
///
/// \param inputs A vector of input tensors.
/// \param input_subscripts A vector of subscripts corresponding to each input tensor.
/// \param input_ind The index of the input tensor to be processed.
/// \param subgraph_nodes      A vector of operation nodes that is included into
///                            a sub-graph decomposing Einsum that is needed for copy_runtime_info
///
void extract_diagonal(ov::OutputVector& inputs,
                      std::vector<std::string>& input_subscripts,
                      size_t input_ind,
                      ov::NodeVector& subgraph_nodes) {
    // perform sanity check for arguments
    const auto& num_inputs = inputs.size();
    OPENVINO_ASSERT(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
    OPENVINO_ASSERT(input_ind < num_inputs, "Input index is out of range.");

    const auto& input_node = inputs[input_ind];
    const auto& input_subscript = input_subscripts[input_ind];

    // Compute the label to dimension map for the input subscript
    const auto label_dim_map = compute_label_dim_map(input_node.get_partial_shape().rank(), input_subscript);
    std::string resultant_subscript;
    std::string resultant_subscript_with_duplicates;
    std::vector<std::string> repeated_labels;
    std::vector<std::string> unrepeated_labels;
    ov::AxisVector reduced_axes;

    // Prepare data for diagonal extraction
    prepare_diagonal_extraction_data(input_subscript,
                                     label_dim_map,
                                     resultant_subscript,
                                     resultant_subscript_with_duplicates,
                                     repeated_labels,
                                     unrepeated_labels,
                                     reduced_axes);

    // If there are no repeated labels, return early
    if (repeated_labels.size() == 0) {
        return;
    }

    // Transpose input so that repeated labels are grouped by same label and un-repeated labels are moved to the end
    transpose_input(inputs, input_subscripts, resultant_subscript, input_ind, subgraph_nodes);

    // Create a ShapeOf operation to get the shape of the input tensor
    const auto& input_shape = std::make_shared<ov::op::v3::ShapeOf>(input_node);
    ov::NodeVector begins;
    ov::NodeVector ends;

    // Compute the label to dimension map for the transposed input subscript
    const auto transposed_label_dim_map =
        compute_label_dim_map(input_node.get_partial_shape().rank(), resultant_subscript_with_duplicates);
    const auto& const_0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    const auto& const_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    subgraph_nodes.insert(subgraph_nodes.end(), {input_shape, const_0, const_1});
    ov::NodeVector convenient_shape_vector;
    ov::NodeVector shape_after_pad_vector;

    // Process each repeated label
    for (const std::string& repeated_label : repeated_labels) {
        const auto dim_map_repeated_label = transposed_label_dim_map.find(repeated_label);
        OPENVINO_ASSERT(dim_map_repeated_label != transposed_label_dim_map.end());
        const auto& repeated_label_dims = dim_map_repeated_label->second;
        const auto& repeated_label_indices =
            ov::op::v0::Constant::create(ov::element::i64, {repeated_label_dims.size()}, repeated_label_dims);
        const auto& repeated_label_indices_len =
            ov::op::v0::Constant::create(ov::element::i64, {}, {repeated_label_dims.size()});
        const auto& repeated_dimensions =
            std::make_shared<ov::op::v7::Gather>(input_shape, repeated_label_indices, const_0);
        const auto& repeated_dimension = std::make_shared<ov::op::v7::Gather>(repeated_dimensions, const_0, const_0);
        const auto& range_max_val = std::make_shared<ov::op::v1::Power>(repeated_dimension, repeated_label_indices_len);
        const auto& step_numerator = std::make_shared<ov::op::v1::Subtract>(range_max_val, const_1);
        const auto& step_denominator = std::make_shared<ov::op::v1::Subtract>(repeated_dimension, const_1);
        const auto& step_denominator_but_not_0 = std::make_shared<ov::op::v1::Maximum>(step_denominator, const_1);
        const auto& step_numerator_but_not_0 = std::make_shared<ov::op::v1::Maximum>(step_numerator, const_1);
        const auto& step = std::make_shared<ov::op::v1::Divide>(step_numerator_but_not_0, step_denominator_but_not_0);
        const auto& end = std::make_shared<ov::op::v1::Subtract>(step, const_1);
        const auto& reduced_size = std::make_shared<ov::op::v1::ReduceProd>(repeated_dimensions, const_0, true);
        // Flatten dimensions of repeated label
        convenient_shape_vector.push_back(reduced_size);
        // Compute the new shape after padding, separate diagonal elements
        shape_after_pad_vector.push_back(repeated_dimension);
        shape_after_pad_vector.push_back(step);
        begins.push_back(const_0);
        ends.push_back(end);
        subgraph_nodes.insert(subgraph_nodes.end(),
                              {repeated_label_indices,
                               repeated_label_indices_len,
                               repeated_dimensions,
                               repeated_dimension,
                               range_max_val,
                               step_numerator,
                               step_denominator,
                               step_denominator_but_not_0,
                               step_numerator_but_not_0,
                               step,
                               end,
                               reduced_size});
    }

    // Process unrepeated labels - do not modify or pad dimensions
    std::vector<size_t> unrepeated_dimension_indices_vec;
    for (std::string unrepeated_label : unrepeated_labels) {
        const auto& dim_map_unrepeated_label = transposed_label_dim_map.find(unrepeated_label);
        OPENVINO_ASSERT(dim_map_unrepeated_label != transposed_label_dim_map.end());
        const auto& unrepeated_label_dims = dim_map_unrepeated_label->second;
        unrepeated_dimension_indices_vec.insert(unrepeated_dimension_indices_vec.end(),
                                                unrepeated_label_dims.begin(),
                                                unrepeated_label_dims.end());
        begins.insert(begins.end(), unrepeated_label_dims.size(), const_0);
        ends.insert(ends.end(), unrepeated_label_dims.size(), const_0);
    }

    // Gather the dimensions for unrepeated labels in single call
    const auto& unrepeated_dimensions_indices = ov::op::v0::Constant::create(ov::element::i64,
                                                                             {unrepeated_dimension_indices_vec.size()},
                                                                             unrepeated_dimension_indices_vec);
    const auto& unrepeated_dimensions =
        std::make_shared<ov::op::v7::Gather>(input_shape, unrepeated_dimensions_indices, const_0);
    subgraph_nodes.insert(subgraph_nodes.end(), {unrepeated_dimensions_indices, unrepeated_dimensions});
    convenient_shape_vector.push_back(unrepeated_dimensions);
    shape_after_pad_vector.push_back(unrepeated_dimensions);

    // Create the new shape for the input tensor that would flatten repeated label dimensions
    const auto& convenient_shape = std::make_shared<ov::op::v0::Concat>(convenient_shape_vector, 0);
    const auto& reshaped_input = std::make_shared<ov::op::v1::Reshape>(input_node, convenient_shape, false);
    // Create the pads for the label-flattened input tensor to extract the diagonal elements
    const auto& pads_end = std::make_shared<ov::op::v0::Concat>(ends, 0);
    const auto& pads_begin = std::make_shared<ov::op::v0::Concat>(begins, 0);
    const auto& pad =
        std::make_shared<ov::op::v1::Pad>(reshaped_input, pads_begin, pads_end, ov::op::PadMode::CONSTANT);
    // Reshape the tensor after padding to extract the diagonal elements to separate dimensions
    const auto& reshape_after_pad_target = std::make_shared<ov::op::v0::Concat>(shape_after_pad_vector, 0);
    const auto& reshape_after_pad = std::make_shared<ov::op::v1::Reshape>(pad, reshape_after_pad_target, false);
    subgraph_nodes.insert(
        subgraph_nodes.end(),
        {convenient_shape, pads_begin, pads_end, reshaped_input, pad, reshape_after_pad_target, reshape_after_pad});

    // Gather the diagonal elements
    std::shared_ptr<ov::Node> gather = reshape_after_pad;
    for (auto axis : reduced_axes) {
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
        gather = std::make_shared<ov::op::v7::Gather>(gather, const_0, axis_const);
        subgraph_nodes.insert(subgraph_nodes.end(), {axis_const, gather});
    }

    // Squeeze the gathered tensor to remove the reduced axes
    const auto& reduced_indices = ov::op::v0::Constant::create(ov::element::i64, {reduced_axes.size()}, reduced_axes);
    const auto& out_node = std::make_shared<ov::op::v0::Squeeze>(gather, reduced_indices);
    subgraph_nodes.insert(subgraph_nodes.end(), {reduced_indices, out_node});

    // Update the input tensor and its subscript
    inputs[input_ind] = out_node->output(0);
    input_subscripts[input_ind] = resultant_subscript;
}

/// \brief Adjusts the ranks of two input tensors by unsqueezing ellipses to the same rank.
///
/// This function ensures that the ellipses in the input subscripts of the two tensors have the same rank by unsqueezing
/// the necessary dimensions. It modifies the inputs in place.
///
/// \param inputs A vector of input tensors.
/// \param input_subscripts A vector of input subscripts corresponding to the input tensors.
/// \param input_ind1 The index of the first input tensor in the inputs vector.
/// \param input_ind2 The index of the second input tensor in the inputs vector.
/// \param subgraph_nodes A vector of operation nodes that is included into
///                      a sub-graph decomposing Einsum that is needed for copy_runtime_info
void unsqueeze_ellipses_to_same_rank(ov::OutputVector& inputs,
                                     std::vector<std::string>& input_subscripts,
                                     size_t input_ind1,
                                     size_t input_ind2,
                                     ov::NodeVector& subgraph_nodes) {
    constexpr char ellipsis[] = "...";
    const auto& input1 = inputs[input_ind1];
    const auto& input2 = inputs[input_ind2];
    OPENVINO_ASSERT(input1.get_partial_shape().rank().is_static() && input2.get_partial_shape().rank().is_static());
    auto label_to_dim_map1 = compute_label_dim_map(input1.get_partial_shape().size(), input_subscripts[input_ind1]);
    auto label_to_dim_map2 = compute_label_dim_map(input2.get_partial_shape().size(), input_subscripts[input_ind2]);
    if (label_to_dim_map1.find(ellipsis) != label_to_dim_map1.end() &&
        label_to_dim_map2.find(ellipsis) != label_to_dim_map2.end()) {
        std::vector<int64_t> unsqueeze_axis1, unsqueeze_axis2;
        const auto& ellipsis_dims1 = label_to_dim_map1[ellipsis];
        const auto& ellipsis_dims2 = label_to_dim_map2[ellipsis];
        if (ellipsis_dims2.size() > ellipsis_dims1.size()) {
            for (size_t i = 0; i < ellipsis_dims2.size() - ellipsis_dims1.size(); ++i) {
                unsqueeze_axis1.push_back(ellipsis_dims1[0] + i);
            }
        } else if (ellipsis_dims1.size() > ellipsis_dims2.size()) {
            for (size_t i = 0; i < ellipsis_dims1.size() - ellipsis_dims2.size(); ++i) {
                unsqueeze_axis2.push_back(ellipsis_dims2[0] + i);
            }
        }
        ov::Output<ov::Node> unsqueeze_output1 = unsqueeze_input(input1, unsqueeze_axis1, subgraph_nodes);
        ov::Output<ov::Node> unsqueeze_output2 = unsqueeze_input(input2, unsqueeze_axis2, subgraph_nodes);
        inputs[input_ind1] = unsqueeze_output1;
        inputs[input_ind2] = unsqueeze_output2;
        return;
    }
}

/// \brief      Contract two inputs of Einsum operation according to equation.
/// The result of the contraction is appended into input_nodes along with its subscript.
/// The input nodes for these two operands are removed from input_nodes along with their input
/// subscripts
///
/// \param      input_nodes             A vector of input nodes to Einsum operation
/// \param      input_subscripts        A vector of corresponding subscripts for the input nodes
/// \param      output_subscript        The output subscript
/// \param      input_ind1              An index of the first operand
/// \param      input_ind2              An index of the second operand
/// \param      subgraph_nodes          A vector of operation nodes that is included into a
/// sub-graph decomposing Einsum that is needed for copy_runtime_info
///
void contract_two_inputs(ov::OutputVector& input_nodes,
                         std::vector<std::string>& input_subscripts,
                         const std::string& output_subscript,
                         size_t input_ind1,
                         size_t input_ind2,
                         ov::NodeVector& subgraph_nodes) {
    // assume that input_ind1 < input_ind2 without loss of generality, otherwise, just swap them
    if (input_ind2 < input_ind1) {
        std::swap(input_ind1, input_ind2);
    }

    // perform sanity check for arguments
    auto num_inputs = input_nodes.size();
    OPENVINO_ASSERT(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
    OPENVINO_ASSERT(input_ind2 < num_inputs && input_ind1 != input_ind2, "Incorrect input index is specified.");

    const auto& input_node1 = input_nodes[input_ind1];
    const auto& input_node2 = input_nodes[input_ind2];

    // unsqueeze inputs to have same rank of ellipsis for correct broadcasting
    unsqueeze_ellipses_to_same_rank(input_nodes, input_subscripts, input_ind1, input_ind2, subgraph_nodes);

    // extract diagonals in case repeated labels in the corresponding input subscripts
    extract_diagonal(input_nodes, input_subscripts, input_ind1, subgraph_nodes);
    extract_diagonal(input_nodes, input_subscripts, input_ind2, subgraph_nodes);

    // reduce dimensions for input operands if possible
    reduce_input(input_nodes, input_subscripts, output_subscript, input_ind1, subgraph_nodes);
    reduce_input(input_nodes, input_subscripts, output_subscript, input_ind2, subgraph_nodes);

    // step 0. split dimensions of both operands into three groups:
    // 1. dimension indices with the same labels (in both subscripts) that are NOT reduced -
    // common labels (dimensions)
    // 2. dimension indices with labels that are met only in one of two subscripts - separate
    // labels (dimensions)
    // 3. dimension indices with the same labels (in both subscripts) that are reduced - reduced
    // labels (dimensions) NOTE: dimension is reduced iff. the corresponding label are met in
    // neither the output subscript nor the input subscripts for other Einsum inputs excluding
    // two given inputs
    auto& input_subscript1 = input_subscripts[input_ind1];
    auto labels1 = ov::op::v7::Einsum::extract_labels(input_subscript1);
    auto& input_subscript2 = input_subscripts[input_ind2];
    auto labels2 = ov::op::v7::Einsum::extract_labels(input_subscript2);
    std::string common_part = "";
    std::string separate_part1 = "";
    std::string separate_part2 = "";
    std::vector<int64_t> common_labels_inds1, common_labels_inds2;
    std::vector<int64_t> separate_labels_inds1, separate_labels_inds2;
    std::vector<int64_t> reduced_labels_inds1, reduced_labels_inds2;
    std::vector<std::string> common_labels, sep_labels1, sep_labels2, reduced_labels;
    for (size_t label_ind = 0; label_ind < labels1.size(); ++label_ind) {
        const auto& label = labels1[label_ind];
        auto iter = std::find(labels2.begin(), labels2.end(), label);
        if (iter != labels2.end()) {
            bool is_dim_reduced =
                is_dimension_reduced(input_subscripts, output_subscript, label, {input_ind1, input_ind2});
            common_part += label;
            if (is_dim_reduced) {
                reduced_labels_inds1.push_back(static_cast<int64_t>(label_ind));
                reduced_labels_inds2.push_back(static_cast<int64_t>(iter - labels2.begin()));
                reduced_labels.push_back(label);
            } else {
                common_labels_inds1.push_back(static_cast<int64_t>(label_ind));
                common_labels_inds2.push_back(static_cast<int64_t>(iter - labels2.begin()));
                common_labels.push_back(label);
            }
        } else {
            separate_part1 += label;
            separate_labels_inds1.push_back(static_cast<int64_t>(label_ind));
            sep_labels1.push_back(label);
        }
    }
    for (size_t label_ind = 0; label_ind < labels2.size(); ++label_ind) {
        const auto& label = labels2[label_ind];
        auto iter = std::find(labels1.begin(), labels1.end(), label);
        if (iter == labels1.end()) {
            separate_part2 += label;
            separate_labels_inds2.push_back(static_cast<int64_t>(label_ind));
            sep_labels2.push_back(label);
        }
    }

    // if there is no common dimension to reduce, apply eltwise multiplication
    if (reduced_labels_inds1.empty()) {
        std::string convenient_subscript = common_part + separate_part2;
        std::string resultant_subscript = input_subscript1 + separate_part2;

        // transpose the second operand in order to get the convenient layout
        // for further unsqueezing
        transpose_input(input_nodes, input_subscripts, convenient_subscript, input_ind2, subgraph_nodes);

        // unsqueeze the first operand with new dimensions in the tail
        // and the number of them is equal to the number of separate labels in the second
        // subscript

        int64_t unsqueeze_dim = input_node1.get_partial_shape().size();
        size_t axis_1_unsqueeze_amount = separate_labels_inds2.size();
        if (separate_part2.find("...") != std::string::npos) {
            size_t input2_ellipsis_rank = input_node2.get_partial_shape().size() - labels2.size();
            axis_1_unsqueeze_amount += input2_ellipsis_rank;
        }
        std::vector<int64_t> unsqueeze_axis1;
        for (size_t label_ind = 0; label_ind < axis_1_unsqueeze_amount; ++label_ind) {
            unsqueeze_axis1.push_back(unsqueeze_dim++);
        }
        const auto& unsqueeze_axis2 = separate_labels_inds1;

        // unsqueeze input operands for elementwise-multiplication with broadcasting
        auto unsqueeze_output1 = unsqueeze_input(input_node1, unsqueeze_axis1, subgraph_nodes);
        auto unsqueeze_output2 = unsqueeze_input(input_node2, unsqueeze_axis2, subgraph_nodes);

        // multiply both operands with broadcasting
        auto mul = std::make_shared<ov::op::v1::Multiply>(unsqueeze_output1,
                                                          unsqueeze_output2,
                                                          ov::op::AutoBroadcastType::NUMPY);

        // update input operand and input subscript for Einsum operation
        update_operands(input_nodes, input_subscripts, input_ind1, input_ind2, mul->output(0), resultant_subscript);

        // update a vector of nodes for copy_runtime_info
        subgraph_nodes.insert(subgraph_nodes.end(), {mul});
        return;
    }

    // in this case a set of reduced labels is not empty and it can apply MatMul operation
    // step 1. transpose both operands so that common labels, separated and reduced labels
    // are grouped for both operands
    bool is_separate_first1 = false;
    auto int_subscript1 = generate_grouping_subscript(input_subscript1,
                                                      common_labels_inds1,
                                                      separate_labels_inds1,
                                                      reduced_labels_inds1,
                                                      is_separate_first1);
    transpose_input(input_nodes, input_subscripts, int_subscript1, input_ind1, subgraph_nodes);
    bool is_separate_first2 = false;
    auto int_subscript2 = generate_grouping_subscript(input_subscript2,
                                                      common_labels_inds2,
                                                      separate_labels_inds2,
                                                      reduced_labels_inds2,
                                                      is_separate_first2);
    transpose_input(input_nodes, input_subscripts, int_subscript2, input_ind2, subgraph_nodes);

    auto matmul_operand1 = input_node1;
    auto matmul_operand2 = input_node2;

    size_t common_dims_begin, common_dims_end, reduced_dims_begin, reduced_dims_end, separate1_dims_begin,
        separate1_dims_end;
    compute_ranges(input_node1.get_partial_shape().rank(),
                   input_subscript1,
                   common_labels,
                   sep_labels1,
                   reduced_labels,
                   common_dims_begin,
                   common_dims_end,
                   separate1_dims_begin,
                   separate1_dims_end,
                   reduced_dims_begin,
                   reduced_dims_end,
                   is_separate_first1);

    size_t common_dims_begin2, common_dims_end2, reduced_dims_begin2, reduced_dims_end2, separate2_dims_begin,
        separate2_dims_end;
    compute_ranges(input_node2.get_partial_shape().rank(),
                   input_subscript2,
                   common_labels,
                   sep_labels2,
                   reduced_labels,
                   common_dims_begin2,
                   common_dims_end2,
                   separate2_dims_begin,
                   separate2_dims_end,
                   reduced_dims_begin2,
                   reduced_dims_end2,
                   is_separate_first2);

    ov::OutputVector common_sub_shape, separate1_sub_shape, separate2_sub_shape;

    auto data_shape1 = std::make_shared<ov::op::v3::ShapeOf>(input_node1);
    auto data_shape2 = std::make_shared<ov::op::v3::ShapeOf>(input_node2);
    subgraph_nodes.insert(subgraph_nodes.end(), {data_shape1});
    subgraph_nodes.insert(subgraph_nodes.end(), {data_shape2});
    common_sub_shape = compute_sub_shape(data_shape1, common_dims_begin, common_dims_end, subgraph_nodes);
    auto common_sub_shape2 = compute_sub_shape(data_shape2, common_dims_begin2, common_dims_end2, subgraph_nodes);
    OPENVINO_ASSERT(common_sub_shape.size() == common_sub_shape2.size());
    common_sub_shape = broadcast_merge_shapes(common_sub_shape, common_sub_shape2, subgraph_nodes);
    auto reduced_sub_shape =
        compute_sub_shape(data_shape1, reduced_dims_begin, reduced_dims_end, subgraph_nodes, false);
    auto reduced_sub_shape2 =
        compute_sub_shape(data_shape2, reduced_dims_begin2, reduced_dims_end2, subgraph_nodes, false);

    reduced_sub_shape = broadcast_merge_shapes(reduced_sub_shape, reduced_sub_shape2, subgraph_nodes);

    separate1_sub_shape = compute_sub_shape(data_shape1, separate1_dims_begin, separate1_dims_end, subgraph_nodes);
    matmul_operand1 = broadcast_input(input_node1,
                                      common_sub_shape,
                                      separate1_sub_shape,
                                      reduced_sub_shape,
                                      is_separate_first1,
                                      subgraph_nodes);
    separate2_sub_shape = compute_sub_shape(data_shape2, separate2_dims_begin, separate2_dims_end, subgraph_nodes);
    matmul_operand2 = broadcast_input(input_node2,
                                      common_sub_shape,
                                      separate2_sub_shape,
                                      reduced_sub_shape,
                                      is_separate_first2,
                                      subgraph_nodes);

    // step 2. reshape both operands so that separate labels and reduced labels are represented
    // with just one dimension this is needed by MatMul operation requirement to operands
    // format. For example, the shape must be in a format [B1, ..., Bm, X1, Y] or [B1, ..., Bm,
    // Y, X2], where B1, ..., Bm are common dimensions, X1 and X2 are collapsed dimensions
    // for separate labels and Y is collapsed dimension for reduced labels
    // this step is not needed for the operand if it satisfies to one of the requirements:
    // 1. there is just one separate dimension and just one reduced dimension
    // 2. there is no separate dimension, no common dimensions, and just one reduced dimension
    const auto common_labels1_size = common_dims_end - common_dims_begin;
    const auto common_labels2_size = common_dims_end2 - common_dims_begin2;
    const auto reduced_labels1_size = reduced_dims_end - reduced_dims_begin;
    const auto reduced_labels2_size = reduced_dims_end2 - reduced_dims_begin2;
    const auto separate_labels1_size = separate1_dims_end - separate1_dims_begin;
    const auto separate_labels2_size = separate2_dims_end - separate2_dims_begin;
    bool no_reshape_for_matmul1 = (reduced_labels1_size == 1 && separate_labels1_size == 1) ||
                                  (reduced_labels1_size == 1 && common_labels1_size == 0 && separate_labels1_size == 0);
    bool no_reshape_for_matmul2 = (reduced_labels2_size == 1 && separate_labels2_size == 1) ||
                                  (reduced_labels2_size == 1 && common_labels2_size == 0 && separate_labels2_size == 0);
    // reshape back after MatMul is not needed if one of two requirements satisfies for both operands:
    // 1. there is just one separate dimension
    // 2. there is no separate dimension and no common dimensions present.
    // If there is no separate dimension and common dimensions present, reshape is needed
    // because auxiliary separate dimension has been added by Unsqueeze operation
    // in the purpose for MatMul
    bool no_reshape_back1 = (separate_labels1_size == 1) || (common_labels1_size == 0 && separate_labels1_size == 0);
    bool no_reshape_back2 = (separate_labels2_size == 1) || (common_labels2_size == 0 && separate_labels2_size == 0);
    bool no_reshape_after_matmul = no_reshape_back1 && no_reshape_back2;
    if (no_reshape_for_matmul1 == false || no_reshape_after_matmul == false) {
        matmul_operand1 = reshape_input_for_matmul(matmul_operand1,
                                                   common_sub_shape,
                                                   separate1_sub_shape,
                                                   reduced_sub_shape,
                                                   is_separate_first1,
                                                   subgraph_nodes);
    }

    if (no_reshape_for_matmul2 == false || no_reshape_after_matmul == false) {
        matmul_operand2 = reshape_input_for_matmul(matmul_operand2,
                                                   common_sub_shape,
                                                   separate2_sub_shape,
                                                   reduced_sub_shape,
                                                   is_separate_first2,
                                                   subgraph_nodes);
    }

    // step 3. apply MatMul operation for formatted inputs
    bool transpose_a = (is_separate_first1 ? false : true);
    bool transpose_b = (is_separate_first2 ? true : false);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(matmul_operand1, matmul_operand2, transpose_a, transpose_b);

    // step 4. reshape back by unrolling dimensions corresponding to separate labels if needed
    // now dimensions corresponding to reduced labels are reduced by the MatMul operation
    common_part = "";
    for (const auto& common_label : common_labels) {
        common_part += common_label;
    }
    const std::string resultant_subscript = common_part + separate_part1 + separate_part2;
    if (no_reshape_after_matmul) {
        // this is a case when Reshape is not needed after MatMul operation
        // since there are no collapsed (or auxiliary added) separated dimensions
        update_operands(input_nodes, input_subscripts, input_ind1, input_ind2, matmul->output(0), resultant_subscript);
    } else {
        ov::OutputVector new_shape;
        new_shape.insert(new_shape.end(), common_sub_shape.begin(), common_sub_shape.end());
        new_shape.insert(new_shape.end(), separate1_sub_shape.begin(), separate1_sub_shape.end());
        new_shape.insert(new_shape.end(), separate2_sub_shape.begin(), separate2_sub_shape.end());
        auto result_shape_op = std::make_shared<ov::op::v0::Concat>(new_shape, 0);

        // if new shape is possible to compute on the shape infer stage, insert Constant node immediately
        // in order to prevent repeated computing during constant-folding pass
        std::shared_ptr<ov::op::v1::Reshape> result_op;
        if (auto new_shape_const = ov::util::get_constant_from_source(result_shape_op)) {
            result_op = std::make_shared<ov::op::v1::Reshape>(matmul->output(0), new_shape_const, false);
            subgraph_nodes.insert(subgraph_nodes.end(), {result_shape_op, new_shape_const});
        } else {
            result_op = std::make_shared<ov::op::v1::Reshape>(matmul->output(0), result_shape_op->output(0), false);
            subgraph_nodes.insert(subgraph_nodes.end(), {result_shape_op});
        }

        // update input operand and input subscript for Einsum operation
        update_operands(input_nodes,
                        input_subscripts,
                        input_ind1,
                        input_ind2,
                        result_op->output(0),
                        resultant_subscript);
        subgraph_nodes.insert(subgraph_nodes.end(), {result_op});
    }

    // update a vector of nodes for copy_runtime_info
    subgraph_nodes.insert(subgraph_nodes.end(), {matmul});
}

/// \brief Adjusts input subscripts and nodes to handle 0-dimensional ellipsis in Einsum operations.
///
/// Handle ellipses labels that do not represent any dimensions:
/// 1. If there is no ellipsis in the input subscripts, remove ellipsis from the output subscript.
/// 2. If all ellipses in the input subscripts do not represent any dimensions, remove ellipses from all subscripts.
/// 3. If there is at least one ellipsis that represents dimension, unsqueeze ellipses that do not represent any,
///
/// \param input_nodes A vector of input nodes for the Einsum operation.
/// \param input_subscripts A vector of input subscripts corresponding to the input nodes.
/// \param output_subscript The output subscript for the Einsum operation.
/// \param subgraph_nodes A vector to store nodes created during the subgraph transformation.
void fix_inputs_with_0d_ellipsis(ov::OutputVector& input_nodes,
                                 std::vector<std::string>& input_subscripts,
                                 std::string& output_subscript,
                                 ov::NodeVector& subgraph_nodes) {
    static const std::string ellipsis = "...";
    bool has_ellipsis = false;
    bool all_no_ellipsis_or_empty = true;

    for (size_t i = 0; i < input_nodes.size(); ++i) {
        const auto& labels = ov::op::v7::Einsum::extract_labels(input_subscripts[i]);
        bool has_ellipsis_in_input = std::find(labels.begin(), labels.end(), ellipsis) != labels.end();
        has_ellipsis |= has_ellipsis_in_input;
        all_no_ellipsis_or_empty &=
            !has_ellipsis_in_input || (input_nodes[i].get_partial_shape().size() + 1 == labels.size());
    }

    if (!has_ellipsis) {
        if (output_subscript.find(ellipsis) != std::string::npos) {
            output_subscript.erase(output_subscript.find(ellipsis), ellipsis.size());
        }
    } else if (all_no_ellipsis_or_empty) {
        for (auto& subscript : input_subscripts) {
            if (subscript.find(ellipsis) != std::string::npos) {
                subscript.erase(subscript.find(ellipsis), ellipsis.size());
            }
        }
        if (output_subscript.find(ellipsis) != std::string::npos) {
            output_subscript.erase(output_subscript.find(ellipsis), ellipsis.size());
        }
    } else {
        for (size_t i = 0; i < input_nodes.size(); ++i) {
            const auto& labels = ov::op::v7::Einsum::extract_labels(input_subscripts[i]);
            if (std::find(labels.begin(), labels.end(), ellipsis) != labels.end() &&
                input_nodes[i].get_partial_shape().size() + 1 == labels.size()) {
                input_nodes[i] = unsqueeze_input(
                    input_nodes[i],
                    {static_cast<int64_t>(
                        std::distance(labels.begin(), std::find(labels.begin(), labels.end(), ellipsis)))},
                    subgraph_nodes);
            }
        }
    }
}
}  // namespace

/// \brief Constructor for the EinsumDecomposition transformation pass.
///
/// This transformation decomposes the Einsum operation into a sequence of more basic operations.
/// It matches the Einsum operation and replaces it with a sub-graph of operations that perform
/// the same computation.
///
/// The transformation follows these steps:
/// 1. Parse the Einsum equation to extract input and output subscripts.
/// 2. Check if the transformation is applicable by ensuring all input nodes have static ranks.
/// 3. Compute the optimal path for contracting pairs of operands.
/// 4. Fix inputs where ellipsis does not contain any dimensions.
/// 5. Contract inputs by Einsum until only one input remains.
/// 6. Extract the diagonal for the single remaining operand.
/// 7. Reduce dimensions for the remaining input node.
/// 8. Transpose dimensions to match the layout required by the output subscript.
/// 9. Replace the original Einsum node with the last node from the decomposed sub-graph,
///    preserving the original node's name and runtime information.
ov::pass::EinsumDecomposition::EinsumDecomposition() {
    MATCHER_SCOPE(EinsumDecomposition);
    auto einsum = ov::pass::pattern::wrap_type<ov::op::v7::Einsum>();
    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto einsum_node = ov::as_type_ptr<ov::op::v7::Einsum>(m.get_match_root());
        if (!einsum_node) {
            return false;
        }

        // Parse the Einsum equation to get input and output subscripts
        auto equation = einsum_node->get_equation();
        std::vector<std::string> input_subscripts;
        std::string output_subscript;
        ov::op::v7::Einsum::parse_equation(equation, input_subscripts, output_subscript);

        // create a list of input nodes with preserving their order
        // and a vector of sub-graph nodes for copy_runtime_info
        ov::OutputVector input_nodes = einsum_node->input_values();
        ov::NodeVector subgraph_nodes;

        // Check if the transformation is applicable by ensuring all input nodes have static ranks
        if (std::any_of(input_nodes.cbegin(), input_nodes.cend(), [](ov::Output<Node> node) {
                return node.get_partial_shape().rank().is_dynamic();
            })) {
            return false;
        }

        // compute einsum path that is used to contract a pair of operands
        // in more optimal order
        auto einsum_path = compute_einsum_path(einsum_node);

        // fix inputs where ellipsis does not contain any dimensions
        fix_inputs_with_0d_ellipsis(input_nodes, input_subscripts, output_subscript, subgraph_nodes);

        // contract inputs by Einsum until just one is remained
        for (auto const& inds_pair : einsum_path) {
            contract_two_inputs(input_nodes,
                                input_subscripts,
                                output_subscript,
                                inds_pair.first,
                                inds_pair.second,
                                subgraph_nodes);
        }

        // Ensure only one input node remains after contraction
        OPENVINO_ASSERT(input_nodes.size() == 1);

        // extract diagonal for the single operand
        extract_diagonal(input_nodes, input_subscripts, 0, subgraph_nodes);
        // reduce dimensions for the remained input node
        reduce_input(input_nodes, input_subscripts, output_subscript, 0, subgraph_nodes);
        // transpose dimensions to layout required by the output subscript
        transpose_input(input_nodes, input_subscripts, output_subscript, 0, subgraph_nodes);
        // replace the original Einsum node with the last node from decomposing sub-graph
        // preserve the original node name
        auto last_node = input_nodes[0].get_node_shared_ptr();
        last_node->set_friendly_name(einsum_node->get_friendly_name());
        ov::copy_runtime_info(einsum_node, subgraph_nodes);
        ov::replace_node(einsum_node, last_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(einsum, matcher_name);
    register_matcher(m, callback);
}

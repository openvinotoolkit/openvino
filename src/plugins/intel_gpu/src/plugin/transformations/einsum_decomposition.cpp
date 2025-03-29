// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "einsum_decomposition.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/einsum.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/matmul.hpp"

#include <unordered_map>

namespace ov::intel_gpu {

namespace {
using LabelDimMap = std::unordered_map<std::string, std::vector<size_t>>;

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
bool is_dimension_reduced(const std::vector<std::string>& input_subscripts, const std::string& output_subscript,
    const std::string label_to_check, const std::vector<size_t>& excluded_indices) {
    for (size_t input_ind = 0; input_ind < input_subscripts.size(); ++input_ind) {
        const auto& input_subscript = input_subscripts[input_ind];
        // the subscript is checked only if its index is not in excluded indices list
        bool check_subscript = (std::find(excluded_indices.begin(), excluded_indices.end(), input_ind) == excluded_indices.end());
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
bool is_range_0_to_n(const std::vector<int64_t> &labels_inds) {
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
std::string generate_grouping_subscript(const std::string& input_subscript, const std::vector<int64_t>& common_labels_inds,
                                        const std::vector<int64_t>& separate_labels_inds, const std::vector<int64_t>& reduced_labels_inds,
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
void update_operands(ov::OutputVector& input_nodes, std::vector<std::string>& input_subscripts, size_t input_ind1, size_t input_ind2,
                     const ov::Output<ov::Node>& new_node, const std::string& new_subscript) {
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

/// \brief      Return sub-shape defined by range [s_begin;s_end)
///
/// \param      input_shape         Input shape
/// \param      s_begin             Start index of dimension
/// \param      s_end               End index of dimension
/// \param      is_product          A boolean flag that indicates if to compute a product of
/// dimension sizes in the computed sub-shape
///
/// \return     sub-shape
///
ov::Shape compute_sub_shape(const ov::Shape& input_shape, size_t begin, size_t end, bool is_product = false) {
    OPENVINO_ASSERT(end <= input_shape.size());
    if (end <= begin) {
        return ov::Shape();
    }
    ov::Shape sub_shape(input_shape.begin() + begin, input_shape.begin() + end);

    if (is_product) {
        const auto prod = shape_size(sub_shape);
        sub_shape = {prod};
    }
    return sub_shape;
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
ov::Output<ov::Node> unsqueeze_input(const ov::Output<ov::Node>& input_node, const std::vector<int64_t>& unsqueeze_axes,
                                             ov::NodeVector& subgraph_nodes) {
    if (unsqueeze_axes.empty()) {
        return input_node;
    }
    auto unsqueeze_axes_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape {unsqueeze_axes.size()}, unsqueeze_axes);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_node, unsqueeze_axes_const);
    subgraph_nodes.insert(subgraph_nodes.end(), {unsqueeze_axes_const, unsqueeze});
    return unsqueeze->output(0);
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
                                                      const ov::Shape& common_sub_shape,
                                                      const ov::Shape& separate_sub_shape,
                                                      const ov::Shape& reduced_sub_shape_prod,
                                                      bool is_separate_first,
                                                      ov::NodeVector& subgraph_nodes) {
    ov::Shape new_shape{common_sub_shape.begin(), common_sub_shape.end()};

    // compute a product of a sub-shape for separate labels
    ov::Shape separate_sub_shape_prod = separate_sub_shape;
    if (!common_sub_shape.empty() && separate_sub_shape_prod.empty()) {
        // in this case new dimension corresponding to separate labels must be added
        // since MatMul operation is not possible to do without separate dimensions
        // if the common dimension presents
        separate_sub_shape_prod.push_back(1);
    } else if (!separate_sub_shape_prod.empty()) {
        // in this case compute a product of separate dimension sizes since they
        // must be presented with just one dimension for MatMul
        const auto prod = shape_size(separate_sub_shape_prod);
        separate_sub_shape_prod = {prod};
    }

    // form a new shape for input so that collapsed dimensions corresponding
    // to the common, separate and reduced dimensions are placed in the correct order
    if (is_separate_first) {
        new_shape.insert(new_shape.end(), separate_sub_shape_prod.begin(), separate_sub_shape_prod.end());
        new_shape.insert(new_shape.end(), reduced_sub_shape_prod.begin(), reduced_sub_shape_prod.end());
    } else {
        new_shape.insert(new_shape.end(), reduced_sub_shape_prod.begin(), reduced_sub_shape_prod.end());
        new_shape.insert(new_shape.end(), separate_sub_shape_prod.begin(), separate_sub_shape_prod.end());
    }

    // when new shape is equal to the current one, there is no need in reshape
    if (new_shape == input_node.get_shape()) {
        return input_node;
    }

    const auto new_shape_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape {new_shape.size()}, new_shape);
    const auto reshaped_input_op = std::make_shared<ov::op::v1::Reshape>(input_node, new_shape_const, false);
    subgraph_nodes.insert(subgraph_nodes.end(), {new_shape_const, reshaped_input_op});
    return reshaped_input_op->output(0);
}

LabelDimMap compute_label_dim_map(const ov::Rank& input_rank,
                                  const std::string& input_subscript) {
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
            std::vector<size_t> label_dims;
            label_dims.push_back(current_dim);
            resulted_map[label] = label_dims;
            ++current_dim;
        }
    }

    return resulted_map;
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
void transpose_input(ov::OutputVector& input_nodes, std::vector<std::string>& input_subscripts, const std::string& required_subscript, size_t input_ind,
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
    OPENVINO_ASSERT(labels.size() == required_labels.size());
    const auto label_dim_map = compute_label_dim_map(input_node.get_partial_shape().rank(), input_subscript);
    for (const auto& required_label : required_labels) {
        const auto label_dims_it = label_dim_map.find(required_label);
        OPENVINO_ASSERT(label_dims_it != label_dim_map.end());
        const auto& label_dims = label_dims_it->second;
        permutation.insert(permutation.end(), label_dims.begin(), label_dims.end());
    }

    // create a sub-graph for transposing into the required layout
    const auto permutation_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape {permutation.size()}, permutation);
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
/// \param      einsum_decompose_ptr    A pointer to Einsum decomposing pass
/// \param      input_nodes             A vector of input nodes to Einsum operation
/// \param      input_subscripts        A vector of corresponding subscripts for the input nodes
/// \param      output_subscript        The output subscript
/// \param      input_ind               An index of the input node for which it will check
/// dimensions to be reduced
/// \param      subgraph_nodes          A vector of operation nodes that is included into
/// a sub-graph decomposing Einsum that is needed for copy_runtime_info
///
void reduce_input(EinsumDecomposition *einsum_decompose_ptr,
                  ov::OutputVector& input_nodes, std::vector<std::string>& input_subscripts,
                  const std::string& output_subscript, size_t input_ind, ov::NodeVector& subgraph_nodes) {
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
    const auto axes_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape {reduced_axes.size()}, reduced_axes_vec);
    const auto reduce_sum = einsum_decompose_ptr->register_new_node<ov::op::v1::ReduceSum>(input_node, axes_const, false);

    // update a vector of inputs and input subscripts
    input_nodes[input_ind] = reduce_sum->output(0);
    input_subscripts[input_ind] = new_input_subscript;

    // update a vector of nodes for copy_runtime_info
    subgraph_nodes.insert(subgraph_nodes.end(), {axes_const, reduce_sum});
}

/// \brief      Broadcast input to a new shape. The MatMul operation requires the
/// same shape of both operands in the common (or batch) dimensions.
///
void broadcast_input(ov::OutputVector& inputs,
                     size_t input_ind,
                     const ov::Shape& new_common_shape,
                     const ov::Shape& separate_shape,
                     const ov::Shape& reduced_shape,
                     bool is_separate_first,
                     ov::NodeVector& subgraph_nodes) {
    OPENVINO_ASSERT(input_ind < inputs.size());
    const auto& input = inputs[input_ind];

    ov::Shape new_shape{new_common_shape.begin(), new_common_shape.end()};
    if (is_separate_first) {
        new_shape.insert(new_shape.end(), separate_shape.begin(), separate_shape.end());
        new_shape.insert(new_shape.end(), reduced_shape.begin(), reduced_shape.end());
    } else {
        new_shape.insert(new_shape.end(), reduced_shape.begin(), reduced_shape.end());
        new_shape.insert(new_shape.end(), separate_shape.begin(), separate_shape.end());
    }

    const ov::Shape old_shape = input.get_shape();
    if (old_shape == new_shape) {
        return;
    }
    const auto old_shape_size = old_shape.size();
    const auto new_shape_size = new_shape.size();
    OPENVINO_ASSERT(old_shape_size <= new_shape_size);

    const auto new_shape_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape {new_shape.size()}, new_shape);
    const auto broadcast = std::make_shared<ov::op::v3::Broadcast>(input, new_shape_const, ov::op::BroadcastType::NUMPY);

    inputs[input_ind] = broadcast->output(0);

    subgraph_nodes.insert(subgraph_nodes.end(), {new_shape_const, broadcast});
}

ov::Output<ov::Node> build_identity(const ov::Output<ov::Node>& input_node,
                                            const std::vector<size_t>& repeated_label_dims,
                                            ov::NodeVector& subgraph_nodes) {
    OPENVINO_ASSERT(repeated_label_dims.size() > 1);
    const auto input_shape = input_node.get_shape();
    ov::Shape identity_shape(input_shape.size(), 1);
    const size_t repeated_label_dim_size = input_shape[repeated_label_dims[0]];
    for (const auto dim : repeated_label_dims) {
        OPENVINO_ASSERT(dim < input_shape.size());
        OPENVINO_ASSERT(repeated_label_dim_size == input_shape[dim]);
        identity_shape[dim] = repeated_label_dim_size;
    }

    const size_t number_of_identity_elements = shape_size(identity_shape);
    std::vector<int> identity_values(number_of_identity_elements, 0);
    const size_t p = repeated_label_dim_size;
    if (p == 1) {
        identity_values[0] = 1;
    } else {
        const size_t n = repeated_label_dims.size();
        const size_t alpha = (static_cast<size_t>(std::pow(p, n)) - 1) / (p - 1);
        size_t offset = 0;
        for (size_t k = 0; k < p; ++k) {
            identity_values[offset] = 1;
            offset += alpha;
        }
    }
    const auto identity = ov::op::v0::Constant::create(input_node.get_element_type(), identity_shape, identity_values);
    subgraph_nodes.insert(subgraph_nodes.end(), {identity});
    return subgraph_nodes.back();
}

ov::Output<ov::Node> build_multi_identity(EinsumDecomposition* einsum_decompose_ptr,
                                                  const ov::Output<ov::Node>& input_node,
                                                  const std::vector<std::string>& repeated_labels,
                                                  const LabelDimMap& label_dim_map,
                                                  ov::NodeVector& subgraph_nodes) {
    OPENVINO_ASSERT(repeated_labels.size() > 0);

    const auto get_identity = [&](size_t idx) {
        const auto repeated_label_dims = label_dim_map.find(repeated_labels[idx]);
        OPENVINO_ASSERT(repeated_label_dims != label_dim_map.end());
        return build_identity(input_node, repeated_label_dims->second, subgraph_nodes);
    };

    // initially set multi-identity with identity for the first repeated label
    const auto multi_identity = get_identity(0);

    for (size_t label_ind = 1; label_ind < repeated_labels.size(); ++label_ind) {
        const auto identity = get_identity(label_ind);
        const auto mul = std::make_shared<ov::op::v1::Multiply>(multi_identity, identity, ov::op::AutoBroadcastType::NUMPY);
        subgraph_nodes.insert(subgraph_nodes.end(), {mul});
    }

    return subgraph_nodes.back();
}

/// \brief      Helper function to fill in the data needed for diagonal extraction  - result shape
/// and subscript, repeated labels, axes to reduce.
///
void prepare_diagonal_extraction_data(
        const ov::Shape& input_shape,
        const std::string& input_subscript,
        const LabelDimMap& label_dim_map,
        ov::Shape& result_shape,
        std::string& resultant_subscript,
        std::vector<std::string>& repeated_labels,
        ov::AxisSet& reduced_axes
) {
    static const std::string ellipsis = "...";
    const auto labels = ov::op::v7::Einsum::extract_labels(input_subscript);

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
            for (size_t dim_ind = 1; dim_ind < dims_size; ++dim_ind) {
                reduced_axes.insert(dims[dim_ind]);
            }
            // save only the first dimension corresponding to the repeated label
            dims = {dims[0]};
            repeated_labels.push_back(label);
        }
        resultant_subscript += label;
        for (const auto dim : dims) {
            OPENVINO_ASSERT(dim < input_shape.size());
            result_shape.push_back(input_shape[dim]);
        }
    }
}

void extract_diagonal(EinsumDecomposition* einsum_decompose_ptr,
                      ov::OutputVector& inputs,
                      std::vector<std::string>& input_subscripts,
                      size_t input_ind,
                      ov::NodeVector& subgraph_nodes) {
    // perform sanity check for arguments
    const auto num_inputs = inputs.size();
    OPENVINO_ASSERT(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
    OPENVINO_ASSERT(input_ind < num_inputs, "Input index is out of range.");

    const auto& input_node = inputs[input_ind];
    const auto& input_subscript = input_subscripts[input_ind];
    const auto input_shape = input_node.get_shape();

    const auto label_dim_map = compute_label_dim_map(input_node.get_partial_shape().rank(), input_subscript);
    ov::Shape result_shape;
    std::string resultant_subscript;
    std::vector<std::string> repeated_labels;
    ov::AxisSet reduced_axes;

    prepare_diagonal_extraction_data(input_shape, input_subscript, label_dim_map,
                                     result_shape, resultant_subscript, repeated_labels, reduced_axes);

    if (input_shape == result_shape) {
        return;
    }

    const auto multi_identity = build_multi_identity(einsum_decompose_ptr, input_node, repeated_labels, label_dim_map, subgraph_nodes);

    // multiply both operands with broadcasting
    const auto mul = std::make_shared<ov::op::v1::Multiply>(input_node, multi_identity, ov::op::AutoBroadcastType::NUMPY);
    subgraph_nodes.insert(subgraph_nodes.end(), {mul});

    const std::vector<int64_t> reduced_axes_vec{reduced_axes.cbegin(), reduced_axes.cend()};
    const auto axes_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape {reduced_axes.size()}, reduced_axes_vec);
    const auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(mul->output(0), axes_const, false);
    subgraph_nodes.insert(subgraph_nodes.end(), {axes_const, reduce_sum});

    inputs[input_ind] = reduce_sum->output(0);
    input_subscripts[input_ind] = resultant_subscript;
}

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
    static const std::string ellipsis = "...";

    size_t common_rank = common_labels.size();
    if (std::find(common_labels.begin(), common_labels.end(), ellipsis) != common_labels.end()) {
        OPENVINO_ASSERT(label_to_dim_map.find(ellipsis) != label_to_dim_map.end());
        common_rank += label_to_dim_map[ellipsis].size() - 1;
    }

    size_t sep_rank = sep_labels.size();
    if (std::find(sep_labels.begin(), sep_labels.end(), ellipsis) != sep_labels.end()) {
        OPENVINO_ASSERT(label_to_dim_map.find(ellipsis) != label_to_dim_map.end());
        sep_rank += label_to_dim_map[ellipsis].size() - 1;
    }

    size_t reduced_rank = reduced_labels.size();
    if (std::find(reduced_labels.begin(), reduced_labels.end(), ellipsis) != reduced_labels.end()) {
        OPENVINO_ASSERT(label_to_dim_map.find(ellipsis) != label_to_dim_map.end());
        reduced_rank += label_to_dim_map[ellipsis].size() - 1;
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

/// \brief      Contract two inputs of Einsum operation according to equation.
/// The result of the contraction is appended into input_nodes along with its subscript.
/// The input nodes for these two operands are removed from input_nodes along with their input
/// subscripts
///
/// \param      einsum_decompose_ptr    A pointer to Einsum decomposing pass
/// \param      input_nodes             A vector of input nodes to Einsum operation
/// \param      input_subscripts        A vector of corresponding subscripts for the input nodes
/// \param      output_subscript        The output subscript
/// \param      input_ind1              An index of the first operand
/// \param      input_ind2              An index of the second operand
/// \param      subgraph_nodes          A vector of operation nodes that is included into a
/// sub-graph decomposing Einsum that is needed for copy_runtime_info
///
void contract_two_inputs(EinsumDecomposition* einsum_decompose_ptr,
                         ov::OutputVector& input_nodes, std::vector<std::string>& input_subscripts,
                         const std::string& output_subscript, size_t input_ind1,
                         size_t input_ind2, ov::NodeVector& subgraph_nodes) {
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

    // extract diagonals in case repeated labels in the corresponding input subscripts
    extract_diagonal(einsum_decompose_ptr, input_nodes, input_subscripts, input_ind1, subgraph_nodes);
    extract_diagonal(einsum_decompose_ptr, input_nodes, input_subscripts, input_ind2, subgraph_nodes);

    // reduce dimensions for input operands if possible
    reduce_input(einsum_decompose_ptr, input_nodes, input_subscripts, output_subscript, input_ind1, subgraph_nodes);
    reduce_input(einsum_decompose_ptr, input_nodes, input_subscripts, output_subscript, input_ind2, subgraph_nodes);

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
    std::vector<std::string> common_labels, sep_labels1, sep_labels2, reduced_labels; // +++++
    for (size_t label_ind = 0; label_ind < labels1.size(); ++label_ind) {
        const auto& label = labels1[label_ind];
        auto iter = std::find(labels2.begin(), labels2.end(), label);
        if (iter != labels2.end()) {
            bool is_dim_reduced = is_dimension_reduced(input_subscripts, output_subscript, label, {input_ind1, input_ind2});
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

        const auto separate_labels1 = ov::op::v7::Einsum::extract_labels(separate_part1);
        const auto separate_labels2 = ov::op::v7::Einsum::extract_labels(separate_part2);
        auto label_to_dim_map1 = compute_label_dim_map(input_node1.get_partial_shape().rank(), input_subscript1);
        auto label_to_dim_map2 = compute_label_dim_map(input_node2.get_partial_shape().rank(), input_subscript2);

        // unsqueeze the first operand with new dimensions in the tail
        // and the number of them is equal to the number of separate labels in the second
        // subscript
        int64_t unsqueeze_dim = input_node1.get_shape().size();
        std::vector<int64_t> unsqueeze_axis1;
        std::vector<int64_t> unsqueeze_axis2;
        for (const auto& sep_label2 : separate_labels2) {
            OPENVINO_ASSERT(label_to_dim_map2.find(sep_label2) != label_to_dim_map2.end());
            const auto label_dims = label_to_dim_map2[sep_label2];
            for (size_t dim_ind = 0; dim_ind < label_dims.size(); ++dim_ind) {
                unsqueeze_axis1.push_back(unsqueeze_dim + static_cast<int64_t>(dim_ind));
            }
            ++unsqueeze_dim;
        }
        for (const auto& sep_label1 : separate_labels1) {
            OPENVINO_ASSERT(label_to_dim_map1.find(sep_label1) != label_to_dim_map1.end());
            const auto label_dims = label_to_dim_map1[sep_label1];
            for (const auto label_dim : label_dims) {
                unsqueeze_axis2.push_back(label_dim);
            }
        }

        // unsqueeze input operands for elementwise-multiplication with broadcasting
        auto unsqueeze_output1 = unsqueeze_input(input_node1, unsqueeze_axis1, subgraph_nodes);
        auto unsqueeze_output2 = unsqueeze_input(input_node2, unsqueeze_axis2, subgraph_nodes);

        // multiply both operands with broadcasting
        auto mul = std::make_shared<ov::op::v1::Multiply>(unsqueeze_output1, unsqueeze_output2, ov::op::AutoBroadcastType::NUMPY);

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
    auto int_subscript1 = generate_grouping_subscript(input_subscript1, common_labels_inds1, separate_labels_inds1,
        reduced_labels_inds1, is_separate_first1);
    transpose_input(input_nodes, input_subscripts, int_subscript1, input_ind1, subgraph_nodes);
    bool is_separate_first2 = false;
    auto int_subscript2 = generate_grouping_subscript(input_subscript2, common_labels_inds2, separate_labels_inds2,
        reduced_labels_inds2, is_separate_first2);
    transpose_input(input_nodes, input_subscripts, int_subscript2, input_ind2, subgraph_nodes);

    // step 2. reshape both operands so that separate labels and reduced labels are represented
    // with just one dimension this is needed by MatMul operation requirement to operands
    // format. For example, the shape must be in a format [B1, ..., Bm, X1, Y] or [B1, ..., Bm,
    // Y, X2], where B1, ..., Bm are common dimensions, X1 and X2 are collapsed dimensions
    // for separate labels and Y is collapsed dimension for reduced labels

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

    const auto input_shape1 = input_node1.get_shape();
    const auto input_shape2 = input_node2.get_shape();

    ov::PartialShape common_sub_shape1 = compute_sub_shape(input_shape1, common_dims_begin, common_dims_end);
    ov::PartialShape common_sub_shape2 = compute_sub_shape(input_shape2, common_dims_begin2, common_dims_end2);
    const auto reduced_sub_shape_prod = compute_sub_shape(input_shape1, reduced_dims_begin, reduced_dims_end, true);
    const auto reduced_sub_shape = compute_sub_shape(input_shape1, reduced_dims_begin, reduced_dims_end);
    const auto separate1_sub_shape = compute_sub_shape(input_shape1, separate1_dims_begin, separate1_dims_end);
    const auto separate2_sub_shape = compute_sub_shape(input_shape2, separate2_dims_begin, separate2_dims_end);

    // broadcast both inputs to have common sub-shape broadcasted that is needed in case of ellipsis among the common labels
    ov::PartialShape::broadcast_merge_into(common_sub_shape1, common_sub_shape2, ov::op::AutoBroadcastType::NUMPY);
    const auto common_sub_shape = common_sub_shape1.get_shape();
    broadcast_input(input_nodes,
                    input_ind1,
                    common_sub_shape,
                    separate1_sub_shape,
                    reduced_sub_shape,
                    is_separate_first1,
                    subgraph_nodes);
    broadcast_input(input_nodes,
                    input_ind2,
                    common_sub_shape,
                    separate2_sub_shape,
                    reduced_sub_shape,
                    is_separate_first2,
                    subgraph_nodes);

    const auto matmul_operand1 = reshape_input_for_matmul(input_node1,
                common_sub_shape,
                separate1_sub_shape,
                reduced_sub_shape_prod,
                is_separate_first1,
                subgraph_nodes);
    const auto matmul_operand2 = reshape_input_for_matmul(input_node2,
                common_sub_shape,
                separate2_sub_shape,
                reduced_sub_shape_prod,
                is_separate_first2,
                subgraph_nodes);

    // step 3. apply MatMul operation for formatted inputs
    const bool transpose_a = (is_separate_first1 ? false : true);
    const bool transpose_b = (is_separate_first2 ? true : false);
    const auto matmul = std::make_shared<ov::op::v0::MatMul>(matmul_operand1, matmul_operand2, transpose_a, transpose_b);

    // step 4. reshape back by unrolling dimensions corresponding to separate labels if needed
    // now dimensions corresponding to reduced labels are reduced by the MatMul operation
    common_part = "";
    for (const auto& common_label : common_labels) {
        common_part += common_label;
    }
    const std::string resultant_subscript = common_part + separate_part1 + separate_part2;
    ov::Shape back_shape;
    back_shape.insert(back_shape.end(), common_sub_shape.begin(), common_sub_shape.end());
    back_shape.insert(back_shape.end(), separate1_sub_shape.begin(), separate1_sub_shape.end());
    back_shape.insert(back_shape.end(), separate2_sub_shape.begin(), separate2_sub_shape.end());

    const auto new_shape_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape {back_shape.size()}, back_shape);
    const auto reshape_result_op = std::make_shared<ov::op::v1::Reshape>(matmul->output(0), new_shape_const, false);
    subgraph_nodes.insert(subgraph_nodes.end(), {new_shape_const, reshape_result_op});

    // update input operand and input subscript for Einsum operation
    update_operands(input_nodes, input_subscripts, input_ind1, input_ind2, reshape_result_op->output(0), resultant_subscript);

    // update a vector of nodes for copy_runtime_info
    subgraph_nodes.insert(subgraph_nodes.end(), {matmul});
}
}  // namespace

EinsumDecomposition::EinsumDecomposition() {
    ov::matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& m) {
        auto einsum_node = ov::as_type_ptr<ov::op::v7::Einsum>(m.get_match_root());
        if (!einsum_node) {
            return false;
        }

        if (einsum_node->is_dynamic()) {
            return false;
        }

        auto equation = einsum_node->get_equation();
        std::vector<std::string> input_subscripts;
        std::string output_subscript;
        ov::op::v7::Einsum::parse_equation(equation, input_subscripts, output_subscript);

        // create a list of input nodes with preserving their order
        // and a vector of sub-graph nodes for copy_runtime_info
        ov::OutputVector input_nodes = einsum_node->input_values();
        ov::NodeVector subgraph_nodes;

        // compute einsum path that is used to contract a pair of operands
        // in more optimal order
        auto einsum_path = compute_einsum_path(einsum_node);

        // contract inputs by Einsum until just one is remained
        for (auto const& inds_pair : einsum_path) {
            contract_two_inputs(this, input_nodes, input_subscripts, output_subscript, inds_pair.first, inds_pair.second, subgraph_nodes);
        }

        // extract diagonal for the single operand
        extract_diagonal(this, input_nodes, input_subscripts, 0, subgraph_nodes);

        // reduce dimensions for the remained input node
        OPENVINO_ASSERT(input_nodes.size() == 1);
        reduce_input(this, input_nodes, input_subscripts, output_subscript, 0, subgraph_nodes);

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

    const auto einsum = ov::pass::pattern::wrap_type<ov::op::v7::Einsum>();
    const auto matcher = std::make_shared<ov::pass::pattern::Matcher>(einsum, "EinsumDecompositionGpu");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu

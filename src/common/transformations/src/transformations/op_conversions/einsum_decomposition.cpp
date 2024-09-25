// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/einsum_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/einsum.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {
/// \brief      Check if the EinsumDecomposition transformation is applicable to a given Einsum.
/// The transformation is applicable if input subscript does not have repeated labels and ellipsis.
///
/// \param      subscript          A subscript to check its format
///
/// \return     true - applicable, false - not applicable
///
bool is_subscript_applicable(const std::string& subscript) {
    auto labels = ov::op::v7::Einsum::extract_labels(subscript);
    auto unique_labels = std::unordered_set<std::string>(labels.begin(), labels.end());
    return std::find(labels.begin(), labels.end(), "...") == labels.end() && unique_labels.size() == labels.size();
}

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
                                              const ov::OutputVector& reduced_sub_shape_prod,
                                              bool is_separate_first,
                                              ov::NodeVector& subgraph_nodes) {
    ov::OutputVector new_shape_parts;
    new_shape_parts.insert(new_shape_parts.end(), common_sub_shape.begin(), common_sub_shape.end());

    // compute a product of a sub-shape for separate labels
    ov::OutputVector separate_parts;
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
        auto reduce_axis_const = ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape{1}, {0});
        auto separate_shape_prod =
            std::make_shared<ov::op::v1::ReduceProd>(separate_sub_shape[0], reduce_axis_const, true);
        separate_parts.push_back(separate_shape_prod->output(0));
        subgraph_nodes.insert(subgraph_nodes.end(), {reduce_axis_const, separate_shape_prod});
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

    // if new shape is possible to compute on the shape infer stage, insert Constant node immediatelly
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
    auto num_inputs = input_nodes.size();
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
    auto labels = ov::op::v7::Einsum::extract_labels(input_subscript);
    auto required_labels = ov::op::v7::Einsum::extract_labels(required_subscript);
    OPENVINO_ASSERT(labels.size() == required_labels.size());
    for (const auto& required_label : required_labels) {
        auto it = std::find(labels.begin(), labels.end(), required_label);
        OPENVINO_ASSERT(it != labels.end());
        int64_t found_index = static_cast<int64_t>(it - labels.begin());
        permutation.push_back(found_index);
    }

    // create a sub-graph for transposing into the required layout
    const auto& input_node = input_nodes[input_ind];
    auto permutation_const =
        ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape{permutation.size()}, permutation);
    auto transpose = std::make_shared<ov::op::v1::Transpose>(input_node, permutation_const);

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
void reduce_input(ov::pass::EinsumDecomposition* einsum_decompose_ptr,
                  ov::OutputVector& input_nodes,
                  std::vector<std::string>& input_subscripts,
                  const std::string& output_subscript,
                  size_t input_ind,
                  ov::NodeVector& subgraph_nodes) {
    // perform sanity check for arguments
    auto num_inputs = input_nodes.size();
    OPENVINO_ASSERT(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
    OPENVINO_ASSERT(input_ind < num_inputs, "Input index is out of range.");

    std::vector<int64_t> reduced_axes;
    auto labels = ov::op::v7::Einsum::extract_labels(input_subscripts[input_ind]);
    std::string new_input_subscript = "";
    for (size_t dim_ind = 0; dim_ind < labels.size(); ++dim_ind) {
        const auto& label = labels[dim_ind];

        // check if the current label is met in the other input subscripts
        // or the output subscript
        bool is_dim_reduced = is_dimension_reduced(input_subscripts, output_subscript, label, {input_ind});

        // if label is not met, dimension corresponding to the label is to reduce
        if (is_dim_reduced) {
            reduced_axes.push_back(dim_ind);
        } else {
            new_input_subscript += label;
        }
    }

    if (reduced_axes.size() == 0) {
        // there is no axis to reduce
        return;
    }

    // reduce by summed up elements along dimension for which label is met just once
    const auto& input_node = input_nodes[input_ind];
    auto axes_const =
        ov::op::v0::Constant::create(ov::element::Type_t::i64, ov::Shape{reduced_axes.size()}, reduced_axes);
    auto reduce_sum = einsum_decompose_ptr->register_new_node<ov::op::v1::ReduceSum>(input_node, axes_const, false);

    // update a vector of inputs and input subscripts
    input_nodes[input_ind] = reduce_sum->output(0);
    input_subscripts[input_ind] = new_input_subscript;

    // update a vector of nodes for copy_runtime_info
    subgraph_nodes.insert(subgraph_nodes.end(), {axes_const, reduce_sum});
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
void contract_two_inputs(ov::pass::EinsumDecomposition* einsum_decompose_ptr,
                         ov::OutputVector& input_nodes,
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
            } else {
                common_labels_inds1.push_back(static_cast<int64_t>(label_ind));
                common_labels_inds2.push_back(static_cast<int64_t>(iter - labels2.begin()));
            }
        } else {
            separate_part1 += label;
            separate_labels_inds1.push_back(static_cast<int64_t>(label_ind));
        }
    }
    for (size_t label_ind = 0; label_ind < labels2.size(); ++label_ind) {
        const auto& label = labels2[label_ind];
        auto iter = std::find(labels1.begin(), labels1.end(), label);
        if (iter == labels1.end()) {
            separate_part2 += label;
            separate_labels_inds2.push_back(static_cast<int64_t>(label_ind));
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
        int64_t unsqueeze_dim = labels1.size();
        std::vector<int64_t> unsqueeze_axis1;
        for (size_t label_ind = 0; label_ind < separate_labels_inds2.size(); ++label_ind) {
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

    // step 2. reshape both operands so that separate labels and reduced labels are represented
    // with just one dimension this is needed by MatMul operation requirement to operands
    // format. For example, the shape must be in a format [B1, ..., Bm, X1, Y] or [B1, ..., Bm,
    // Y, X2], where B1, ..., Bm are common dimensions, X1 and X2 are collapsed dimensions
    // for separate labels and Y is collapsed dimension for reduced labels
    // this step is not needed for the operand if it satisfies to one of the requirements:
    // 1. there is just one separate dimension and just one reduced dimension
    // 2. there is no separate dimension, no common dimensions, and just one reduced dimension
    bool no_reshape_for_matmul1 =
        (reduced_labels_inds1.size() == 1 && separate_labels_inds1.size() == 1) ||
        (reduced_labels_inds1.size() == 1 && common_labels_inds1.size() == 0 && separate_labels_inds1.size() == 0);
    bool no_reshape_for_matmul2 =
        (reduced_labels_inds2.size() == 1 && separate_labels_inds2.size() == 1) ||
        (reduced_labels_inds2.size() == 1 && common_labels_inds2.size() == 0 && separate_labels_inds2.size() == 0);
    // reshape back after MatMul is not needed if one of two requrements satisfies for both operands:
    // 1. there is just one separate dimension
    // 2. there is no separate dimension and no common dimensions present.
    // If there is no separate dimension and common dimensions present, reshape is needed
    // because auxiliary separate dimension has been added by Unsqueeze operation
    // in the purpose for MatMul
    bool no_reshape_back1 =
        (separate_labels_inds1.size() == 1) || (common_labels_inds1.size() == 0 && separate_labels_inds1.size() == 0);
    bool no_reshape_back2 =
        (separate_labels_inds2.size() == 1) || (common_labels_inds2.size() == 0 && separate_labels_inds2.size() == 0);
    bool no_reshape_after_matmul = no_reshape_back1 && no_reshape_back2;

    auto matmul_operand1 = input_node1;
    auto matmul_operand2 = input_node2;
    int64_t common_dims_begin = 0;
    int64_t common_dims_end = common_labels_inds1.size();
    ov::OutputVector common_sub_shape, separate1_sub_shape, separate2_sub_shape;
    if (no_reshape_for_matmul1 == false || no_reshape_for_matmul2 == false) {
        auto data_shape1 = std::make_shared<ov::op::v3::ShapeOf>(input_node1);
        common_sub_shape = compute_sub_shape(data_shape1, common_dims_begin, common_dims_end, subgraph_nodes);
        int64_t reduced_dims_begin = (is_separate_first1 ? common_labels_inds1.size() + separate_labels_inds1.size()
                                                         : common_labels_inds1.size());
        int64_t reduced_dims_end = reduced_dims_begin + reduced_labels_inds1.size();
        auto reduced_sub_shape_prod =
            compute_sub_shape(data_shape1, reduced_dims_begin, reduced_dims_end, subgraph_nodes, true);

        if (no_reshape_for_matmul1 == false || no_reshape_after_matmul == false) {
            int64_t separate1_dims_begin =
                (is_separate_first1 ? common_labels_inds1.size()
                                    : common_labels_inds1.size() + reduced_labels_inds1.size());
            int64_t separate1_dims_end = separate1_dims_begin + separate_labels_inds1.size();
            separate1_sub_shape =
                compute_sub_shape(data_shape1, separate1_dims_begin, separate1_dims_end, subgraph_nodes);
            matmul_operand1 = reshape_input_for_matmul(input_node1,
                                                       common_sub_shape,
                                                       separate1_sub_shape,
                                                       reduced_sub_shape_prod,
                                                       is_separate_first1,
                                                       subgraph_nodes);
        }

        if (no_reshape_for_matmul2 == false || no_reshape_after_matmul == false) {
            auto data_shape2 = std::make_shared<ov::op::v3::ShapeOf>(input_node2);
            int64_t separate2_dims_begin =
                (is_separate_first2 ? common_labels_inds2.size()
                                    : common_labels_inds2.size() + reduced_labels_inds2.size());
            int64_t separate2_dims_end = separate2_dims_begin + separate_labels_inds2.size();
            separate2_sub_shape =
                compute_sub_shape(data_shape2, separate2_dims_begin, separate2_dims_end, subgraph_nodes);
            matmul_operand2 = reshape_input_for_matmul(input_node2,
                                                       common_sub_shape,
                                                       separate2_sub_shape,
                                                       reduced_sub_shape_prod,
                                                       is_separate_first2,
                                                       subgraph_nodes);
            subgraph_nodes.insert(subgraph_nodes.end(), {data_shape2});
        }
        subgraph_nodes.insert(subgraph_nodes.end(), {data_shape1});
    }

    // step 3. apply MatMul operation for formatted inputs
    bool transpose_a = (is_separate_first1 ? false : true);
    bool transpose_b = (is_separate_first2 ? true : false);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(matmul_operand1, matmul_operand2, transpose_a, transpose_b);

    // step 4. reshape back by unrolling dimensions corresponding to separate labels if needed
    // now dimensions corresponding to reduced labels are reduced by the MatMul operation
    std::string resultant_subscript =
        input_subscript1.substr(common_dims_begin, common_dims_end) + separate_part1 + separate_part2;
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

        // if new shape is possible to compute on the shape infer stage, insert Constant node immediatelly
        // in order to prevent repeated computing during constant-folding pass
        std::shared_ptr<ov::op::v1::Reshape> result_op;
        if (auto new_shape_const = ov::util::get_constant_from_source(result_shape_op)) {
            result_op = std::make_shared<ov::op::v1::Reshape>(matmul->output(0), new_shape_const, false);
            subgraph_nodes.insert(subgraph_nodes.end(), {new_shape_const});
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
}  // namespace

ov::pass::EinsumDecomposition::EinsumDecomposition() {
    // NOTE: The transformation is applicable if Einsum equation does not contain ellipsis label
    // and does not contain subscripts with repeated labels.
    // For example, the transformation is applicable to Einsum with equation="abc,bd->ad"
    // but not applicable to a case with equation="aabc,bd->ad" due to repeated labels
    // in the first input subscript.
    MATCHER_SCOPE(EinsumDecomposition);
    auto einsum = ov::pass::pattern::wrap_type<ov::op::v7::Einsum>();
    matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& m) {
        auto einsum_node = ov::as_type_ptr<ov::op::v7::Einsum>(m.get_match_root());
        if (!einsum_node) {
            return false;
        }

        auto equation = einsum_node->get_equation();
        std::vector<std::string> input_subscripts;
        std::string output_subscript;
        ov::op::v7::Einsum::parse_equation(equation, input_subscripts, output_subscript);

        // check that the transformation is applicable
        if (std::any_of(input_subscripts.cbegin(), input_subscripts.cend(), [](const std::string& subscript) {
                return is_subscript_applicable(subscript) == false;
            })) {
            return false;
        }

        // create a list of input nodes with preserving their order
        // and a vector of sub-graph nodes for copy_runtime_info
        ov::OutputVector input_nodes = einsum_node->input_values();
        ov::NodeVector subgraph_nodes;

        // compute einsum path that is used to contract a pair of operands
        // in more optimal order
        auto einsum_path = compute_einsum_path(einsum_node);

        // contract inputs by Einsum until just one is remained
        for (auto const& inds_pair : einsum_path) {
            contract_two_inputs(this,
                                input_nodes,
                                input_subscripts,
                                output_subscript,
                                inds_pair.first,
                                inds_pair.second,
                                subgraph_nodes);
        }

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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(einsum, matcher_name);
    register_matcher(m, callback);
}

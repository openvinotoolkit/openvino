// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/op_conversions/einsum_decomposition.hpp"

#include <memory>
#include <transformations/utils/utils.hpp>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

namespace {
    /// \brief      Check if the transformation is applicable to Einsum with a given equation
    ///
    /// \param      subscript          A subscript to check its format
    ///
    /// \return     true - applicable, false - not applicable
    ///
    bool is_subscript_applicable(const std::string& subscript) {
        auto labels = ngraph::op::v7::Einsum::extract_labels(subscript);
        std::unordered_set<std::string> met_labels;
        for (auto const& label : labels) {
            if (label == "..." || met_labels.find(label) != met_labels.end()) {
                return false;
            }
            met_labels.insert(label);
        }
        return true;
    }

    /// \brief      Compute einsum_path for a given Einsum node meaning that the (pseudo-)optimal order of operands contraction
    /// in terms of performance and memory consumption
    ///
    /// \param      einsum_node         An input Einsum node
    ///
    /// \return     a vector of pairs with input indices assuming that the intermediate result is appended in the tail
    ///
    std::vector<std::pair<size_t, size_t>> compute_einsum_path(std::shared_ptr<const ngraph::opset7::Einsum> einsum_node) {
        // TODO: implement greedy algorithm for finding pseudo-optimal einsum_path
        std::vector<std::pair<size_t, size_t>> einsum_path;
        auto num_inputs = einsum_node->get_input_size();
        for (size_t input_ind = num_inputs - 1; input_ind > 0; --input_ind) {
            einsum_path.push_back(std::make_pair(0, input_ind));
        }
        return einsum_path;
    }

    /// \brief      Transpose one of the Einsum inputs to layout required through subscript
    ///
    /// \param      input_nodes         A vector of output nodes
    /// \param      input_subscripts    A vector of corresponding subscripts for output nodes
    /// \param      required_subscript  The required subscript that defines layout to which the input is to transpose
    /// \param      input_ind           An index of the input in the vector
    /// \param      subgraph_nodes      A vector of operation nodes that is included into a sub-graph decomposing Einsum
    /// that is needed for copy_runtime_info
    ///
    void transpose_input(ngraph::OutputVector& input_nodes, std::vector<std::string>& input_subscripts,
        const std::string& required_subscript, size_t input_ind, ngraph::NodeVector& subgraph_nodes) {
        // perform sanity check for arguments
        auto num_inputs = input_nodes.size();
        NGRAPH_CHECK(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
        NGRAPH_CHECK(input_ind < num_inputs, "Input index is out of range.");

        // generate permutation vector by searching for bijection between input_subscripts and required_subscript
        std::vector<int64_t> permutation;
        const auto& input_subscript = input_subscripts[input_ind];
        auto labels = ngraph::op::v7::Einsum::extract_labels(input_subscript);
        auto required_labels = ngraph::op::v7::Einsum::extract_labels(required_subscript);
        NGRAPH_CHECK(labels.size() == required_labels.size());
        bool is_identity = true;
        for (size_t label_ind = 0; label_ind < labels.size(); ++label_ind) {
            const auto& required_label = required_labels[label_ind];
            auto it = std::find(labels.begin(), labels.end(), required_label);
            NGRAPH_CHECK(it != labels.end());
            int64_t found_index = static_cast<int64_t>(it - labels.begin());
            permutation.push_back(found_index);

            // check that bijection is not identity
            if (found_index != label_ind) {
                is_identity = false;
            }
        }

        if (is_identity) {
            // the transpose is not required if permutation is equal to [0, 1, ..., n-1] where n is input rank
            return;
        }

        // create a sub-graph for transposing into the required layout
        const auto& input_node = input_nodes[input_ind];
        auto permutation_const = ngraph::opset7::Constant::create(ngraph::element::Type_t::i64,
            ngraph::Shape{ permutation.size() }, permutation);
        auto transpose = std::make_shared<ngraph::opset7::Transpose>(input_node, permutation_const);

        // update a vector of inputs and input subscripts
        input_nodes[input_ind] = transpose->output(0);
        input_subscripts[input_ind] = required_subscript;

        // update a vector of nodes for copy_runtime_info
        subgraph_nodes.insert(subgraph_nodes.end(), { permutation_const, transpose });
    }

    /// \brief      Find labels (in a given input subscript) that are met once in the equation
    /// and reduce the corresponding dimensions
    ///
    /// \param      input_nodes         A vector of output nodes
    /// \param      input_subscripts    A vector of corresponding subscripts for output nodes
    /// \param      output_subscript    The output subscript
    /// \param      input_ind           An index of the input in the vector
    /// \param      subgraph_nodes      A vector of operation nodes that is included into a sub-graph decomposing Einsum
    /// that is needed for copy_runtime_info
    ///
    void reduce_dimensions(ngraph::OutputVector& input_nodes, std::vector<std::string>& input_subscripts,
        const std::string& output_subscript, size_t input_ind, ngraph::NodeVector& subgraph_nodes) {
        // perform sanity check for arguments
        auto num_inputs = input_nodes.size();
        NGRAPH_CHECK(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
        NGRAPH_CHECK(input_ind < num_inputs, "Input index is out of range.");

        std::vector<size_t> axes;
        auto labels = ngraph::op::v7::Einsum::extract_labels(input_subscripts[input_ind]);
        std::string new_input_subscript = "";
        for (size_t dim_ind = 0; dim_ind < labels.size(); ++dim_ind) {
            const auto& label = labels[dim_ind];
            // check if the current label is met in the other input subscripts or the output subscript
            bool is_label_met = false;
            for (size_t other_input_ind = 0; other_input_ind < num_inputs; ++other_input_ind) {
                const auto& other_input_subscript = input_subscripts[other_input_ind];
                if (other_input_ind != input_ind && other_input_subscript.find(label) != std::string::npos) {
                    is_label_met = true;
                    break;
                }
            }
            if (output_subscript.find(label) != std::string::npos) {
                is_label_met = true;
            }

            // if label is not met, dimension corresponding to the label is a candidate for reduction
            if (is_label_met == false) {
                axes.push_back(dim_ind);
            }
            else {
                new_input_subscript += label;
            }
        }

        if (axes.size() == 0)
        {
            // there is no axis to reduce
            return;
        }

        // reduce by summed up elements along dimension for which label is met just once
        const auto& input_node = input_nodes[input_ind];
        auto axes_const = ngraph::opset7::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{ axes.size() }, axes);
        auto reduce_sum = std::make_shared<ngraph::opset7::ReduceSum>(input_node, axes_const, false);

        // update a vector of inputs and input subscripts
        input_nodes[input_ind] = reduce_sum->output(0);
        input_subscripts[input_ind] = new_input_subscript;

        // update a vector of nodes for copy_runtime_info
        subgraph_nodes.insert(subgraph_nodes.end(), { axes_const, reduce_sum });
    }

    void contract_two_operands(ngraph::OutputVector& input_nodes, std::vector<std::string>& input_subscripts,
        const std::string& output_subscript, size_t input_ind1, size_t input_ind2, ngraph::NodeVector& subgraph_nodes) {
        // perform sanity check for arguments
        auto num_inputs = input_nodes.size();
        NGRAPH_CHECK(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
        NGRAPH_CHECK(input_ind1 < num_inputs&& input_ind2 < num_inputs&&
            input_ind1 != input_ind2, "Input index is out of range.");

        // assume that input_ind1 is less input_ind2 without loss of generality
        if (input_ind2 < input_ind1) {
            std::swap(input_ind1, input_ind2);
        }

        // reduce dimensions of operands if possible
        reduce_dimensions(input_nodes, input_subscripts, output_subscript, input_ind1, subgraph_nodes);
        reduce_dimensions(input_nodes, input_subscripts, output_subscript, input_ind2, subgraph_nodes);

        // 1. compute new subscript (convinient_subscript) for the second operand that is required for transposing
        // to convinient layout for further unsqueezing and elementwise-multiplication with broadcasting
        // 2. compute unsqueezing axes for both operands to make them bidirectionally broadcasted
        // 3. compute a new subscript (resultant_subscript) that will correspond to the resultant
        // of the elementwise-multiplication
        auto& input_subscript1 = input_subscripts[input_ind1];
        auto labels1 = ngraph::op::v7::Einsum::extract_labels(input_subscript1);
        auto& input_subscript2 = input_subscripts[input_ind2];
        auto labels2 = ngraph::op::v7::Einsum::extract_labels(input_subscript2);
        std::string intercepted_part = "";
        std::vector<int64_t> unsqueeze_axes2;
        for (size_t dim_ind = 0; dim_ind < labels1.size(); ++dim_ind) {
            const auto& label = labels1[dim_ind];
            if (input_subscript2.find(label) != std::string::npos) {
                intercepted_part += label;
            }
            else {
                unsqueeze_axes2.push_back(static_cast<int64_t>(dim_ind));
            }
        }
        std::string not_found = "";
        std::vector<int64_t> unsqueeze_axes1;
        int64_t unsqueeze_dim = static_cast<int64_t>(labels1.size());
        for (const auto& label : labels2) {
            if (input_subscript1.find(label) == std::string::npos) {
                not_found += label;
                unsqueeze_axes1.push_back(unsqueeze_dim++);
            }
        }
        std::string convinient_subscript = intercepted_part + not_found;
        std::string resultant_subscript = input_subscript1 + not_found;

        // transpose the second operand in order to get the convinient layout for further unsqueezing
        transpose_input(input_nodes, input_subscripts, convinient_subscript, input_ind2, subgraph_nodes);

        // unsqueeze input operands for elementwise-multiplication with broadcasting
        const auto& input_node1 = input_nodes[input_ind1];
        const auto& input_node2 = input_nodes[input_ind2];
        auto unsqueeze_axes1_const = ngraph::opset7::Constant::create(ngraph::element::Type_t::i64,
            ngraph::Shape{ unsqueeze_axes1.size() }, unsqueeze_axes1);
        auto unsqueeze1 = std::make_shared<ngraph::opset7::Unsqueeze>(input_node1, unsqueeze_axes1_const);
        auto unsqueeze_axes2_const = ngraph::opset7::Constant::create(ngraph::element::Type_t::i64,
            ngraph::Shape{ unsqueeze_axes2.size() }, unsqueeze_axes2);
        auto unsqueeze2 = std::make_shared<ngraph::opset7::Unsqueeze>(input_node2, unsqueeze_axes2_const);

        // multiply both operands with broadcasting
        auto mul = std::make_shared<ngraph::opset7::Multiply>(unsqueeze1, unsqueeze2, ngraph::op::AutoBroadcastSpec::NUMPY);

        // update a vector of inputs and input subscripts
        // remember that input_ind1 is less than input_ind2
        input_nodes.erase(input_nodes.begin() + input_ind2);
        input_nodes.erase(input_nodes.begin() + input_ind1);
        input_nodes.push_back(mul->output(0));
        input_subscripts.erase(input_subscripts.begin() + input_ind2);
        input_subscripts.erase(input_subscripts.begin() + input_ind1);
        input_subscripts.push_back(resultant_subscript);

        // update a vector of nodes for copy_runtime_info
        subgraph_nodes.insert(subgraph_nodes.end(), { unsqueeze_axes1_const, unsqueeze1, unsqueeze_axes2_const, unsqueeze2, mul });
    }
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::EinsumDecomposition, "EinsumDecomposition", 0);

ngraph::pass::EinsumDecomposition::EinsumDecomposition() {
    MATCHER_SCOPE(EinsumDecomposition);
    auto einsum = ngraph::pattern::wrap_type<opset7::Einsum>();
    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto einsum_node = std::dynamic_pointer_cast<ngraph::opset7::Einsum> (m.get_match_root());
        if (!einsum_node || transformation_callback(einsum_node)) {
            return false;
        }

        auto equation = einsum_node->get_equation();
        std::vector<std::string> input_subscripts;
        auto num_inputs = input_subscripts.size();
        std::string output_subscript;
        ngraph::op::v7::Einsum::parse_equation(equation, input_subscripts, output_subscript);

        // check that the transformation is applicable
        for (auto const& input_subscript : input_subscripts) {
            if (is_subscript_applicable(input_subscript) == false) {
                return false;
            }
        }

        // create a list of input nodes with preserving their order and a vector of sub-graph nodes for copy_runtime_info
        ngraph::OutputVector input_nodes = einsum_node->input_values();
        ngraph::NodeVector subgraph_nodes;

        // compute einsum path
        auto einsum_path = compute_einsum_path(einsum_node);

        // contract inputs by Einsum until just one is remained
        for (auto const& inds_pair : einsum_path) {
            contract_two_operands(input_nodes, input_subscripts, output_subscript,
                inds_pair.first, inds_pair.second, subgraph_nodes);
        }

        // reduce dimensions for the remained input node after the transformation of the others
        NGRAPH_CHECK(input_nodes.size() == 1);
        reduce_dimensions(input_nodes, input_subscripts, output_subscript, 0, subgraph_nodes);

        // transpose dimensions to layout required by the output subscript
        transpose_input(input_nodes, input_subscripts, output_subscript, 0, subgraph_nodes);

        // replace the original Einsum node with the last node from decomposing sub-graph
        // preserve the original node name
        auto last_node = input_nodes[0].get_node_shared_ptr();
        last_node->set_friendly_name(einsum_node->get_friendly_name());
        ngraph::copy_runtime_info(einsum_node, subgraph_nodes);
        ngraph::replace_node(einsum_node, last_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(einsum, matcher_name);
    register_matcher(m, callback);
}

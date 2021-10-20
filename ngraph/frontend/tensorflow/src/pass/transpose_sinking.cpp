// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_sinking.hpp"

#include <openvino/opsets/opset8.hpp>

#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace opset8;

using TransposeMap = unordered_map<string, shared_ptr<Transpose>>;

template <class T>
static T apply_permutation(const T& input, AxisVector order) {
    T output(input.size());
    for (size_t i = 0; i < order.size(); i++) {
        output[i] = input.at(order.at(i));
    }
    return output;
}

static AxisVector permutation_to_default_order(const AxisVector& axis_order) {
    AxisVector out(axis_order.size());
    for (size_t i = 0; i < axis_order.size(); i++) {
        out.at(axis_order[i]) = i;
    }
    return out;
}

template <typename T>
static string describe(shared_ptr<Node> node) {
    // ensure that it's either a reshape or a transpose
    // TODO: use static_assert
    if (!(std::is_base_of<Reshape, T>::value || std::is_base_of<Transpose, T>::value)) {
        throw runtime_error("describe template specialization has to be either reshape or "
                            "transpose");
    }
    stringstream ss;
    auto transpose = as_type_ptr<T>(node);
    auto const1 = as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
    ss << transpose->get_name() << " ( axis order = " << ngraph::vector_to_string(const1->get_axis_vector_val())
       << " , shape = " << ngraph::vector_to_string(transpose->get_shape()) << " ) "
       << " , input = " << transpose->input_value(0).get_node()->get_name();
    return ss.str();
}

static shared_ptr<Transpose> make_transpose(const Output<Node>& arg, const AxisVector& input_order) {
    auto order = std::make_shared<Constant>(element::u64, Shape{input_order.size()}, input_order);
    auto transpose = make_shared<Transpose>(arg, order);
    transpose->set_friendly_name("sinked_transpose");
    NGRAPH_VLOG() << "Make Transpose " << describe<Transpose>(transpose);
    return transpose;
}

static shared_ptr<Reshape> make_reshape(const Output<Node>& arg, const AxisVector& input_order) {
    auto order = std::make_shared<Constant>(element::u64, Shape{input_order.size()}, input_order);
    auto transpose = make_shared<Reshape>(arg, order, false);
    NGRAPH_VLOG() << "Make Reshape " << describe<Reshape>(transpose);
    return transpose;
}

static void write_transposemap(TransposeMap& reorders,
                               const Output<Node>& target,
                               const shared_ptr<Transpose>& transpose) {
    auto name = target.get_node()->get_name() + "." + to_string(target.get_index());
    NGRAPH_VLOG() << "Write TransposeMap[" << name << "] = " << describe<Transpose>(transpose);
    reorders[name] = transpose;
}

static shared_ptr<Transpose> read_transposemap(TransposeMap& reorders, const Output<Node>& target) {
    auto name = target.get_node()->get_name() + "." + to_string(target.get_index());
    auto transpose = reorders[name];
    NGRAPH_VLOG() << "Read TransposeMap[" << name << "]  -> " << describe<Transpose>(transpose);
    return transpose;
}

static shared_ptr<Transpose> combine_transposes(const shared_ptr<Transpose>& t1, const shared_ptr<Transpose>& t2) {
    auto default_order = ngraph::get_default_order(t1->get_shape());
    auto t1_const = as_type_ptr<Constant>(t1->input_value(1).get_node_shared_ptr());
    auto t2_const = as_type_ptr<Constant>(t2->input_value(1).get_node_shared_ptr());

    auto perm_t1 = apply_permutation(default_order, t1_const->get_axis_vector_val());
    auto perm_t2 = apply_permutation(perm_t1, t2_const->get_axis_vector_val());

    auto combined = make_transpose(t2->input_value(0), perm_t2);
    NGRAPH_VLOG() << "Combining " << describe<Transpose>(t1) << " and " << describe<Transpose>(t2) << " into "
                  << describe<Transpose>(combined);
    return combined;
}

static void insert_transpose(const shared_ptr<Node>& target, const shared_ptr<Node>& transpose, size_t input_index) {
    NGRAPH_VLOG() << "Inserting transpose at input " << target->get_name() << " input index " << input_index;
    auto arg = target->input(input_index).get_source_output();
    NGRAPH_VLOG() << "Arg shape: " << arg.get_shape();
    auto new_order = as_type_ptr<Constant>(transpose->input_value(1).get_node_shared_ptr());
    auto new_transpose = make_transpose(arg.get_node_shared_ptr(), new_order->get_axis_vector_val());
    NGRAPH_VLOG() << "Inserting transpose " << describe<Transpose>(new_transpose) << " at input " << target->get_name()
                  << " input index " << input_index;

    target->input(input_index).replace_source_output(new_transpose->output(0));
}

static void delete_transpose(const shared_ptr<Node>& transpose) {
    NGRAPH_VLOG() << "Removing transpose " << transpose->get_name();
    if (!transpose->get_users().empty()) {
        Output<Node> output = transpose->output(0);
        NGRAPH_VLOG() << "output " << output.get_node_shared_ptr()->get_name();
        NGRAPH_VLOG() << "target input size " << output.get_target_inputs().size();
        for (auto input : output.get_target_inputs()) {
            NGRAPH_VLOG() << "input " << input.get_node()->get_name();
            input.replace_source_output(transpose->input_value(0));
        }
    }
}

static void mark_transpose_for_deletion(const shared_ptr<Node>& transpose,
                                        set<shared_ptr<Node>>& transposes_to_delete) {
    NGRAPH_VLOG() << "Marking transpose " << transpose->get_name() << " for deletion";
    transposes_to_delete.insert(transpose);
}

static shared_ptr<Transpose> create_default_transpose(const Output<Node>& n) {
    auto default_order = ngraph::get_default_order(n.get_shape());
    auto order = std::make_shared<Constant>(element::u64, Shape{default_order.size()}, default_order);
    return make_shared<Transpose>(n, order);
}

// convert_binary_to_default_order is used when one of the arguments
// of a binary op isn't in the default format (i.e. nhwc instead of nchw)
// We normalize the "left" argument to match the order of the "right" argument
// by either inserting a transpose or a reshape, depending on the shape of the
// "left" argument.
static void convert_binary_to_default_order(const shared_ptr<Node>& binary,
                                            const Input<Node>& input,
                                            const Output<Node>& right,
                                            TransposeMap& reorders,
                                            set<shared_ptr<Node>>& transposes_to_delete) {
    auto left = input.get_source_output();
    auto right_t = read_transposemap(reorders, right);
    auto right_const = as_type_ptr<Constant>(right_t->input_value(1).get_node_shared_ptr());

    auto perm_to_def = permutation_to_default_order(right_const->get_axis_vector_val());

    // if right input is being implicitly broadcasted, insert a reshape
    // instead of a transpose
    shared_ptr<Node> new_node;
    auto left_shape = left.get_shape();
    if (left_shape.size() < perm_to_def.size()) {
        left_shape.insert(left_shape.begin(), perm_to_def.size() - left_shape.size(), 1);

        auto new_shape = apply_permutation(left_shape, perm_to_def);

        new_node = make_reshape(left, new_shape);
    } else if (left_shape.size() == perm_to_def.size()) {
        new_node = make_transpose(left, perm_to_def);
    } else {
        throw runtime_error("case not supported when converting binary to default order");
    }
    input.replace_source_output(new_node->output(0));

    NGRAPH_VLOG() << "right = " << ngraph::vector_to_string(right.get_shape()) << ", "
                  << right.get_node_shared_ptr()->get_name();
    // this should now insert transpose on right
    mark_transpose_for_deletion(right_t, transposes_to_delete);
    write_transposemap(reorders, binary, right_t);
}

static void materialize_shapes(const shared_ptr<Node>& n,
                               TransposeMap& reorders,
                               set<shared_ptr<Node>>& transposes_to_delete) {
    // For each node, create a default transpose for
    // each of the outputs and store in the map
    for (auto& it : n->outputs()) {
        write_transposemap(reorders, it, create_default_transpose(it));
    }

    for (size_t i = 0; i < n->input_values().size(); i++) {
        // materialize all pending transposes, flush pending transposes
        auto arg = n->input_value(i);
        auto arg_transpose = read_transposemap(reorders, arg);
        NGRAPH_VLOG() << "Materializing " << describe<Transpose>(arg_transpose) << " for "
                      << arg.get_node_shared_ptr()->get_name();
        mark_transpose_for_deletion(arg_transpose, transposes_to_delete);
        auto arg_transpose_order = as_type_ptr<Constant>(arg_transpose->input_value(1).get_node_shared_ptr());
        if (arg_transpose_order->get_axis_vector_val() != ngraph::get_default_order(arg.get_shape())) {
            // Insert if arg needs to be transposed.
            insert_transpose(n, arg_transpose, i);
        }
    }
}

static void sink_transpose(const shared_ptr<Transpose>& transpose,
                           TransposeMap& reorders,
                           set<shared_ptr<Node>>& transposes_to_delete) {
    NGRAPH_VLOG() << "Sinking Transpose :" << describe<Transpose>(transpose);
    auto transpose_in = transpose->input_value(0);
    auto orig_transpose = read_transposemap(reorders, transpose_in);
    // combine both transposes
    auto new_transpose = combine_transposes(orig_transpose, transpose);
    // remove original transpose now it's combined with a new one
    // should be safe to remove an already detached node
    mark_transpose_for_deletion(orig_transpose, transposes_to_delete);
    // replace transpose with combined one
    replace_node(transpose, new_transpose);
    mark_transpose_for_deletion(new_transpose, transposes_to_delete);
    write_transposemap(reorders, new_transpose, new_transpose);
}

static void sink_unary(const shared_ptr<Node>& n,
                       TransposeMap& reorders,
                       set<shared_ptr<Node>>& /* transposes_to_delete */) {
    auto arg_transpose = read_transposemap(reorders, n->input_value(0));
    NGRAPH_VLOG() << "Propagating " << describe<Transpose>(arg_transpose) << " for " << n->get_name();
    write_transposemap(reorders, n, arg_transpose);
}

static void sink_binary(const shared_ptr<Node>& binary,
                        TransposeMap& reorders,
                        set<shared_ptr<Node>>& transposes_to_delete) {
    auto left = binary->input_value(0);
    auto right = binary->input_value(1);
    auto left_t = read_transposemap(reorders, left);
    auto right_t = read_transposemap(reorders, right);
    auto left_const = as_type_ptr<Constant>(left_t->input_value(1).get_node_shared_ptr());
    auto right_const = as_type_ptr<Constant>(right_t->input_value(1).get_node_shared_ptr());

    auto left_order = left_const->get_axis_vector_val();
    auto right_order = right_const->get_axis_vector_val();

    auto left_mismatch = left_order != ngraph::get_default_order(left.get_shape());
    auto right_mismatch = right_order != ngraph::get_default_order(right.get_shape());

    NGRAPH_VLOG() << "Sink binary " << binary->get_name() << " left transpose: " << ngraph::vector_to_string(left_order)
                  << " left default: " << ngraph::vector_to_string(ngraph::get_default_order(left.get_shape()))
                  << " right transpose: " << ngraph::vector_to_string(right_order)
                  << " right default: " << ngraph::vector_to_string(ngraph::get_default_order(right.get_shape()));

    if ((left_order.size() == right_order.size() && left_order == right_order) || (!left_mismatch && !right_mismatch)) {
        // Propagate the reshape which matches the shape of the binary node
        auto new_transpose = (binary->get_output_shape(0) == left.get_shape()) ? left_t : right_t;
        NGRAPH_VLOG() << "Propagating " << describe<Transpose>(new_transpose) << " for " << binary->get_name();
        write_transposemap(reorders, binary, new_transpose);
        // at this point, both transposes will be eventually removed
        mark_transpose_for_deletion(left_t, transposes_to_delete);
        mark_transpose_for_deletion(right_t, transposes_to_delete);
    } else {
        try {
            if (right_mismatch) {
                convert_binary_to_default_order(binary, binary->input(0), right, reorders, transposes_to_delete);
            } else {
                if (left_mismatch) {
                    convert_binary_to_default_order(binary, binary->input(1), left, reorders, transposes_to_delete);
                }
            }
        } catch (const std::exception& ex) {
            throw std::runtime_error("");
        }
    }
}

static void sink_pad(shared_ptr<Pad> n, TransposeMap& reorders, set<shared_ptr<Node>>& /* transposes_to_delete */) {
    auto n_in = n->input_value(0);
    auto arg_transpose = read_transposemap(reorders, n_in);
    describe<Transpose>(arg_transpose);
    auto arg_transpose_order = as_type_ptr<Constant>(arg_transpose->input_value(1).get_node_shared_ptr());
    auto order = arg_transpose_order->get_axis_vector_val();
    // we need the correct input shape to produce the right output shape
    // we are going to create a label of the right input shape,
    // so a new pad will have the right shape
    auto def_order = permutation_to_default_order(order);

    auto input_shape = apply_permutation(arg_transpose->get_shape(), def_order);

    auto dummy_correct_shape = make_shared<ngraph::pattern::op::Label>(arg_transpose->get_element_type(), input_shape);

    auto pad_begin = apply_permutation(n->get_pads_begin(), def_order);
    auto pad_end = apply_permutation(n->get_pads_end(), def_order);

    auto new_begin = make_shared<Constant>(element::i64, Shape{pad_begin.size()}, pad_begin);
    auto new_end = make_shared<Constant>(element::i64, Shape{pad_end.size()}, pad_end);
    auto new_pad = make_shared<Pad>(dummy_correct_shape, new_begin, new_end, n->input_value(3), n->get_pad_mode());
    replace_node(dummy_correct_shape, n->input_value(0).get_node_shared_ptr());
    NGRAPH_VLOG() << "Replacing " << n->get_name() << " with " << new_pad->get_name();
    replace_node(n, new_pad);
    auto new_transpose = make_transpose(new_pad, order);
    NGRAPH_VLOG() << "Propagating " << describe<Transpose>(new_transpose) << " for " << n->get_name();
    write_transposemap(reorders, new_pad, new_transpose);
}

static void sink_concat(const shared_ptr<Concat>& n,
                        TransposeMap& reorders,
                        set<shared_ptr<Node>>& transposes_to_delete) {
    auto n_in = n->input_value(0);
    auto arg_transpose = read_transposemap(reorders, n_in);
    auto arg_transpose_order = as_type_ptr<Constant>(arg_transpose->input_value(1).get_node_shared_ptr());
    auto order = arg_transpose_order->get_axis_vector_val();
    // we need the correct input shape to produce the right output shape
    // we are going to create a label of the right input shape,
    // so a new concat will have the right shape
    auto def_order = permutation_to_default_order(order);

    auto input_shape = apply_permutation(arg_transpose->get_shape(), def_order);

    auto dummy_correct_shape = make_shared<ngraph::pattern::op::Label>(arg_transpose->get_element_type(), input_shape);

    NodeVector new_args;
    new_args.push_back(dummy_correct_shape);

    for (size_t i = 1; i < n->get_input_size(); i++) {
        auto iarg = n->input_value(i);
        auto iarg_transpose = read_transposemap(reorders, iarg);
        auto iarg_transpose_order = as_type_ptr<Constant>(iarg_transpose->input_value(1).get_node_shared_ptr());
        auto iorder = iarg_transpose_order->get_axis_vector_val();
        if (iorder != order) {
            NGRAPH_VLOG() << " input order at " << i << "-th arg is different from first arg";
            materialize_shapes(n, reorders, transposes_to_delete);
            return;
        }

        auto iinput_shape = apply_permutation(iarg_transpose->get_shape(), def_order);

        auto idummy_correct_shape =
            make_shared<ngraph::pattern::op::Label>(iarg_transpose->get_element_type(), iinput_shape);
        new_args.push_back(idummy_correct_shape);
    }

    auto new_axis = order.at(n->get_concatenation_axis());
    auto new_concat = make_shared<Concat>(new_args, new_axis);
    // put back the original arguments
    for (size_t i = 0; i < new_concat->get_input_size(); i++) {
        NGRAPH_VLOG() << "Replacing " << new_concat->get_name() << " input " << i << " with " << n->get_name()
                      << " input " << i;
        new_concat->input(i).replace_source_output(n->input_value(i));
    }
    NGRAPH_VLOG() << "Replacing " << n->get_name() << " with " << new_concat->get_name();
    replace_node(n, new_concat);
    auto new_transpose = make_transpose(new_concat, order);
    NGRAPH_VLOG() << "Propagating " << describe<Transpose>(new_transpose) << " for " << n->get_name();
    write_transposemap(reorders, new_concat, new_transpose);
}

// The goal of TransposeSinking is to remove
// round-trip transposes(i.e. nhwc->nchw(nchw-only-op)->nhwc)
// around nchw-only-op (e.g.Convolution, Batchnorm, Avg/MaxPool)
// This is achieved by both **sinking**, propagating transposes
// through ops towards op::Results,
// or **swimming** Transposes up towards op::Parameter
// For each op type we support we can either combine
// two transposes by replacing the existing Transpose,
// materialize pending transposes if they can't be propagated through op
bool ov::frontend::tf::pass::TransposeSinkingOVTF::run_on_function(shared_ptr<Function> f) {
    TransposeMap reorders;
    set<shared_ptr<Node>> transposes_to_delete;
    unordered_map<std::string, Shape> orig_result_out_shape;

    // STEP 1 : Sink or Swim transposes away for op clusters
    try {
        for (const auto& n : f->get_ordered_ops()) {
            NGRAPH_VLOG() << "Processing " << n->get_name();
            // collect output shape of all Result nodes for a sanity check
            if (ngraph::op::is_output(n)) {
                orig_result_out_shape[n->get_name()] = n->get_output_shape(0);
            }
            if (auto transpose = as_type_ptr<opset8::Transpose>(n)) {
                sink_transpose(transpose, reorders, transposes_to_delete);
            } else if (ngraph::op::is_unary_elementwise_arithmetic(n)) {
                sink_unary(n, reorders, transposes_to_delete);
            } else if (ngraph::op::is_binary_elementwise_arithmetic(n)) {
                sink_binary(n, reorders, transposes_to_delete);
            } else if (auto pad = as_type_ptr<Pad>(n)) {
                sink_pad(pad, reorders, transposes_to_delete);
            } else if (auto concat = as_type_ptr<Concat>(n)) {
                sink_concat(concat, reorders, transposes_to_delete);
            } else {
                materialize_shapes(n, reorders, transposes_to_delete);
            }
        }
    } catch (...) {
        NGRAPH_VLOG() << "Caught exception while sinking op";
        return false;
    }

    // STEP 2: purge all the transposes we either sunk or swam.
    NGRAPH_VLOG() << "Purging transposes ";
    for (auto r : transposes_to_delete) {
        delete_transpose(r);
    }

    // STEP 3: fix wrong shape info wholesale
    NGRAPH_VLOG() << "Fixing wrong shape info for the whole graph";
    for (auto n : f->get_ordered_ops()) {
        n->revalidate_and_infer_types();
    }

    const ResultVector& results = f->get_results();
    for (const auto& r : results) {
        // make sure shapes are always materialized before results
        NGRAPH_CHECK(
            r->get_shape() == r->get_input_shape(0) && r->get_element_type() == r->input_value(0).get_element_type(),
            " op::Result = ",
            *r,
            ", Arg = ",
            r->input_value(0).get_node());

        // make sure that after TransposeSinking pass the output_shape for Result
        // does not change from the expected output_shape before the pass
        NGRAPH_CHECK(r->get_output_shape(0) == orig_result_out_shape[r->get_name()],
                     " op::Result = ",
                     *r,
                     " expected output shape = ",
                     orig_result_out_shape[r->get_name()]);
    }
    return true;
}

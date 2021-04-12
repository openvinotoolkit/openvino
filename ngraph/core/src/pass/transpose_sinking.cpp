//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_set>

#include <ngraph/pattern/op/label.hpp>
#include <ngraph/util.hpp>

#include <ngraph/pass/transpose_sinking.h>

using namespace std;

#include "ngraph/opsets/opset5.hpp"

#if 0
#define NGRAPH_VLOG(LEVEL) std::cerr << "[ DEBUG ][ " << LEVEL << " ]"
#else
#define NGRAPH_VLOG(LEVEL) std::ostringstream()
#endif

namespace ngraph
{
    namespace pass
    {
        namespace opset = ngraph::opset5;
        namespace default_opset = ngraph::opset5;

        using TransposeMap = unordered_map<string, shared_ptr<opset::Transpose>>;

        static ngraph::CoordinateDiff apply_permutation(ngraph::CoordinateDiff input,
                                                        ngraph::AxisVector order)
        {
            ngraph::CoordinateDiff output(input.size());
            for (size_t i = 0; i < order.size(); i++)
            {
                output[i] = input.at(order.at(i));
            }
            return output;
        }

        static ngraph::AxisVector permutation_to_default_order(const ngraph::AxisVector& axis_order)
        {
            ngraph::AxisVector out(axis_order.size());
            for (size_t i = 0; i < axis_order.size(); i++)
            {
                out.at(axis_order[i]) = i;
            }
            return out;
        }

        template <typename T>
        static string describe(shared_ptr<ngraph::Node> node)
        {
            // ensure that it's either a reshape or a transpose
            // TODO: use static_assert
            if (!(std::is_base_of<opset::Reshape, T>::value ||
                  std::is_base_of<opset::Transpose, T>::value))
            {
                throw runtime_error(
                    "describe template specialization has to be either reshape or "
                    "transpose");
            }
            stringstream ss;
            auto transpose = ngraph::as_type_ptr<T>(node);
            auto const1 =
                ngraph::as_type_ptr<opset::Constant>(transpose->get_input_node_shared_ptr(1));
            ss << transpose->get_name()
               << " ( axis order = " << ngraph::vector_to_string(const1->get_axis_vector_val())
               << " , shape = " << ngraph::vector_to_string(transpose->get_shape()) << " ) "
               << " , input = " << transpose->input_value(0).get_node()->get_name();
            return ss.str();
        }

        static shared_ptr<opset::Transpose> make_transpose(ngraph::Output<ngraph::Node> arg,
                                                           const ngraph::AxisVector& input_order)
        {
            auto order = std::make_shared<opset::Constant>(
                ngraph::element::u64, ngraph::Shape{input_order.size()}, input_order);
            auto transpose = make_shared<opset::Transpose>(arg, order);
            NGRAPH_VLOG(4) << "Make Transpose " << describe<opset::Transpose>(transpose);
            return transpose;
        }

        static shared_ptr<opset::Reshape> make_reshape(ngraph::Output<ngraph::Node> arg,
                                                       const ngraph::AxisVector& input_order)
        {
            auto order = std::make_shared<opset::Constant>(
                ngraph::element::u64, ngraph::Shape{input_order.size()}, input_order);
            auto transpose = make_shared<opset::Reshape>(arg, order, false);
            NGRAPH_VLOG(4) << "Make Reshape " << describe<opset::Reshape>(transpose);
            return transpose;
        }

        static void write_transposemap(TransposeMap& reorders,
                                       ngraph::Output<ngraph::Node> target,
                                       shared_ptr<opset::Transpose> transpose)
        {
            auto name = target.get_node()->get_name() + "." + to_string(target.get_index());
            NGRAPH_VLOG(4) << "Write TransposeMap[" << name
                           << "] = " << describe<opset::Transpose>(transpose);
            reorders[name] = transpose;
        }

        static shared_ptr<opset::Transpose> read_transposemap(TransposeMap& reorders,
                                                              ngraph::Output<ngraph::Node> target)
        {
            auto name = target.get_node()->get_name() + "." + to_string(target.get_index());
            auto transpose = reorders[name];
            NGRAPH_VLOG(4) << "Read TransposeMap[" << name << "]  -> "
                           << describe<opset::Transpose>(transpose);
            return transpose;
        }

        static shared_ptr<opset::Transpose> combine_transposes(shared_ptr<opset::Transpose> t1,
                                                               shared_ptr<opset::Transpose> t2)
        {
            auto default_order = ngraph::get_default_order(t1->get_shape());
            auto t1_const =
                ngraph::as_type_ptr<opset::Constant>(t1->input_value(1).get_node_shared_ptr());
            auto t2_const =
                ngraph::as_type_ptr<opset::Constant>(t2->input_value(1).get_node_shared_ptr());
            auto perm_t1 =
                ngraph::apply_permutation(default_order, t1_const->get_axis_vector_val());
            auto perm_t2 = ngraph::apply_permutation(perm_t1, t2_const->get_axis_vector_val());
            auto combined = make_transpose(t2->input_value(0), perm_t2);
            NGRAPH_VLOG(4) << "Combining " << describe<opset::Transpose>(t1) << " and "
                           << describe<opset::Transpose>(t2) << " into "
                           << describe<opset::Transpose>(combined);
            return combined;
        }

        static void insert_transpose(shared_ptr<ngraph::Node> target,
                                     shared_ptr<ngraph::Node> transpose,
                                     size_t input_index)
        {
            NGRAPH_VLOG(4) << "Inserting transpose at input " << target->get_name()
                           << " input index " << input_index;
            auto arg = target->input(input_index).get_source_output();
            NGRAPH_VLOG(4) << "Arg shape: " << arg.get_shape();
            auto new_order = ngraph::as_type_ptr<opset::Constant>(
                transpose->input_value(1).get_node_shared_ptr());
            auto new_transpose =
                make_transpose(arg.get_node_shared_ptr(), new_order->get_axis_vector_val());
            NGRAPH_VLOG(4) << "Inserting transpose " << describe<opset::Transpose>(new_transpose)
                           << " at input " << target->get_name() << " input index " << input_index;

            target->input(input_index).replace_source_output(new_transpose->output(0));
        }

        static void delete_transpose(shared_ptr<ngraph::Node> transpose)
        {
            NGRAPH_VLOG(4) << "Removing transpose " << transpose->get_name();
            if (!transpose->get_users().empty())
            {
                ngraph::Output<ngraph::Node> output = transpose->output(0);
                NGRAPH_VLOG(5) << "output " << output.get_node_shared_ptr()->get_name();
                NGRAPH_VLOG(5) << "target input size " << output.get_target_inputs().size();
                for (auto input : output.get_target_inputs())
                {
                    NGRAPH_VLOG(5) << "input " << input.get_node()->get_name();
                    input.replace_source_output(transpose->input_value(0));
                }
            }
        }

        static void mark_transpose_for_deletion(shared_ptr<ngraph::Node> transpose,
                                                set<shared_ptr<ngraph::Node>>& transposes_to_delete)
        {
            NGRAPH_VLOG(4) << "Marking transpose " << transpose->get_name() << " for deletion";
            transposes_to_delete.insert(transpose);
        }

        static shared_ptr<opset::Transpose> create_default_transpose(ngraph::Output<ngraph::Node> n)
        {
            auto default_order = ngraph::get_default_order(n.get_shape());
            auto order = std::make_shared<opset::Constant>(
                ngraph::element::u64, ngraph::Shape{default_order.size()}, default_order);
            return make_shared<opset::Transpose>(n, order);
        }

        // convert_binary_to_default_order is used when one of the arguments
        // of a binary op isn't in the default format (i.e. nhwc instead of nchw)
        // We normalize the "left" argument to match the order of the "right" argument
        // by either inserting a transpose or a reshape, depending on the shape of the
        // "left" argument.
        static void
            convert_binary_to_default_order(shared_ptr<ngraph::Node> binary,
                                            const ngraph::Input<ngraph::Node>& input,
                                            ngraph::Output<ngraph::Node> right,
                                            TransposeMap& reorders,
                                            set<shared_ptr<ngraph::Node>>& transposes_to_delete)
        {
            auto left = input.get_source_output();
            auto right_t = read_transposemap(reorders, right);
            auto right_const =
                ngraph::as_type_ptr<opset::Constant>(right_t->input_value(1).get_node_shared_ptr());

            auto perm_to_def = permutation_to_default_order(right_const->get_axis_vector_val());

            // if right input is being implicitly broadcasted, insert a reshape
            // instead of a transpose
            shared_ptr<ngraph::Node> new_node;
            auto left_shape = left.get_shape();
            if (left_shape.size() < perm_to_def.size())
            {
                left_shape.insert(left_shape.begin(), perm_to_def.size() - left_shape.size(), 1);
                auto new_shape = ngraph::apply_permutation(left_shape, perm_to_def);
                new_node = make_reshape(left, new_shape);
            }
            else if (left_shape.size() == perm_to_def.size())
            {
                new_node = make_transpose(left, perm_to_def);
            }
            else
            {
                throw runtime_error("case not supported when converting binary to default order");
            }
            input.replace_source_output(new_node->output(0));

            NGRAPH_VLOG(4) << "right = " << ngraph::vector_to_string(right.get_shape()) << ", "
                           << right.get_node_shared_ptr()->get_name();
            // this should now insert transpose on right
            mark_transpose_for_deletion(right_t, transposes_to_delete);
            write_transposemap(reorders, binary, right_t);
        }

        static void materialize_shapes(shared_ptr<ngraph::Node> n,
                                       TransposeMap& reorders,
                                       set<shared_ptr<ngraph::Node>>& transposes_to_delete)
        {
            // For each node, create a default transpose for
            // each of the outputs and store in the map
            for (auto& it : n->outputs())
            {
                write_transposemap(reorders, it, create_default_transpose(it));
            }

            for (size_t i = 0; i < n->input_values().size(); i++)
            {
                // materialize all pending transposes, flush pending transposes
                auto arg = n->input_value(i);
                auto arg_transpose = read_transposemap(reorders, arg);
                NGRAPH_VLOG(4) << "Materializing " << describe<opset::Transpose>(arg_transpose)
                               << " for " << arg.get_node_shared_ptr()->get_name();
                mark_transpose_for_deletion(arg_transpose, transposes_to_delete);
                auto arg_shape = arg.get_shape();
                auto arg_transpose_order = ngraph::as_type_ptr<opset::Constant>(
                    arg_transpose->input_value(1).get_node_shared_ptr());
                if (arg_transpose_order->get_axis_vector_val() !=
                    get_default_order(arg.get_shape()))
                {
                    // Insert if arg needs to be transposed.
                    insert_transpose(n, arg_transpose, i);
                }
            }
        }

        static void sink_transpose(shared_ptr<opset::Transpose> transpose,
                                   TransposeMap& reorders,
                                   set<shared_ptr<ngraph::Node>>& transposes_to_delete)
        {
            NGRAPH_VLOG(4) << "Sinking Transpose :" << describe<opset::Transpose>(transpose);
            auto transpose_in = transpose->input_value(0);
            auto orig_transpose = read_transposemap(reorders, transpose_in);
            // combine both transposes
            auto new_transpose = combine_transposes(orig_transpose, transpose);
            // remove original transpose now it's combined with a new one
            // should be safe to remove an already detached node
            mark_transpose_for_deletion(orig_transpose, transposes_to_delete);
            // replace transpose with combined one
            ngraph::replace_node(transpose, new_transpose);
            mark_transpose_for_deletion(new_transpose, transposes_to_delete);
            write_transposemap(reorders, new_transpose, new_transpose);
        }

        static void sink_unary(shared_ptr<ngraph::Node> n,
                               TransposeMap& reorders,
                               set<shared_ptr<ngraph::Node>>& /* transposes_to_delete */)
        {
            auto arg_transpose = read_transposemap(reorders, n->input_value(0));
            NGRAPH_VLOG(4) << "Propagating " << describe<opset::Transpose>(arg_transpose) << " for "
                           << n->get_name();
            write_transposemap(reorders, n, arg_transpose);
        }

        static void sink_binary(shared_ptr<ngraph::Node> binary,
                                TransposeMap& reorders,
                                set<shared_ptr<ngraph::Node>>& transposes_to_delete)
        {
            auto left = binary->input_value(0);
            auto right = binary->input_value(1);
            auto left_t = read_transposemap(reorders, left);
            auto right_t = read_transposemap(reorders, right);
            auto left_const =
                ngraph::as_type_ptr<opset::Constant>(left_t->input_value(1).get_node_shared_ptr());
            auto right_const =
                ngraph::as_type_ptr<opset::Constant>(right_t->input_value(1).get_node_shared_ptr());

            auto left_order = left_const->get_axis_vector_val();
            auto right_order = right_const->get_axis_vector_val();

            auto left_mismatch = left_order != ngraph::get_default_order(left.get_shape());
            auto right_mismatch = right_order != ngraph::get_default_order(right.get_shape());

            NGRAPH_VLOG(4) << "Sink binary " << binary->get_name()
                           << " left transpose: " << ngraph::vector_to_string(left_order)
                           << " left default: "
                           << ngraph::vector_to_string(ngraph::get_default_order(left.get_shape()))
                           << " right transpose: " << ngraph::vector_to_string(right_order)
                           << " right default: "
                           << ngraph::vector_to_string(
                                  ngraph::get_default_order(right.get_shape()));

            if ((left_order.size() == right_order.size() && left_order == right_order) ||
                (!left_mismatch && !right_mismatch))
            {
                // Propagate the reshape which matches the shape of the binary node
                auto new_transpose =
                    (binary->get_output_shape(0) == left.get_shape()) ? left_t : right_t;
                NGRAPH_VLOG(4) << "Propagating " << describe<opset::Transpose>(new_transpose)
                               << " for " << binary->get_name();
                write_transposemap(reorders, binary, new_transpose);
                // at this point, both transposes will be eventually removed
                mark_transpose_for_deletion(left_t, transposes_to_delete);
                mark_transpose_for_deletion(right_t, transposes_to_delete);
            }
            else
            {
                if (right_mismatch)
                {
                    convert_binary_to_default_order(
                        binary, binary->input(0), right, reorders, transposes_to_delete);
                }
                if (left_mismatch)
                {
                    convert_binary_to_default_order(
                        binary, binary->input(1), left, reorders, transposes_to_delete);
                }
            }
        }

        static void sink_pad(shared_ptr<opset::Pad> n,
                             TransposeMap& reorders,
                             set<shared_ptr<ngraph::Node>>& /* transposes_to_delete */)
        {
            auto n_in = n->input_value(0);
            auto arg_transpose = read_transposemap(reorders, n_in);
            describe<opset::Transpose>(arg_transpose);
            auto arg_transpose_order = ngraph::as_type_ptr<opset::Constant>(
                arg_transpose->input_value(1).get_node_shared_ptr());
            auto order = arg_transpose_order->get_axis_vector_val();
            // we need the correct input shape to produce the right output shape
            // we are going to create a label of the right input shape,
            // so a new pad will have the right shape
            auto def_order = permutation_to_default_order(order);
            auto input_shape = ngraph::apply_permutation(arg_transpose->get_shape(), def_order);
            auto dummy_correct_shape = make_shared<ngraph::pattern::op::Label>(
                arg_transpose->get_element_type(), input_shape);

            auto pad_begin = apply_permutation(n->get_pads_begin(), def_order);
            auto pad_end = apply_permutation(n->get_pads_end(), def_order);
            auto new_begin = make_shared<opset::Constant>(
                ngraph::element::i64, ngraph::Shape{pad_begin.size()}, pad_begin);
            auto new_end = make_shared<opset::Constant>(
                ngraph::element::i64, ngraph::Shape{pad_end.size()}, pad_end);
            auto new_pad = make_shared<opset::Pad>(
                dummy_correct_shape, new_begin, new_end, n->input_value(3), n->get_pad_mode());
            ngraph::replace_node(dummy_correct_shape, n->input_value(0).get_node_shared_ptr());
            NGRAPH_VLOG(4) << "Replacing " << n->get_name() << " with " << new_pad->get_name();
            ngraph::replace_node(n, new_pad);
            auto new_transpose = make_transpose(new_pad, order);
            NGRAPH_VLOG(4) << "Propagating " << describe<opset::Transpose>(new_transpose) << " for "
                           << n->get_name();
            write_transposemap(reorders, new_pad, new_transpose);
        }

        static void sink_concat(shared_ptr<opset::Concat> n,
                                TransposeMap& reorders,
                                set<shared_ptr<ngraph::Node>>& transposes_to_delete)
        {
            auto n_in = n->input_value(0);
            auto arg_transpose = read_transposemap(reorders, n_in);
            auto arg_transpose_order = ngraph::as_type_ptr<opset::Constant>(
                arg_transpose->input_value(1).get_node_shared_ptr());
            auto order = arg_transpose_order->get_axis_vector_val();
            // we need the correct input shape to produce the right output shape
            // we are going to create a label of the right input shape,
            // so a new concat will have the right shape
            auto def_order = permutation_to_default_order(order);
            auto input_shape = ngraph::apply_permutation(arg_transpose->get_shape(), def_order);
            auto dummy_correct_shape = make_shared<ngraph::pattern::op::Label>(
                arg_transpose->get_element_type(), input_shape);

            ngraph::NodeVector new_args;
            new_args.push_back(dummy_correct_shape);

            for (size_t i = 1; i < n->get_input_size(); i++)
            {
                auto iarg = n->input_value(i);
                auto iarg_transpose = read_transposemap(reorders, iarg);
                auto iarg_transpose_order = ngraph::as_type_ptr<opset::Constant>(
                    iarg_transpose->input_value(1).get_node_shared_ptr());
                auto iorder = iarg_transpose_order->get_axis_vector_val();
                if (iorder != order)
                {
                    NGRAPH_VLOG(4)
                        << " input order at " << i << "-th arg is different from first arg";
                    materialize_shapes(n, reorders, transposes_to_delete);
                    return;
                }

                auto iinput_shape =
                    ngraph::apply_permutation(iarg_transpose->get_shape(), def_order);
                auto idummy_correct_shape = make_shared<ngraph::pattern::op::Label>(
                    iarg_transpose->get_element_type(), iinput_shape);
                new_args.push_back(idummy_correct_shape);
            }

            auto new_axis = order.at(n->get_concatenation_axis());
            auto new_concat = make_shared<opset::Concat>(new_args, new_axis);
            // put back the original arguments
            for (size_t i = 0; i < new_concat->get_input_size(); i++)
            {
                NGRAPH_VLOG(4) << "Replacing " << new_concat->get_name() << " input " << i
                               << " with " << n->get_name() << " input " << i;
                new_concat->input(i).replace_source_output(n->input_value(i));
            }
            NGRAPH_VLOG(4) << "Replacing " << n->get_name() << " with " << new_concat->get_name();
            ngraph::replace_node(n, new_concat);
            auto new_transpose = make_transpose(new_concat, order);
            NGRAPH_VLOG(4) << "Propagating " << describe<opset::Transpose>(new_transpose) << " for "
                           << n->get_name();
            write_transposemap(reorders, new_concat, new_transpose);
        }

        // The goal of TransposeSinking is to remove
        // round-trip transposes(i.e. nhwc->nchw(nchw-only-op)->nhwc)
        // around nchw-only-op (e.g.Convolution, Batchnorm, Avg/MaxPool)
        // This is achieved by both **sinking**, propagating transposes
        // through ops towards ngraph::op::Results,
        // or **swimming** Transposes up towards ngraph::op::Parameter
        // For each op type we support we can either combine
        // two transposes by replacing the existing Transpose,
        // materialize pending transposes if they can't be propagated through op
        bool TransposeSinking::run_on_function(shared_ptr<ngraph::Function> f)
        {
            TransposeMap reorders;
            set<shared_ptr<ngraph::Node>> transposes_to_delete;
            unordered_map<std::string, ngraph::Shape> orig_result_out_shape;

            // STEP 1 : Sink or Swim transposes away for op clusters
            for (auto n : f->get_ordered_ops())
            {
                NGRAPH_VLOG(4) << "Processing " << n->get_name();
                // collect output shape of all Result nodes for a sanity check
                if (ngraph::op::is_output(n))
                {
                    orig_result_out_shape[n->get_name()] = n->get_output_shape(0);
                }
                if (auto transpose = ngraph::as_type_ptr<opset::Transpose>(n))
                {
                    sink_transpose(transpose, reorders, transposes_to_delete);
                }
                else if (ngraph::op::is_unary_elementwise_arithmetic(n))
                {
                    sink_unary(n, reorders, transposes_to_delete);
                }
                else if (ngraph::op::is_binary_elementwise_arithmetic(n))
                {
                    sink_binary(n, reorders, transposes_to_delete);
                }
                else if (auto pad = ngraph::as_type_ptr<opset::Pad>(n))
                {
                    sink_pad(pad, reorders, transposes_to_delete);
                }
                else if (auto concat = ngraph::as_type_ptr<opset::Concat>(n))
                {
                    sink_concat(concat, reorders, transposes_to_delete);
                }
                else
                {
                    materialize_shapes(n, reorders, transposes_to_delete);
                }
            }

            // STEP 2: purge all the transposes we either sunk or swam.
            NGRAPH_VLOG(4) << "Purging transposes ";
            for (auto r : transposes_to_delete)
            {
                delete_transpose(r);
            }

            // STEP 3: fix wrong shape info wholesale
            NGRAPH_VLOG(4) << "Fixing wrong shape info for the whole graph";
            for (auto n : f->get_ordered_ops())
            {
                n->revalidate_and_infer_types();
            }

            const ngraph::ResultVector& results = f->get_results();
            for (auto r : results)
            {
                // make sure shapes are always materialized before results
                NGRAPH_CHECK(r->get_shape() == r->get_input_shape(0) &&
                                 r->get_element_type() == r->input_value(0).get_element_type(),
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

    } // namespace pass
} // namespace ngraph

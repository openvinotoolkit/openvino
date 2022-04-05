// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pruning.hpp"
#include "mask_attribute.hpp"

#include <algorithm>
#include <memory>
#include <iterator>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/log.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>

namespace ngraph {
namespace pass {
namespace mask_propagation {

class MatMul;
class Convolution;
class GroupConvolution;
class GroupConvolutionReshape;
class Elementwise;
class PassThrough;
class Reduce;
class Reshape;
class StopPropagation;
class SkipPropagation;
class FakeQuantize;
class Concat;

} // namespace mask_propagation
} // namespace pass
} // namespace ngraph

static ngraph::Shape broadcast_shape_to_rank(ngraph::Shape shape_to_broadcast, int64_t dst_rank) {
    auto initial_rank = static_cast<int64_t>(shape_to_broadcast.size());
    auto num_of_broadcased_dims = dst_rank - initial_rank;
    std::vector<size_t> dims(num_of_broadcased_dims, 1);
    dims.insert(dims.end(), shape_to_broadcast.begin(), shape_to_broadcast.end());
    auto new_shape = ngraph::Shape(dims);
    return new_shape;
}


class ngraph::pass::mask_propagation::MatMul : public MatcherPass {
public:
    MatMul() {
        auto a = pattern::any_input(pattern::has_static_shape());
        auto b = pattern::any_input(pattern::has_static_shape());
        auto matmul = pattern::wrap_type<opset6::MatMul>({a, b});

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_a = pattern_map.at(a);
            const auto & m_b = pattern_map.at(b);
            const auto & m_matmul = pattern_map.at(matmul);

            auto a_mask = getMask(m_a);
            auto b_mask = getMask(m_b);

            if (!a_mask || !b_mask) {
                NGRAPH_DEBUG << "No mask for any input of " << m_matmul.get_node()->get_friendly_name() << "\n";
                return false;
            }
            auto a_mask_row = a_mask.get();
            auto b_mask_row = b_mask.get();

            const auto matmul_op = std::dynamic_pointer_cast<opset6::MatMul>(m_matmul.get_node_shared_ptr());
            const auto transpose_a = matmul_op->get_transpose_a();
            const auto transpose_b = matmul_op->get_transpose_b();

            const auto shape_a = m_a.get_shape();
            const auto shape_b = m_b.get_shape();

            const auto a_inner_dim = (transpose_a)? shape_a.size() - 2 : shape_a.size() - 1;
            const auto a_outer_dim = (transpose_a)? shape_a.size() - 1 : shape_a.size() - 2;
            const auto b_inner_dim = (transpose_b)? shape_b.size() - 1 : shape_b.size() - 2;
            const auto b_outer_dim = (transpose_b)? shape_b.size() - 2 : shape_b.size() - 1;


            const auto matmul_range = m_matmul.get_shape().size();
            auto matmul_mask = std::make_shared<Mask>(matmul_range);
            auto matmul_mask_row = matmul_mask.get();
            const auto matmul_cols_dim = matmul_range - 1;
            const auto matmul_rows_dim = matmul_range - 2;

            const auto matmul_callback = [=](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(matmul_rows_dim) = a_mask_row->at(a_outer_dim);
                cur_mask->at(matmul_cols_dim) = b_mask_row->at(b_outer_dim);
                if (a_mask_row->at(a_inner_dim) != b_mask_row->at(b_inner_dim))
                    cur_mask->initialize_dependencies();
                return true;
            };
            // Connect a with matmul mask
            matmul_mask->add_callback(matmul_callback, a_mask);
            a_mask->add_callback([=](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(a_inner_dim) = b_mask_row->at(b_inner_dim);
                cur_mask->at(a_outer_dim) = matmul_mask_row->at(matmul_rows_dim);
                return true;
            }, matmul_mask);
            // connect b with matmul mask
            matmul_mask->add_callback(matmul_callback, b_mask);
            b_mask->add_callback([=](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(b_inner_dim) = a_mask_row->at(a_inner_dim);
                cur_mask->at(b_outer_dim) = matmul_mask_row->at(matmul_cols_dim);
                return true;
            }, matmul_mask);

            if (!matmul_mask->apply_callback(a_mask)) {
                return false;
            }

            setMask(m_matmul, matmul_mask);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, "MatMulMaskPropagation");
        register_matcher(m, callback);
    }
};


class ngraph::pass::mask_propagation::Convolution : public MatcherPass {
public:
    Convolution() {
        auto input = pattern::any_input();
        auto weights = pattern::any_input(pattern::has_static_shape());
        auto conv = pattern::wrap_type<opset6::Convolution>({input, weights});

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_weights = pattern_map.at(weights);
            const auto & m_output = pattern_map.at(conv);
            const auto & m_input = pattern_map.at(input);

            auto weights_mask = getMask(m_weights);

            // Nullptr in weights-mask means that mask for this node wasn't initialized earlier.
            // Weights mask for convolution should be initialized in the InitMasks pass (and propagate after it).
            // If mask isn't initialized - this weights (and hence all convolution) can't be pruned for some reason.
            if (!weights_mask) {
                NGRAPH_DEBUG << "No weights mask for " << m_output.get_node()->get_friendly_name() << "\n";
                return false;
            }
            auto weights_mask_row = weights_mask.get();

            // Create output mask that describes which channel dimensions will be removed
            auto conv_mask = std::make_shared<Mask>(m_weights.get_shape().size());
            auto conv_mask_row = conv_mask.get();
            auto input_mask = getMask(m_input);
            Mask* input_mask_row = nullptr;
            if (input_mask)
                input_mask_row = input_mask.get();

            const auto conv_mask_callback = [input_mask_row, weights_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1/*input data channel*/) = weights_mask_row->at(0 /* weights output channel dim*/);
                if (input_mask_row && input_mask_row->at(1) != weights_mask_row->at(1))
                    cur_mask->initialize_dependencies();
                return true;
            };

            if (input_mask) {
                // Weights input channel is connected to the convolution input channel dimension
                // so we update weights mask to be aligned with input shape.
                conv_mask->add_callback(conv_mask_callback, input_mask);
                input_mask->add_callback([weights_mask_row](Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(1) = weights_mask_row->at(1);
                    return true;
                }, conv_mask);
            }

            conv_mask->add_callback(conv_mask_callback, weights_mask);
            weights_mask->add_callback([input_mask_row, conv_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(0) = conv_mask_row->at(1);
                if (input_mask_row)
                    cur_mask->at(1) = input_mask_row->at(1);
                return true;
            }, conv_mask);

            bool status;
            if (input_mask)
                status = conv_mask->apply_callback(input_mask);
            else
                status = conv_mask->apply_callback(weights_mask);

            if (!status)
                return false;

            setMask(m_output, conv_mask);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "ConvolutionMaskPropagation");
        register_matcher(m, callback);
    }
};

class ngraph::pass::mask_propagation::GroupConvolution : public MatcherPass {
public:
    GroupConvolution() {
        auto input = pattern::any_input(pattern::has_static_dim(1));
        auto weights = pattern::any_input(pattern::has_static_shape());
        auto group_conv = pattern::wrap_type<opset6::GroupConvolution>({input, weights});

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_weights = pattern_map.at(weights);
            const auto & m_output = pattern_map.at(group_conv);
            const auto & m_input = pattern_map.at(input);

            // TODO: check static rank in pattern, use only particular dims
            auto weights_shape = m_weights.get_shape();
            auto input_shape = m_input.get_partial_shape();
            // support only depthwise convolutions
            if (weights_shape[0] != static_cast<size_t>(input_shape[1].get_length())) {
                return false;
            }

            auto input_mask = getMask(m_input);
            if (!input_mask) return false;
            auto input_mask_row = input_mask.get();

            auto weights_mask = getMask(m_weights);
            if (!weights_mask) {
                // Setting mask only if weights are constant
                if (ngraph::is_type<opset6::Constant>(m_output.get_node_shared_ptr())) {
                    weights_mask = std::make_shared<Mask>(weights_shape.size());
                    setMask(m_weights, weights_mask);
                } else {
                    NGRAPH_DEBUG << "GroupConvolution: No weights mask and weights aren't constant for " <<
                    *m_output.get_node() << "\n";
                    return false;
                }
            }
            auto weights_mask_row = weights_mask.get();

            auto conv_mask = std::make_shared<Mask>(input_shape.rank().get_length());
            auto conv_mask_row = conv_mask.get();

            conv_mask->add_callback([input_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1/*input data channel*/) = input_mask_row->at(1/*output data channel*/);
                return true;
            }, input_mask);

            input_mask->add_callback([conv_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1/*output data channel*/) = conv_mask_row->at(1/*input data channel*/);
                return true;
            }, conv_mask);

            conv_mask->add_callback([weights_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1/*input data channel*/) = weights_mask_row->at(0/*weights output channel dim*/);
                return true;
            }, weights_mask);

            weights_mask->add_callback([conv_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(0/*weights output channel dim*/) = conv_mask_row->at(1/*output data channel*/);
                return true;
            }, conv_mask);

            if (!conv_mask->apply_callback(input_mask)) {
                return false;
            }

            setMask(m_output, conv_mask);

            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(group_conv, "GroupConvolutionMaskPropagation");
        register_matcher(m, callback);
    }
};

class ngraph::pass::mask_propagation::GroupConvolutionReshape : public MatcherPass {
public:
    GroupConvolutionReshape() {
        auto input = pattern::any_input(pattern::has_static_shape());
        auto shape = pattern::any_input();
        // Working only for Reshapes on Group Convolution weights
        auto reshape = pattern::wrap_type<opset6::Reshape>({input, shape}, pattern::consumers_count(1));
        auto gconv = pattern::wrap_type<opset6::GroupConvolution>({pattern::any_input(), reshape},
                                                                  pattern::has_static_shape());

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_shape = pattern_map.at(shape);
            const auto & m_output = pattern_map.at(reshape);
            const auto & m_input = pattern_map.at(input);

            auto shape_val = m_shape.get_node_shared_ptr();

            // In Depthwise Convolutions Reshape on weights just add additional dimension for output channels count
            // (1 in case of the depthwise) of kernel.
            // Example: Reshape from [G, 1 (I), X, Y, Z] -> [G, 1 (O), 1 (I), X, Y, Z], where G - group numbers,
            // X, Y, Z -  spartial dimensions (can be only X or X, Y), I, O - number of input/output channels of kernel.

            // Checking that matched Reshape meets this conditions (add 1-d dim on 1 position of shape constant)
            auto inp_shape = m_input.get_shape();
            auto out_shape = m_output.get_shape();
            inp_shape.insert(inp_shape.begin() + 1, 1);
            if (inp_shape != out_shape || out_shape.size() != 5) {
                return false;
            }

            auto input_mask = getMask(m_input);
            if (!input_mask) {
                return false;
            }

            const auto constant = get_constant_from_source(m_shape.get_node_shared_ptr());
            if (!constant) {
                NGRAPH_DEBUG << "Can't get constant from source node " << m_shape.get_node()->get_friendly_name();
                return false;
            }
            auto input_mask_row = input_mask.get();
            auto output_mask = std::make_shared<Mask>(m_output.get_partial_shape().rank().get_length());

            auto output_mask_row = output_mask.get();

            // Depthwise Convolution pruned only by input channels (== groups) ->
            // Propagating mask from Group (0) dim in Reshape input to Group (0) dim in Reshape output and back
            input_mask->add_callback([output_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(0) = output_mask_row->at(0);
                return true;
            }, output_mask);
            output_mask->add_callback([input_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(0) = input_mask_row->at(0);
                return true;
            }, input_mask);
            output_mask->apply_callback(input_mask);

            setMask(m_output, output_mask);
            // To allow pruning on weights (allow reshape input Group (0) dim changing) modify Reshape Shape input:
            // [G, 1, 1, X, Y, Z] by [-1, 1, 1, X, Y, Z].

            const auto m_shape_consumers = m_shape.get_target_inputs();
            const auto output_shape = constant->get_shape();
            const auto axis = opset6::Constant::create(ov::element::i8, {}, {0});
            auto dims_to_keep_vec = std::vector<size_t>{2, 3, 4};

            const auto dims_to_keep = opset6::Constant::create(m_shape.get_element_type(), {dims_to_keep_vec.size()}, dims_to_keep_vec);
            const auto gather = std::make_shared<opset6::Gather>(m_shape, dims_to_keep, axis);
            const auto concat = std::make_shared<opset6::Concat>(NodeVector{opset6::Constant::create(m_shape.get_element_type(), {2}, {-1, 1}), gather}, 0);
            for (auto consumer : m_shape_consumers)
                consumer.replace_source_output(concat);

            // This transformation propagates only Reshape mask and doesn't do anything with GroupConvolution.
            // So, not to disable GroupConvolution mask propagation we return false here.
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(gconv, "GroupConvolutionReshapeMaskPropagation");
        register_matcher(m, callback);
    }
};


class ngraph::pass::mask_propagation::Elementwise : public MatcherPass {
public:
    Elementwise() {
        auto input = pattern::any_input();
        auto weights = pattern::any_input();
        auto eltwise = pattern::wrap_type<opset6::Add, opset6::Subtract, opset6::Maximum, opset6::Minimum,
        opset6::Multiply>({input, weights}, pattern::has_static_rank());
        // TODO: add Div, Power support

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_weights = pattern_map.at(weights);
            const auto & m_output = pattern_map.at(eltwise);
            const auto & m_input = pattern_map.at(input);

            const auto & input_rank = m_input.get_partial_shape().rank().get_length();
            const auto & weights_rank = m_weights.get_partial_shape().rank().get_length();
            // Here assuming that masks can be propagated only through 3/4 dimensional tensors
            // (since channel dim is necessary) or tensors with equal rank.
            if (!((weights_rank > 2 && input_rank > 2) || weights_rank == input_rank)) return false;
            // Case when input masks should be united instead of intersection
            bool union_eltwise_type = ngraph::is_type<opset6::Multiply>(m_output.get_node_shared_ptr());

            // In case if first of the inputs is constant
            InitConstMask({0, 1/* potential output channel dim */}).apply(m_input.get_node_shared_ptr());
            auto input_mask = getMask(m_input);
            if (!input_mask) {
                NGRAPH_DEBUG << "No input mask for: " << m_output.get_node()->get_friendly_name() << std::endl;
                return false;
            }

            InitConstMask({0, 1}).apply(m_weights.get_node_shared_ptr());
            auto weights_mask = getMask(m_weights);
            if (!weights_mask) {
                NGRAPH_DEBUG << "No weights mask for: " << m_output.get_node()->get_friendly_name() << std::endl;
                return false;
            }
            auto input_mask_row = input_mask.get();
            auto weights_mask_row = weights_mask.get();

            // Merging masks from two inputs
            auto output_mask = std::make_shared<Mask>(m_output.get_partial_shape().rank().get_length());
            auto output_mask_row = output_mask.get();

            auto out_mask_callback = [input_mask_row, weights_mask_row, union_eltwise_type](Mask::Ptr cur_mask) -> bool {
                Mask::Ptr result_mask;
                if (union_eltwise_type) {
                    result_mask = input_mask_row->union_masks_reversed(weights_mask_row);
                } else {
                    result_mask = input_mask_row->intersect_masks_reversed(weights_mask_row);
                }
                cur_mask->copy_value_from_mask_reversed(result_mask.get());
                return true;
            };
            output_mask->add_callback(out_mask_callback, input_mask);

            input_mask->add_callback([weights_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->copy_value_from_mask_reversed(weights_mask_row);
                return true;
            }, weights_mask);
            input_mask->add_callback([output_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->copy_value_from_mask_reversed(output_mask_row);
                return true;
            }, output_mask);
            weights_mask->add_callback([input_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->copy_value_from_mask_reversed(input_mask_row);
                return true;
            }, input_mask);

            output_mask->apply_callback(input_mask);
            weights_mask->apply_callback(input_mask);

            setMask(m_output, output_mask);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, "ElementwiseMaskPropagation");
        register_matcher(m, callback);
    }
};

class ngraph::pass::mask_propagation::FakeQuantize : public MatcherPass{
public:
    FakeQuantize(){
        auto input = pattern::any_input(pattern::has_static_shape());
        auto input_low = pattern::any_input(pattern::has_static_shape());
        auto input_high = pattern::any_input(pattern::has_static_shape());
        auto output_low = pattern::any_input(pattern::has_static_shape());
        auto output_high = pattern::any_input(pattern::has_static_shape());
        auto fake_quantize = pattern::wrap_type<opset6::FakeQuantize>({input, input_low, input_high, output_low,
                                                                            output_high});
        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_input = pattern_map.at(input);
            const auto & m_input_low = pattern_map.at(input_low);
            const auto & m_input_high = pattern_map.at(input_high);
            const auto & m_output_low = pattern_map.at(output_low);
            const auto & m_output_high = pattern_map.at(output_high);
            const auto & m_output = pattern_map.at(fake_quantize);

            auto input_mask = getMask(m_input);

            // Input mask is the only source of pruning in FQ
            if (!input_mask) {
                NGRAPH_DEBUG << "FakeQuantize: No input mask for " << *m_output.get_node() << "\n";
                return false;
            }

            auto input_mask_row = input_mask.get();

            // Propagate input mask to output mask and in the opposite direction
            auto output_mask = std::make_shared<Mask>(m_output.get_partial_shape().rank().get_length());
            auto output_mask_row = output_mask.get();

            // Output mask is equal to input mask
            auto output_mask_callback = [input_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->copy_value_from_mask(input_mask_row);
                return true;
            };

            auto input_mask_callback = [output_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->copy_value_from_mask(output_mask_row);
                return true;
            };

            output_mask->add_callback(output_mask_callback, input_mask);
            input_mask->add_callback(input_mask_callback, output_mask);

            // Calculate output mask
            output_mask->apply_callback(input_mask);
            setMask(m_output, output_mask);

            auto input_low_size = shape_size(m_input_low.get_shape());
            auto input_high_size = shape_size(m_input_high.get_shape());
            auto output_low_size = shape_size(m_output_low.get_shape());
            auto output_high_size = shape_size(m_output_high.get_shape());

            // In the per-tensor case FQ params shouldn't be pruned
            if (input_low_size == 1 && output_low_size == 1 && input_high_size == 1 && output_high_size == 1) {
                return true;
            }

            // If input/output ranges in FQ should be broadcasted to input shape -> broadcast this consant values
            // for the convenience of working with the masks
            NodeVector fq_params_nodes{m_input_low.get_node_shared_ptr(),
                                                               m_input_high.get_node_shared_ptr(),
                                                               m_output_low.get_node_shared_ptr(),
                                                               m_output_high.get_node_shared_ptr()};
            auto fq_node = std::dynamic_pointer_cast<op::FakeQuantize>(m_output.get_node_shared_ptr());
            if (!fq_node) return false;
            size_t idx = 0;
            if (fq_node->get_auto_broadcast() != ngraph::op::AutoBroadcastType::NONE) {
                for (auto node : fq_params_nodes) {
                    auto const_node = std::dynamic_pointer_cast<op::Constant>(node);
                    if (!const_node) throw ngraph_error("Unexpected operation type.");
                    auto new_shape = broadcast_shape_to_rank(const_node->get_shape(),
                                                             m_input.get_partial_shape().rank().get_length());
                    auto new_const = std::make_shared<op::Constant>(*const_node, new_shape);
                    new_const->set_friendly_name(const_node->get_friendly_name());
                    ngraph::copy_runtime_info(const_node, new_const);
                    ngraph::replace_node(const_node, new_const);
                    fq_params_nodes[idx++] = new_const;
                }
            }

            auto fq_params_mask_callback = [input_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1/* fq params have same shapes as input */) = input_mask_row->at(1 /* channel dim in data */);
                return true;
            };

            for (auto fq_param : fq_params_nodes) {
                auto mask = std::make_shared<Mask>(fq_param->get_shape().size());
                mask->add_callback(fq_params_mask_callback, input_mask);
                input_mask->add_callback([mask](Mask::Ptr cur_mask) -> bool {
                    return true;
                }, mask);
                mask->apply_callback(input_mask);
                setMask(fq_param->output(0), mask);
            }

            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(fake_quantize, "FakeQuantizeMaskPropagation");
        register_matcher(m, callback);
    }
};

class ngraph::pass::mask_propagation::Concat : public MatcherPass{
public:
    Concat() {
        auto concat = pattern::wrap_type<opset6::Concat>(pattern::has_static_shape());

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_output = pattern_map.at(concat);
            auto concat_ptr = std::dynamic_pointer_cast<opset6::Concat>(m_output.get_node_shared_ptr());
            if (!concat_ptr) {
                return false;
            }
            auto axis = concat_ptr->get_concatenation_axis();

            auto inputs = concat_ptr->inputs();
            std::map<int64_t , Mask::Ptr> input_masks;
            std::map<int64_t , Mask *> input_masks_row;
            std::vector<int64_t> input_sizes;

            size_t first_input_idx = 0;
            Mask::Ptr first_input_mask;
            bool first_initialized = false;
            for (size_t i=0; i < inputs.size(); i++) {
                auto input = inputs[i];
                auto input_mask = getMask(input.get_source_output());
                if (input_mask) {
                    input_masks[i] = input_mask;
                    input_masks_row[i] = input_mask.get();

                    if (!first_initialized) {
                        first_input_idx = i;
                        first_input_mask = input_mask;
                        first_initialized = true;
                    }
                }
                input_sizes.push_back(input.get_shape().at(axis));
            }

            if (!first_initialized) {
                return false;
            }

            auto output_mask = std::make_shared<Mask>(m_output.get_partial_shape().rank().get_length());
            auto output_mask_row = output_mask.get();

            auto out_mask_callback = [input_masks_row, input_sizes, axis](Mask::Ptr cur_mask) -> bool {
                int64_t cur_size = 0;
                cur_mask->at(axis).clear();

                for (size_t i=0; i < input_sizes.size(); ++i) {
                    if (input_masks_row.count(i)) {
                        for (auto idx : input_masks_row.at(i)->at(axis)) {
                            cur_mask->at(axis).insert(idx + cur_size);
                        }
                    }
                    cur_size += input_sizes[i];
                }
                return true;
            };

            auto create_input_mask_callback_for_idx = [output_mask_row, input_sizes, axis](size_t input_idx){
                auto input_mask_callback = [output_mask_row, input_sizes, axis, input_idx](Mask::Ptr cur_mask) -> bool {
                    cur_mask->clean_dim_values();
                    uint64_t min_val = 0;
                    for (size_t i = 0; i < input_idx; i++) {
                        min_val += input_sizes[i];
                    }
                    uint64_t max_val = min_val + input_sizes[input_idx];
                    for (auto idx : output_mask_row->at(axis)) {
                        if (idx < max_val && idx >= min_val) {
                            cur_mask->at(axis).insert(idx - min_val);
                        }
                    }
                    return true;
                };
                return input_mask_callback;
            };
            output_mask->add_callback(out_mask_callback, first_input_mask);

            for (size_t i=0; i < inputs.size(); ++i) {
                if (input_masks.count(i) && i != first_input_idx) {
                    auto input_mask = input_masks.at(i);
                    input_mask->add_callback(create_input_mask_callback_for_idx(i),
                                             first_input_mask);
                    first_input_mask->add_callback([](Mask::Ptr cur_mask) -> bool {
                        return true;
                    }, input_mask);
                }
            }
            first_input_mask->add_callback(create_input_mask_callback_for_idx(first_input_idx),
                                     output_mask);
            output_mask->apply_callback(first_input_mask);
            setMask(m_output, output_mask);

            return true;
        };
        auto m = std::make_shared<ngraph::pattern::Matcher>(concat, "ConcatMaskPropagation");
        register_matcher(m, callback);
    }
};

class ngraph::pass::mask_propagation::PassThrough : public MatcherPass {
public:
    PassThrough() {
        auto unary_op = pattern::wrap_type<op::util::UnaryElementwiseArithmetic, opset6::Clamp, opset6::Swish,
                                           opset6::Elu, opset6::HardSigmoid, opset6::PRelu, opset6::Mish,
                                           opset6::Softmax, opset6::SoftPlus, opset6::Convert, opset6::ConvertLike,
                                           opset6::AvgPool, opset6::MaxPool, opset6::ROIPooling, opset6::PSROIPooling,
                                           opset6::Pad, opset6::MVN>();


        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_output = pattern_map.at(unary_op);
            const auto & m_input = m_output.get_node_shared_ptr()->input_value(0);

            if (auto input_mask = getMask(m_input)) {
                setMask(m_output, input_mask);
            }

            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(unary_op, "PassThroughMaskPropagation");
        register_matcher(m, callback);
    }
};

class ngraph::pass::mask_propagation::Reduce : public MatcherPass {
public:
    Reduce() {
        auto inputs = pattern::any_input();
        auto weights = pattern::wrap_type<opset6::Constant>();
        auto pooling_by_reduce = pattern::wrap_type<opset6::ReduceMin, opset6::ReduceMax, opset6::ReduceMean>({inputs, weights});

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto m_weights = pattern_map.at(weights);
            const auto & m_input = pattern_map.at(inputs);
            const auto & m_output = pattern_map.at(pooling_by_reduce);

            // Check reduce operation reduces only dimension without masks
            if (auto input_mask = getMask(m_input)) {
                auto output_mask = std::make_shared<Mask>(m_output.get_partial_shape().rank().get_length());
                const auto constant = std::dynamic_pointer_cast<opset6::Constant>(m_weights.get_node_shared_ptr());
                const auto reduce_dims = constant->cast_vector<int64_t>();

                auto input_mask_row = input_mask.get();
                auto output_mask_row = output_mask.get();
                input_mask->add_callback([output_mask_row](Mask::Ptr cur_mask) -> bool {
                    cur_mask->copy_value_from_mask(output_mask_row);
                    return true;
                }, output_mask);
                output_mask->add_callback([input_mask_row, reduce_dims](Mask::Ptr cur_mask) -> bool{
                    // Propagate masks through dimension only if this dimension isn't reduced
                    for (size_t dim = 0; dim < std::min(cur_mask->size(), input_mask_row->size()); ++dim)
                        if (std::find(reduce_dims.begin(), reduce_dims.end(), dim) == reduce_dims.end())
                            cur_mask->at(dim) = input_mask_row->at(dim);
                        else if (cur_mask->at(dim) != input_mask_row->at(dim))
                            cur_mask->initialize_dependencies();
                    return true;
                }, input_mask);

                // Invalidate current mask and its parent masks
                output_mask->apply_callback(input_mask);
                setMask(m_output, output_mask);
            }

            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(pooling_by_reduce, "PassThroughReduceMaskPropagation");
        register_matcher(m, callback);
    }
};


static std::pair<std::set<uint64_t>, bool> squeeze_mask(
    const std::set<uint64_t> mask_dim, const size_t elems_per_ch, const bool squeeze) {
    bool should_init_dep = false;
    auto ret_set = std::set<uint64_t>();
    auto mask_dim_copy = std::set<uint64_t>();
    std::copy(mask_dim.begin(), mask_dim.end(), std::inserter(mask_dim_copy, mask_dim_copy.begin()));
    while (mask_dim_copy.size()) {
        const auto elem = *mask_dim_copy.begin();
        const auto ch = elem / elems_per_ch;
        // Check all channel is zeroed
        const auto low = mask_dim_copy.lower_bound(ch * elems_per_ch);
        const auto upper = mask_dim_copy.lower_bound((ch + 1) * elems_per_ch);
        auto channel_zeros = std::set<uint64_t>();
        std::copy(low, upper, std::inserter(channel_zeros, channel_zeros.begin()));

        // Remove all zeros related to current channel from iter mask
        mask_dim_copy.erase(low, upper);
        // In case any of elements are not zeroed - skip entire channel
        if (channel_zeros.size() != elems_per_ch) {
            should_init_dep = true;
            continue;
        }
        // Add zeros for current channel in current mask
        if (squeeze)
            ret_set.insert(ch);
        else
            ret_set.insert(channel_zeros.begin(), channel_zeros.end());
    }
    return std::make_pair(ret_set, should_init_dep);
}


class ngraph::pass::mask_propagation::Reshape : public MatcherPass {
public:
    Reshape() {
        auto inputs = pattern::any_input(pattern::has_static_shape());
        auto weights = pattern::any_input();
        auto reshape = pattern::wrap_type<opset6::Reshape>({inputs, weights}, pattern::has_static_shape());

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto m_weights = pattern_map.at(weights);
            const auto & m_input = pattern_map.at(inputs);
            const auto & m_output = pattern_map.at(reshape);

            // Check if this reshape is before group convolution
            // In such case this reshape should be processed by GroupConvolutionReshape pass
            for (const auto inp : m_output.get_target_inputs())
                if (is_type<opset6::GroupConvolution>(inp.get_node()))
                    return true;

            auto constant = std::dynamic_pointer_cast<opset6::Constant>(m_weights.get_node_shared_ptr());
            if (!constant) {
                    constant = get_constant_from_source(m_weights.get_node_shared_ptr());
                    if (!constant) {
                        NGRAPH_DEBUG << "Can't process reshape node " << m_output.get_node()->get_friendly_name()
                                     <<" with no constant node " << m_weights.get_node()->get_friendly_name()
                                     << " as shape input.";
                        return false;
                    }
            }

            // Check reshape operation reshape only dimension without masks
            if (auto input_mask = getMask(m_input)) {
                auto output_mask = std::make_shared<Mask>(m_output.get_partial_shape().rank().get_length());
                auto weights_mask = std::make_shared<Mask>(m_output.get_partial_shape().rank().get_length(), true);

                const auto input_shape = m_input.get_shape();
                const auto output_shape = m_output.get_node()->output(0).get_shape();

                // Check dimensions equality from the begining and allow
                // to propagate masks only for dimensions which equal from the begining
                size_t not_reshaped_dims;
                {
                    size_t i = 0;
                    for (; i < std::min(input_shape.size(), output_shape.size()); ++i) {
                        if (input_shape[i] != output_shape[i])
                            break;
                    }
                    not_reshaped_dims = i;
                }

                auto input_mask_row = input_mask.get();
                auto weights_mask_row = weights_mask.get();
                auto output_mask_row = output_mask.get();

                // Case when reshape make flatten last dimension
                if (input_shape.size() > output_shape.size() &&
                    output_shape.size() == not_reshaped_dims + 1) {
                    const size_t elems_per_ch = std::accumulate(input_shape.begin() + not_reshaped_dims + 1,
                                                                input_shape.end(), 1, std::multiplies<size_t>());

                    input_mask->add_callback([weights_mask_row, not_reshaped_dims, elems_per_ch](Mask::Ptr cur_mask) -> bool {
                        for (size_t dim = 0; dim < not_reshaped_dims; ++dim)
                            if (dim < not_reshaped_dims)
                                cur_mask->at(dim) = weights_mask_row->at(dim);

                        bool should_init_dep;
                        std::set<uint64_t> updated_mask;
                        std::tie(updated_mask, should_init_dep) = squeeze_mask(weights_mask_row->at(not_reshaped_dims), elems_per_ch, true);

                        cur_mask->at(not_reshaped_dims) = updated_mask;
                        if (should_init_dep) cur_mask->initialize_dependencies();
                        return true;
                    }, weights_mask);

                    weights_mask->add_callback([input_mask_row, not_reshaped_dims, elems_per_ch](Mask::Ptr cur_mask) -> bool {
                        // Propagate masks down through dimension only if this dimension isn't reshaped
                        for (size_t dim = 0; dim < not_reshaped_dims; ++dim)
                            if (dim < not_reshaped_dims)
                                cur_mask->at(dim) = input_mask_row->at(dim);
                        // Flat the last mask
                        for (auto &ch : input_mask_row->at(not_reshaped_dims))
                            for (auto idx = ch * elems_per_ch; idx < (ch + 1) * elems_per_ch; ++idx)
                                cur_mask->at(not_reshaped_dims).insert(idx);
                        return true;
                    }, input_mask);

                    output_mask->add_callback([weights_mask_row](Mask::Ptr cur_mask) -> bool {
                        cur_mask->copy_value_from_mask(weights_mask_row);
                        return true;
                    }, weights_mask);

                    weights_mask->add_callback([output_mask_row, not_reshaped_dims, elems_per_ch](Mask::Ptr cur_mask) -> bool {
                        // Propagate masks up through dimension only if this dimension isn't reshaped
                        for (size_t dim = 0; dim < not_reshaped_dims; ++dim)
                            if (dim < not_reshaped_dims)
                                cur_mask->at(dim) = output_mask_row->at(dim);
                        // For the last dimension keep only those zeros which completely
                        // covering a channel
                        bool should_init_dep;
                        std::set<uint64_t> updated_mask;
                        std::tie(updated_mask, should_init_dep) = squeeze_mask(output_mask_row->at(not_reshaped_dims), elems_per_ch, false);

                        cur_mask->at(not_reshaped_dims) = updated_mask;
                        if (should_init_dep) cur_mask->initialize_dependencies();
                        return true;
                    }, output_mask);
                } else {
                    input_mask->add_callback([weights_mask_row](Mask::Ptr cur_mask) -> bool {
                        cur_mask->copy_value_from_mask(weights_mask_row);
                        return true;
                    }, weights_mask);
                    weights_mask->add_callback([input_mask_row, not_reshaped_dims](Mask::Ptr cur_mask) -> bool{
                        // Propagate masks down through dimension only if this dimension isn't reshaped
                        for (size_t dim = 0; dim < std::min(cur_mask->size(), input_mask_row->size()); ++dim)
                            if (dim < not_reshaped_dims)
                                cur_mask->at(dim) = input_mask_row->at(dim);
                            else if (cur_mask->at(dim) != input_mask_row->at(dim))
                                cur_mask->initialize_dependencies();
                        return true;
                    }, input_mask);

                    output_mask->add_callback([weights_mask_row](Mask::Ptr cur_mask) -> bool {
                        cur_mask->copy_value_from_mask(weights_mask_row);
                        return true;
                    }, weights_mask);

                    weights_mask->add_callback([output_mask_row, not_reshaped_dims](Mask::Ptr cur_mask) -> bool {
                        // Propagate masks up through dimension only if this dimension isn't reshaped
                        for (size_t dim = 0; dim < std::min(cur_mask->size(), output_mask_row->size()); ++dim)
                            if (dim < not_reshaped_dims)
                                cur_mask->at(dim) = output_mask_row->at(dim);
                            else if (cur_mask->at(dim) != output_mask_row->at(dim))
                                cur_mask->initialize_dependencies();
                        return true;
                    }, output_mask);
                }

                weights_mask->apply_callback(input_mask);
                setMask(m_output, output_mask);
                setMask(m_weights, weights_mask);
            }

            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(reshape, "ReshapeMaskPropagation");
        register_matcher(m, callback);
    }
};

class ngraph::pass::mask_propagation::StopPropagation : public MatcherPass {
public:
    StopPropagation() {
        auto any_node = pattern::any_input();

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_output = pattern_map.at(any_node);
            const auto & node = m.get_match_root();

            auto output_mask = std::make_shared<Mask>(m_output.get_partial_shape().rank().get_length());
            auto output_mask_row = output_mask.get();
            bool any_input_with_masks = false;
            for (const auto & input : node->input_values()) {
                if (auto input_mask = getMask(input)) {
                        auto input_mask_row = input_mask.get();
                        input_mask->add_callback([output_mask_row](Mask::Ptr cur_mask) -> bool {
                            cur_mask->clean_dim_values();
                            if (!output_mask_row->all_dims_are_empty())
                                cur_mask->initialize_dependencies();
                            return true;
                        }, output_mask);
                        output_mask->add_callback([input_mask_row](Mask::Ptr cur_mask) -> bool{
                            cur_mask->copy_value_from_mask(input_mask_row);
                            return true;
                        }, input_mask);

                        // Invalidate current mask and its parent masks
                        output_mask->apply_callback(input_mask);
                        NGRAPH_DEBUG << "Invalidate masks for " << *input.get_node() << " because " << node << " is in scope of stop ops.\n";
                        any_input_with_masks = true;
                    }
                }
            if (any_input_with_masks) {
                // Set mask to stop op first input tensor to prevent mask rewriting for
                // nodes which share output tensor with previous node.
                if (ngraph::is_type<opset6::Result>(m_output.get_node_shared_ptr()))
                    setMask(*m_output.get_node()->inputs().begin(), output_mask);
                else
                    setMask(m_output, output_mask);
            }
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(any_node, "StopMaskPropagation");
        register_matcher(m, callback);
    }
};

class ngraph::pass::mask_propagation::SkipPropagation : public MatcherPass {
public:
    SkipPropagation() {
        // Skip mask propagation for ShapeOf operation to prevent this opearation to be
        // processed as stop op.
        auto node = pattern::wrap_type<opset6::ShapeOf>();
        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(node, "SkipPropagation");
        register_matcher(m, callback);
    }
};

ngraph::pass::PropagateMasks::PropagateMasks() {
    add_matcher<mask_propagation::MatMul>();
    add_matcher<mask_propagation::Convolution>();
    add_matcher<mask_propagation::GroupConvolutionReshape>();
    add_matcher<mask_propagation::GroupConvolution>();
    add_matcher<mask_propagation::Elementwise>();
    add_matcher<mask_propagation::PassThrough>();
    add_matcher<mask_propagation::Reduce>();
    add_matcher<mask_propagation::Reshape>();
    add_matcher<mask_propagation::FakeQuantize>();
    add_matcher<mask_propagation::Concat>();
    add_matcher<mask_propagation::SkipPropagation>();
    add_matcher<mask_propagation::StopPropagation>();
}

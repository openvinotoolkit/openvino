// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pruning.hpp"
#include "mask_attribute.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/log.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::PropagateMasks, "PropagateMasks", 0);

namespace ngraph {
namespace pass {
namespace mask_propagation {

class Convolution;
class GroupConvolution;
class Elementwise;
class PassThrough;
class StopPropagation;
class FakeQuantize;
class Concat;

} // namespace mask_propagation
} // namespace pass
} // namespace ngraph

ngraph::Shape broadcast_shape_to_rank(ngraph::Shape shape_to_broadcast, int64_t dst_rank) {
    auto  initial_rank = static_cast<int64_t>(shape_to_broadcast.size());
    std::vector<size_t> dims(dst_rank);
    for (int64_t i = 0; i < dst_rank; i++) {
        auto dsti =
                i < (dst_rank - initial_rank) ? 1 : shape_to_broadcast[i - (dst_rank - initial_rank)];
        dims[i] = dsti;
    }
    auto new_shape = ngraph::Shape(dims);
    return new_shape;
}

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

            // If weights mask still not initialized -> convolution can't be pruned
            if (!weights_mask) {
                NGRAPH_DEBUG << "CONV: No weights mask for " << *m_output.get_node() << "\n";
                return false;
            }
            auto weights_mask_row = weights_mask.get();

            if (auto input_mask = getMask(m_input)) {
                auto input_mask_row = input_mask.get();
                // Weights input channel is connected to the convolution input channel dimension
                // so we update weights mask to be aligned with input shape.
                weights_mask->add_callback([input_mask_row](Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(1/* weights input channel */) = input_mask_row->at(1 /* input data channel */);
                    return true;
                }, input_mask);

                input_mask->add_callback([weights_mask_row](Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(1) = weights_mask_row->at(1);
                    return true;
                }, weights_mask);

                if (!weights_mask->apply_callback(input_mask)) {
                    return false;
                }
            }

            // Create output mask that describes which channel dimensions will be removed
            auto conv_mask = std::make_shared<Mask>(m_weights.get_shape().size());
            auto conv_mask_row = conv_mask.get();

            conv_mask->add_callback([weights_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1) = weights_mask_row->at(0/*weights output channel dim */);
                return true;
            }, weights_mask);

            weights_mask->add_callback([conv_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(0) = conv_mask_row->at(1);
                return true;
            }, conv_mask);

            if (!conv_mask->apply_callback(weights_mask)) {
                return false;
            }

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

            // Weights input channel is connected to the convolution input channel dimension
            // so we update weights mask to be aligned with input shape.
            weights_mask->add_callback([input_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(0) = input_mask_row->at(1);
                return true;
            }, input_mask);

            input_mask->add_callback([weights_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1) = weights_mask_row->at(0);
                return true;
            }, weights_mask);

            if (!weights_mask->apply_callback(input_mask)) {
                return false;
            }

            // Update output channels mask dims
            auto conv_mask = std::make_shared<Mask>(input_shape.rank().get_length());
            auto conv_mask_row = conv_mask.get();

            conv_mask->add_callback([weights_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1) = weights_mask_row->at(0);
                return true;
            }, weights_mask);

            weights_mask->add_callback([conv_mask_row](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(0) = conv_mask_row->at(1);
                return true;
            }, conv_mask);

            if (!conv_mask->apply_callback(weights_mask)) {
                return false;
            }

            setMask(m_output, conv_mask);

            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(group_conv, "GroupConvolutionMaskPropagation");
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

            // Case when input masks should be united instead of intersection
            bool union_eltwise_type = ngraph::is_type<opset6::Multiply>(m_output.get_node_shared_ptr());

            const auto & input_rank = m_input.get_partial_shape().rank().get_length();
            const auto & weights_rank = m_weights.get_partial_shape().rank().get_length();
            // Here assuming that masks can be propagated only through 3/4 dimensional tensors
            // (since channel dim is necessary)
            if (weights_rank < 3 || input_rank < 3) return false;

            // In case if first of the inputs is constant
            InitConstMask({0, 1/* potential output channel dim */}).apply(m_input.get_node_shared_ptr());
            auto input_mask = getMask(m_input);
            if (!input_mask) {
                NGRAPH_DEBUG << "No input mask for: " << m_output.get_node()->get_friendly_name() << std::endl;
                return false;
            }

            if (input_rank == weights_rank) {
                if (union_eltwise_type) {
                    InitConstMask({0, 1}).apply(m_weights.get_node_shared_ptr());
                } else {
                    // Using non-empty dims in input mask for initializing weights mask since masks will be intersect
                    auto mask_dims = input_mask->get_not_empty_dims();
                    InitConstMask(AxisSet(mask_dims)).apply(m_weights.get_node_shared_ptr());
                }
            } else {
                InitConstMask({0}).apply(m_weights.get_node_shared_ptr());
            }

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
        auto input = pattern::any_input();
        auto input_low = pattern::any_input();
        auto input_high = pattern::any_input();
        auto output_low = pattern::any_input();
        auto output_high = pattern::any_input();
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
            std::vector<std::shared_ptr<Node>> fq_params_nodes{m_input_low.get_node_shared_ptr(),
                                                               m_input_high.get_node_shared_ptr(),
                                                               m_output_low.get_node_shared_ptr(),
                                                               m_output_high.get_node_shared_ptr()};
            auto fq_node = std::dynamic_pointer_cast<op::FakeQuantize>(m_output.get_node_shared_ptr());
            size_t idx = 0;
            if (fq_node->get_auto_broadcast() != ngraph::op::AutoBroadcastType::NONE) {
                for (auto const_node : fq_params_nodes) {
                    auto new_shape = broadcast_shape_to_rank(const_node->get_shape(),
                                                             m_input.get_partial_shape().rank().get_length());
                    auto const_copy = const_node->clone_with_new_inputs(const_node->input_values());
                    auto new_const = std::dynamic_pointer_cast<op::Constant>(const_copy);
                    new_const->set_data_shape(new_shape);
                    new_const->validate_and_infer_types();
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
        auto concat = pattern::wrap_type<opset6::Concat>();

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_output = pattern_map.at(concat);
            auto concat_ptr = std::dynamic_pointer_cast<opset6::Concat>(m_output.get_node_shared_ptr());
            auto inputs = concat_ptr->inputs();
            auto axis = concat_ptr->get_concatenation_axis();

            auto output_mask = std::make_shared<Mask>(m_output.get_partial_shape().rank().get_length());
            auto output_mask_row = output_mask.get();

            auto out_mask_callback = [inputs, axis](Mask::Ptr cur_mask) -> bool {
                int64_t cur_size = 0;

                cur_mask->at(axis).clear();

                for (auto input : inputs) {
                    auto output_port = input.get_source_output();
                    auto input_mask = getMask(output_port);

                    for (auto idx : input_mask->at(axis)) {
                        cur_mask->at(axis).insert(idx + cur_size);
                    }

                    cur_size += output_port.get_shape().at(axis);
                }
                return true;
            };

            auto create_input_mask_callback_for_idx = [output_mask_row, inputs, axis](size_t input_idx){
                auto input_mask_callback = [output_mask_row, inputs, axis, input_idx](Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(axis).clear();
                    uint64_t min_val = 0;
                    for (size_t i = 0; i < input_idx; i++) {
                        auto output_port = inputs[i].get_source_output();
                        min_val += output_port.get_shape().at(axis);
                    }
                    uint64_t max_val = min_val + inputs[input_idx].get_shape().at(axis);
                    for (auto idx : output_mask_row->at(axis)) {
                        if (idx < max_val && idx >= min_val) {
                            cur_mask->at(axis).insert(idx);
                        }
                    }
                    return true;
                };
                return input_mask_callback;
            };

            size_t input_idx = 0;

            for (auto input : inputs) {
                auto input_mask = getMask(input.get_source_output());
                if (input_mask) {
                    output_mask->add_callback(out_mask_callback, input_mask);
                    input_mask->add_callback(create_input_mask_callback_for_idx(input_idx),
                                             output_mask);
                }
                input_idx++;
            }

            for (auto input : inputs) {
                auto input_mask = getMask(input.get_source_output());
                if (input_mask) {
                    output_mask->apply_callback(input_mask);
                }
                break;
            }
            setMask(m_output, output_mask);
            std::cout << m_output.get_node_shared_ptr()->get_friendly_name();

            return true;
        };
        auto m = std::make_shared<ngraph::pattern::Matcher>(concat, "ConcatMaskPropagation");
        register_matcher(m, callback);
    }
};

class ngraph::pass::mask_propagation::PassThrough : public MatcherPass {
public:
    PassThrough() {
        auto unary_op = pattern::wrap_type<op::util::UnaryElementwiseArithmetic, opset6::Clamp,
                                            opset6::Convert, opset5::AvgPool, opset6::MaxPool, op::v0::ROIPooling,
                                            opset6::PSROIPooling, opset5::Pad>();

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

class ngraph::pass::mask_propagation::StopPropagation : public MatcherPass {
public:
    StopPropagation() {
        auto any_node = pattern::any_input();

        ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
            const auto & node = m.get_match_root();
            for (const auto & input : node->input_values()) {
                if (auto mask = getMask(input)) {
                    // Invalidate current mask and its parent masks
                    mask->invalidate();
                    NGRAPH_DEBUG << "Invalidate masks for " << *input.get_node() << " because " << node << " is unknown\n";
                }
            }
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(any_node, "StopMaskPropagation");
        register_matcher(m, callback);
    }
};

ngraph::pass::PropagateMasks::PropagateMasks() {
    add_matcher<mask_propagation::Convolution>();
    add_matcher<mask_propagation::GroupConvolution>();
    add_matcher<mask_propagation::Elementwise>();
    add_matcher<mask_propagation::PassThrough>();
    add_matcher<mask_propagation::FakeQuantize>();
    add_matcher<mask_propagation::Concat>();
    add_matcher<mask_propagation::StopPropagation>();
}

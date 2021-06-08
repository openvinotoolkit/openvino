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
class Reshape;

} // namespace mask_propagation
} // namespace pass
} // namespace ngraph

ngraph::Shape broadcast_shape_to_rank(ngraph::Shape shape_to_broadcast, int64_t dst_rank) {
    auto initial_rank = static_cast<int64_t>(shape_to_broadcast.size());
    auto num_of_broadcased_dims = dst_rank - initial_rank;
    std::vector<size_t> dims(num_of_broadcased_dims, 1);
    dims.insert(dims.end(), shape_to_broadcast.begin(), shape_to_broadcast.end());
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

            // Nullptr in weights-mask means that mask for this node wasn't initialized earlier.
            // Weights mask for convolution should be initialized in the InitMasks pass (and propagate after it).
            // If mask isn't initialized - this weights (and hence all convolution) can't be pruned for some reason.
            if (!weights_mask) {
                NGRAPH_DEBUG << "No weights mask for " << m_output.get_node()->get_friendly_name() << "\n";
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

class ngraph::pass::mask_propagation::Reshape : public MatcherPass {
public:
    Reshape() {
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
            if (inp_shape != out_shape) {
                return false;
            }

            auto input_mask = getMask(m_input);
            if (!input_mask) {
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
            input_mask->apply_callback(output_mask);

            // To allow pruning on weights (allow reshape input Group (0) dim changing) replace Reshape Shape constant
            // [G, 1, 1, X, Y, Z] by [-1, 1, 1, X, Y, Z].
            auto old_shape_const = std::dynamic_pointer_cast<opset6::Constant>(m_shape.get_node_shared_ptr());
            auto shape_value = old_shape_const.get()->cast_vector<int64_t>();
            shape_value[0] = -1;
            auto new_const = opset6::Constant::create(old_shape_const->get_element_type(),
                                                      old_shape_const->get_shape(), shape_value);
            new_const->set_friendly_name(old_shape_const->get_friendly_name());
            ngraph::copy_runtime_info(old_shape_const, new_const);
            ngraph::replace_node(old_shape_const, new_const);

            setMask(m_output, output_mask);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(reshape, "ReshapeMaskPropagation");
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
        auto concat = pattern::wrap_type<opset6::Concat>(pattern::has_static_shape());

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_output = pattern_map.at(concat);
            auto concat_ptr = std::dynamic_pointer_cast<opset6::Concat>(m_output.get_node_shared_ptr());
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
        auto unary_op = pattern::wrap_type<op::util::UnaryElementwiseArithmetic, opset6::Clamp,
                                            opset6::Convert, opset6::ConvertLike, opset6::AvgPool, opset6::MaxPool,
                                            opset6::ROIPooling, opset6::PSROIPooling, opset6::Pad>();

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
    add_matcher<mask_propagation::Reshape>();
    add_matcher<mask_propagation::StopPropagation>();
}

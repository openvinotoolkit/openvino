// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pruning.hpp"
#include "mask_attribute.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/log.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::PropagateMasks, "PropagateMasks", 0);

namespace ngraph {
namespace pass {
namespace mask_propagation {

class Convolution;
class GroupConvolution;
class Elementwise;
class PassThrough;
class StopPropagation;

} // namespace mask_propagation
} // namespace pass
} // namespace ngraph

class ngraph::pass::mask_propagation::Convolution : public MatcherPass {
public:
    Convolution() {
        auto input = pattern::any_input();
        auto weights = pattern::any_input();
        auto conv = pattern::wrap_type<opset6::Convolution>({input, weights});

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_weights = pattern_map.at(weights);
            const auto & m_output = pattern_map.at(conv);
            const auto & m_input = pattern_map.at(input);

            // In case if weights are Constant we initialize Mask
            InitConstMask({0}/* check only output channel */).apply(m_weights.get_node_shared_ptr());

            auto weights_mask = getMask(m_weights);
            // If weights are not a Constant and we didn't set Mask value before we will get nullptr
            if (!weights_mask) return false;

            if (auto input_mask = getMask(m_input)) {
                // Weights input channel is connected to the convolution input channel dimension
                // so we update weights mask to be aligned with input shape.
                weights_mask->add_callback([input_mask](Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(1/* weights input channel */) = input_mask->at(1 /* input data channel */);
                    return true;
                }, input_mask);

                input_mask->add_callback([weights_mask](Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(1) = weights_mask->at(1);
                    return true;
                }, weights_mask);

                if (!weights_mask->apply_callback(input_mask)) {
                    return false;
                }
            }

            // Create output mask that describes which channel dimensions will be removed
            auto conv_mask = std::make_shared<Mask>(m_weights.get_shape().size());

            conv_mask->add_callback([weights_mask](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1) = weights_mask->at(0/*weights output channel dim */);
                return true;
            }, weights_mask);

            weights_mask->add_callback([conv_mask](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(0) = conv_mask->at(1);
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
        auto input = pattern::any_input();
        auto weights = pattern::any_input();
        auto group_conv = pattern::wrap_type<opset6::GroupConvolution>({input, weights});

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_weights = pattern_map.at(weights);
            const auto & m_output = pattern_map.at(group_conv);
            const auto & m_input = pattern_map.at(input);

            // TODO: check static rank in pattern, use only particular dims
            auto weights_shape = m_weights.get_shape();
            auto input_shape = m_input.get_shape();
            // support only depthwise convolutions
            if (weights_shape[0] != input_shape[1]) {
                return false;
            }

            auto input_mask = getMask(m_input);
            if (!input_mask) return false;

            auto weights_mask = getMask(m_weights);
            if (!weights_mask) {
                // TODO: only if weights are constant
                weights_mask = std::make_shared<Mask>(weights_shape.size());
                setMask(m_weights, weights_mask);
            }

            // Weights input channel is connected to the convolution input channel dimension
            // so we update weights mask to be aligned with input shape.
            weights_mask->add_callback([input_mask](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(0) = input_mask->at(1);
                return true;
            }, input_mask);

            input_mask->add_callback([weights_mask](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1) = weights_mask->at(0);
                return true;
            }, weights_mask);

            if (!weights_mask->apply_callback(input_mask)) {
                return false;
            }

            // Update output channels mask dims
            auto conv_mask = std::make_shared<Mask>(input_shape.size());

            conv_mask->add_callback([weights_mask](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1) = weights_mask->at(0);
                return true;
            }, weights_mask);

            weights_mask->add_callback([conv_mask](Mask::Ptr cur_mask) -> bool {
                cur_mask->at(0) = conv_mask->at(1);
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
        auto eltwise = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({input, weights},
                                                                                 pattern::has_static_rank());

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_weights = pattern_map.at(weights);
            const auto & m_output = pattern_map.at(eltwise);
            const auto & m_input = pattern_map.at(input);

            // TODO: implement check that compares input shape ranks
            const auto & input_rank = m_input.get_partial_shape().rank().get_length();
            const auto & weights_rank = m_weights.get_partial_shape().rank().get_length();
            if (weights_rank < 3 || input_rank < 3) return false;

            // In case if one of the inputs is constant
            // TODO: need to find channel dimension instead of hardcoded zero
            const size_t & channel_dim = (input_rank == weights_rank ? 1 : 0);
            InitConstMask({channel_dim}).apply(m_input.get_node_shared_ptr());
            InitConstMask({channel_dim}).apply(m_weights.get_node_shared_ptr());

            auto weights_mask = getMask(m_weights);
            auto input_mask = getMask(m_input);

            if (!weights_mask || !input_mask) {
                NGRAPH_DEBUG << "No mask for: " << m_output.get_node()->get_friendly_name() << std::endl;
                return false;
            }

            // Merge masks from two inputs
            auto output_mask = std::make_shared<Mask>(m_output.get_partial_shape().rank().get_length());

            auto out_mask_callback = [input_mask, weights_mask](Mask::Ptr cur_mask) -> bool {
                auto omask_iter = cur_mask->rbegin();
                auto imask_iter = input_mask->rbegin();
                auto wmask_iter = weights_mask->rbegin();

                for (auto & item : *cur_mask) {
                    item.clear();
                }

                while (imask_iter != input_mask->rend() &&
                       wmask_iter != weights_mask->rend()) {
                    // Merge mask dimension values for both masks
                    // Example: (MaskValue[1,2,3,4], MaskValue[2,3]) -> MaskValue[2,3]
                    for (const auto & value : *imask_iter) {
                        if (wmask_iter->count(value)) {
                            omask_iter->insert(value);
                        }
                    }

                    omask_iter++;
                    imask_iter++;
                    wmask_iter++;
                }
                return true;
            };
            output_mask->add_callback(out_mask_callback, input_mask);
            output_mask->add_callback(out_mask_callback, weights_mask);

            auto callback = [output_mask](Mask::Ptr cur_mask) -> bool {
                auto omask_iter = output_mask->rbegin();
                auto cmask_iter = cur_mask->rbegin();
                while (omask_iter != output_mask->rend() &&
                       cmask_iter != cur_mask->rend()) {
                    // TODO: check
                    *cmask_iter = *omask_iter;

                    omask_iter++;
                    cmask_iter++;
                }
                return true;
            };
            input_mask->add_callback(callback, output_mask);
            weights_mask->add_callback(callback, output_mask);

            // Init output mask
            output_mask->apply_callback(input_mask);
            setMask(m_output, output_mask);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, "EltwiseMaskPropagation");
        register_matcher(m, callback);
    }
};

class ngraph::pass::mask_propagation::PassThrough : public MatcherPass {
public:
    PassThrough() {
        auto unary_op = pattern::wrap_type<op::util::UnaryElementwiseArithmetic, opset6::Clamp>();

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
    add_matcher<mask_propagation::StopPropagation>();
}

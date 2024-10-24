// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/static_scaling.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

float get_scale_factor(float scale_factor) {
    const float default_scale_factor = 256.f;

    // scale_factor = (scale_factor < 1) ? default_scale_factor : scale_factor;

    return default_scale_factor;
}

ov::pass::StaticScalingModel::StaticScalingModel(float scale_factor) {
    m_scale_factor = get_scale_factor(scale_factor);
}

bool ov::pass::StaticScalingModel::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(StaticScalingModel);

    std::unordered_set<std::string> scaled_down_subgraph;
    std::unordered_set<std::string> normalized_subgraph;
    std::unordered_set<std::string> constant_subgraph;

    ov::Shape scale_const_shape = {1};
    std::vector<float> scale_value = {m_scale_factor};
    std::vector<float> inverse_scale_value = {(1.f / m_scale_factor)};
    std::shared_ptr<ov::Node> scale_const_f16 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_const_shape, scale_value);
    std::shared_ptr<ov::Node> scale_const_f32 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, scale_const_shape, scale_value);
    std::shared_ptr<ov::Node> inverse_scale_const_f16 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_const_shape, inverse_scale_value);
    std::shared_ptr<ov::Node> inverse_scale_const_f32 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, scale_const_shape, inverse_scale_value);

    for (auto& node : f->get_ordered_ops()) {
        if (node->get_friendly_name().compare("__module.transformer_blocks.0.norm1_context.linear/ov_ext::linear/MatMul") == 0)
            std::cout << "!" << std::endl;

        auto parameter_node = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node);
        if (parameter_node &&
            (parameter_node->get_element_type() == ov::element::f16 ||
             parameter_node->get_element_type() == ov::element::f32)) {
            std::shared_ptr<ov::Node> inverse_scale_const = (parameter_node->get_element_type() == ov::element::f16) ?
                                                             inverse_scale_const_f16 : inverse_scale_const_f32;
            auto scale_down = std::make_shared<ov::op::v1::Multiply>(parameter_node->output(0),
                                                                     inverse_scale_const->output(0));
            ov::replace_node(parameter_node, scale_down);
            scaled_down_subgraph.insert(node->get_friendly_name());
            scaled_down_subgraph.insert(scale_down->get_friendly_name());
            continue;
        }

        auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
        if (const_node) {
            constant_subgraph.insert(node->get_friendly_name());
            continue;
        }

        auto group_norm_node = std::dynamic_pointer_cast<ov::op::v12::GroupNormalization>(node);
        auto mvn_node = std::dynamic_pointer_cast<ov::op::v6::MVN>(node);
        if (group_norm_node || mvn_node) {
            normalized_subgraph.insert(node->get_friendly_name());
            continue;
        }

        size_t num_scaled_down_inputs = 0;
        size_t num_const_inputs = 0;
        size_t num_normalized_inputs = 0;
        for (auto& dep: node->inputs()) {
            auto dep_name = dep.get_source_output().get_node_shared_ptr()->get_friendly_name();

            if (scaled_down_subgraph.find(dep_name) != scaled_down_subgraph.end()) {
                num_scaled_down_inputs += 1;
                continue;
            }
            if (constant_subgraph.find(dep_name) != constant_subgraph.end()) {
                num_const_inputs += 1;
                continue;
            }
            if (normalized_subgraph.find(dep_name) != normalized_subgraph.end()) {
                num_normalized_inputs += 1;
                continue;
            }
        }

        if (node->get_input_size() > 0) {
            if (num_const_inputs == node->get_input_size()) {
                constant_subgraph.insert(node->get_friendly_name());
                continue;
            }
            if ((num_const_inputs + num_normalized_inputs) == node->get_input_size()) {
                normalized_subgraph.insert(node->get_friendly_name());
                continue;
            }
        }

        if (num_scaled_down_inputs == 0) {
            continue;
        }

        //    input0         input1            input0         input1
        // (scaled_down)  (normalized       (scaled_down)  (normalized
        //                 or const)                        or const)
        //          \         /                      \         /
        //           \       /         ==>            \    scale_down
        //            \     /                          \     /
        //              add                              add
        auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(node);
        if (add && num_scaled_down_inputs == 1) {
            for (auto& dep: node->inputs()) {
                if (scaled_down_subgraph.find(dep.get_source_output().get_node_shared_ptr()->get_friendly_name()) == scaled_down_subgraph.end()) {
                    std::shared_ptr<ov::Node> inverse_scale_const = (dep.get_element_type() == ov::element::f16) ?
                                                                    inverse_scale_const_f16 : inverse_scale_const_f32;
                    auto scale_down = std::make_shared<ov::op::v1::Multiply>(dep.get_source_output(),
                                                                             inverse_scale_const->output(0));
                    dep.replace_source_output(scale_down->output(0));
                    std::cout << "scale_down " << scale_down->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name() << " --> "
                            << node->get_friendly_name() << std::endl;
                }
            }
        }

        auto sdpa = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(node);
        if (sdpa) {
            for (size_t i = 0; i < 2; i++) {
                if (scaled_down_subgraph.find(node->input(i).get_source_output().get_node_shared_ptr()->get_friendly_name()) != scaled_down_subgraph.end()) {
                    std::shared_ptr<ov::Node> scale_const = (node->get_input_element_type(i) == ov::element::f16) ? scale_const_f16 : scale_const_f32;
                    auto transpose = std::dynamic_pointer_cast<ov::op::v1::Transpose>(node->input(i).get_source_output().get_node_shared_ptr());
                    if (transpose) {
                        auto scale_up = std::make_shared<ov::op::v1::Multiply>(transpose->get_input_source_output(0),
                                                                               scale_const->output(0));
                        transpose->input(0).replace_source_output(scale_up->output(0));
                    } else {
                        auto scale_up = std::make_shared<ov::op::v1::Multiply>(node->get_input_source_output(i),
                                                                               scale_const->output(0));
                        node->input(i).replace_source_output(scale_up->output(0));
                    }
                }
            }

            if (scaled_down_subgraph.find(node->input(2).get_source_output().get_node_shared_ptr()->get_friendly_name()) != scaled_down_subgraph.end()) {
                scaled_down_subgraph.insert(node->get_friendly_name());
            }
            continue;
        }

        // input(scaled_down) -- activation
        // ==>
        // input(scaled_down) -- multiply(scale_up) -- activation -- multiply(scale_down)
        auto sin = std::dynamic_pointer_cast<ov::op::v0::Sin>(node);
        auto cos = std::dynamic_pointer_cast<ov::op::v0::Cos>(node);
        auto swish = std::dynamic_pointer_cast<ov::op::v4::Swish>(node);
        if ((sin || cos || swish) && num_scaled_down_inputs == 1) {
            std::shared_ptr<ov::Node> scale_const = (node->get_input_element_type(0) == ov::element::f16) ? scale_const_f16 : scale_const_f32;
            auto scale_up = std::make_shared<ov::op::v1::Multiply>(node->get_input_source_output(0),
                                                                   scale_const->output(0));
            node->input(0).replace_source_output(scale_up->output(0));

            std::shared_ptr<ov::Node> inverse_scale_const = (node->get_output_element_type(0) == ov::element::f16) ?
                                                            inverse_scale_const_f16 : inverse_scale_const_f32;
            auto scale_down = std::make_shared<ov::op::v1::Multiply>(node->output(0),
                                                                     inverse_scale_const->output(0));
            ov::replace_node(node, scale_down);
            scaled_down_subgraph.insert(scale_down->get_friendly_name());
            std::cout << "scale activation " << node->get_friendly_name() << std::endl;
        }

        if (num_scaled_down_inputs > 0)
            scaled_down_subgraph.insert(node->get_friendly_name());
    }

    return true;
}

ov::pass::StaticScaling::StaticScaling(float scale_factor) {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    const ov::element::Type infer_prec = ov::element::f16;
    const ov::element::Type scaled_prec = ov::element::f32;

    scale_factor = get_scale_factor(scale_factor);

    auto input_m = any_input();
    auto swish_m = wrap_type<ov::op::v4::Swish>({ input_m });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(swish_m));

        auto swish = std::dynamic_pointer_cast<ov::op::v4::Swish>(pattern_map.at(swish_m).get_node_shared_ptr());
        if (!swish || transformation_callback(swish))
            return false;

        if (swish->get_input_element_type(0) != infer_prec || swish->get_output_element_type(0) != infer_prec) {
            return false;
        }

        auto input = pattern_map.at(input_m);
        auto precision_up = std::make_shared<ov::op::v0::Convert>(input, scaled_prec);

        ov::Shape scale_const_shape = {1};
        std::vector<float> scale_value = {scale_factor};
        std::shared_ptr<ov::Node> scale_const = std::make_shared<ov::op::v0::Constant>(scaled_prec, scale_const_shape, scale_value);
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(precision_up->output(0),
                                                               scale_const->output(0));
        swish->input(0).replace_source_output(scale_up->output(0));
        swish->revalidate_and_infer_types();

        std::vector<float> inverse_scale_value = {(1.f / scale_factor)};
        std::shared_ptr<ov::Node> inverse_scale_const = std::make_shared<ov::op::v0::Constant>(scaled_prec, scale_const_shape, inverse_scale_value);
        auto scale_down = std::make_shared<ov::op::v1::Multiply>(swish->output(0),
                                                                 inverse_scale_const->output(0));
        ov::replace_node(swish, scale_down);

        auto precision_down = std::make_shared<ov::op::v0::Convert>(scale_down->output(0), infer_prec);
        ov::replace_node(scale_down, precision_down);
std::cout << "StaticScaling - converted " << scale_factor << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(swish_m, "StaticScaling");
    this->register_matcher(m, callback);
}

ov::pass::StaticScalingInput::StaticScalingInput(float scale_factor) {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    scale_factor = get_scale_factor(scale_factor);

    auto input_m = wrap_type<ov::op::v0::Parameter>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(input_m));

        auto input = std::dynamic_pointer_cast<ov::op::v0::Parameter>(pattern_map.at(input_m).get_node_shared_ptr());
        if (!input || transformation_callback(input))
            return false;

        auto input_prec = input->get_output_element_type(0);

        ov::Shape scale_const_shape = {1};
        std::vector<float> inverse_scale_value = {(1.f / scale_factor)};
        std::shared_ptr<ov::Node> inverse_scale_const = std::make_shared<ov::op::v0::Constant>(input_prec, scale_const_shape, inverse_scale_value);
        auto scale_down = std::make_shared<ov::op::v1::Multiply>(input->output(0),
                                                                 inverse_scale_const->output(0));

        ov::replace_node(input, scale_down);
std::cout << "StaticScalingInput - converted " << scale_factor << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(input_m, "StaticScalingInput");
    this->register_matcher(m, callback);
}

ov::pass::StaticScalingOutput::StaticScalingOutput(float scale_factor) {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    scale_factor = get_scale_factor(scale_factor);

    auto input_m = any_input();
    auto output_m = wrap_type<ov::op::v0::Result>({input_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(output_m));

        auto output = std::dynamic_pointer_cast<ov::op::v0::Result>(pattern_map.at(output_m).get_node_shared_ptr());
        if (!output || transformation_callback(output))
            return false;

        auto output_prec = output->get_input_element_type(0);

        ov::Shape scale_const_shape = {1};
        std::vector<float> scale_value = {scale_factor};
        std::shared_ptr<ov::Node> scale_const = std::make_shared<ov::op::v0::Constant>(output_prec, scale_const_shape, scale_value);
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(output->get_input_source_output(0),
                                                               scale_const->output(0));
        output->input(0).replace_source_output(scale_up->output(0));
std::cout << "StaticScalingOutput - converted " << scale_factor << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(output_m, "StaticScalingOutput");
    this->register_matcher(m, callback);
}

ov::pass::StaticScalingAdd::StaticScalingAdd(float scale_factor) {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    scale_factor = get_scale_factor(scale_factor);

    auto input_m = any_input();
    auto const_input_m = wrap_type<ov::op::v0::Constant>();
    auto add_m = wrap_type<ov::op::v1::Add>({input_m, const_input_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(add_m));

        auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(pattern_map.at(add_m).get_node_shared_ptr());
        if (!add || transformation_callback(add))
            return false;

        auto const_input = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(const_input_m).get_node_shared_ptr());
        if (!const_input)
            return false;

        auto runtime_prec = add->get_input_element_type(0);

        ov::Shape scale_const_shape = {1};
        std::vector<float> inverse_scale_value = {(1.f / scale_factor)};
        std::shared_ptr<ov::Node> inverse_scale_const = std::make_shared<ov::op::v0::Constant>(runtime_prec, scale_const_shape, inverse_scale_value);
        auto scale_down = std::make_shared<ov::op::v1::Multiply>(const_input->output(0),
                                                                 inverse_scale_const->output(0));
        ov::replace_node(const_input, scale_down);
std::cout << "StaticScalingAdd - converted " << scale_factor << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_m, "StaticScalingAdd");
    this->register_matcher(m, callback);
}

// ov::pass::StaticScaling::StaticScaling(float scale_factor) {
//     using namespace ov::pass::pattern;
//     using ov::pass::pattern::op::Or;

//     const float default_scale_factor = 256.f;
//     const ov::element::Type infer_prec = ov::element::f32;
//     const ov::element::Type scaled_prec = ov::element::f16;

//     scale_factor = (scale_factor < 1.f) ? default_scale_factor : scale_factor;

//     auto input_m = any_input();
//     auto weights_m = wrap_type<ov::op::v0::Constant>(type_matches_any({infer_prec}));
//     auto convolution_m = wrap_type<ov::op::v1::Convolution>({ input_m, weights_m });

//     ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
//         const auto& pattern_map = m.get_pattern_value_map();

//         OPENVINO_ASSERT(pattern_map.count(convolution_m));

//         auto conv = std::dynamic_pointer_cast<ov::op::v1::Convolution>(pattern_map.at(convolution_m).get_node_shared_ptr());
//         if (!conv || transformation_callback(conv))
//             return false;

//         if (conv->get_input_element_type(0) != infer_prec || conv->get_output_element_type(0) != infer_prec) {
//             return false;
//         }

//         auto conv_weight = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(weights_m).get_node_shared_ptr());
//         auto conv_weight_convert = std::make_shared<ov::op::v0::Convert>(conv_weight, scaled_prec);
//         ov::replace_node(conv_weight, conv_weight_convert);

//         auto input = pattern_map.at(input_m);

//         ov::Shape scale_const_shape = {1};
//         std::vector<float> inverse_scale_value = {(1.f / scale_factor)};
//         std::shared_ptr<ov::Node> inverse_scale_const = std::make_shared<ov::op::v0::Constant>(infer_prec, scale_const_shape, inverse_scale_value);
//         auto scale_down = std::make_shared<ov::op::v1::Multiply>(input.get_node_shared_ptr()->output(0),
//                                                                  inverse_scale_const->output(0));
//         auto precision_down = std::make_shared<ov::op::v0::Convert>(scale_down, scaled_prec);
//         conv->input(0).replace_source_output(precision_down->output(0));

//         std::vector<float> scale_value = {scale_factor};
//         std::shared_ptr<ov::Node> scale_const = std::make_shared<ov::op::v0::Constant>(infer_prec, scale_const_shape, scale_value);
//         auto scale_up = std::make_shared<ov::op::v1::Multiply>(conv->output(0),
//                                                                scale_const->output(0));
//         ov::replace_node(conv, scale_up);
// std::cout << "StaticScaling - converted " << scale_factor << std::endl;
//         return true;
//     };

//     auto m = std::make_shared<ov::pass::pattern::Matcher>(convolution_m, "StaticScaling");
//     this->register_matcher(m, callback);
// }

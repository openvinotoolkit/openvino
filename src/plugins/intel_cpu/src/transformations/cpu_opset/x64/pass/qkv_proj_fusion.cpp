// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qkv_proj_fusion.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/cpu_opset/x64/op/qkv_proj.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

using namespace ov::pass;
using namespace ov::op;

ov::intel_cpu::QKVProjFusionPass1::QKVProjFusionPass1() {
    MATCHER_SCOPE(QKVProjFusionPass1);

    auto input = pattern::any_input(pattern::rank_equals(3));

    auto q_proj_weight_const_i8 =
        pattern::wrap_type<v0::Constant>(pattern::type_matches(element::i8) && pattern::rank_equals(2));
    auto q_proj_weight_f32 =
        pattern::wrap_type<v0::Convert>({q_proj_weight_const_i8}, pattern::type_matches(element::f32));
    auto q_proj_weight_scales_per_OC =
        pattern::wrap_type<v0::Constant>(pattern::type_matches(element::f32) && pattern::shape_matches("[?, 1]"));
    auto q_proj_weight_deq = pattern::wrap_type<v1::Multiply>({q_proj_weight_f32, q_proj_weight_scales_per_OC},
                                                              {{"auto_broadcast", "numpy"}});

    auto q_proj_weight_const = pattern::wrap_const();
    auto q_proj_weight_cvt =
        pattern::optional<op::v0::Convert>({q_proj_weight_const}, pattern::type_matches(element::f32));  //  [4096,4096]
    auto q_proj = pattern::wrap_type<v0::MatMul>({input, q_proj_weight_cvt | q_proj_weight_deq},
                                                 {{"transpose_a", false}, {"transpose_b", true}});  //  [?,?,4096]

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto src = pattern_map.at(input);

        auto&& children = src.get_target_inputs();

        if (children.size() < 3) {
            return false;
        }

        if (!src.get_element_type().is_real()) {
            return false;
        }

        bool is_quantized_int8 = pattern_map.find(q_proj_weight_const_i8) != pattern_map.end();

        OutputVector args = {src};
        OutputVector deq_scales;
        OutputVector outputs;
        size_t hidden_size = 0;
        std::vector<int> proj_size;
        for (const auto& child : children) {
            auto* mm = ov::as_type<op::v0::MatMul>(child.get_node());
            if (!mm) {
                // maybe a ShapeOf
                continue;
            }
            if (mm->get_transpose_a() || !mm->get_transpose_b()) {
                return false;
            }

            auto mm_input1 = mm->input_value(1).get_node_shared_ptr();

            std::shared_ptr<op::v0::Constant> constw;
            std::shared_ptr<op::v0::Constant> deq_scale;

            if (is_quantized_int8) {
                auto deq_mul = ov::as_type_ptr<ov::op::v1::Multiply>(mm_input1);
                if (!deq_mul) {
                    return false;
                }

                auto deq_mul_in0 = deq_mul->input_value(0).get_node_shared_ptr();
                auto deq_mul_in1 = deq_mul->input_value(1).get_node_shared_ptr();

                auto cvt = ov::as_type_ptr<op::v0::Convert>(deq_mul_in0);
                if (!cvt) {
                    return false;
                }

                constw = ov::as_type_ptr<op::v0::Constant>(cvt->input_value(0).get_node_shared_ptr());
                if (!constw || constw->get_element_type() != ov::element::i8) {
                    return false;
                }

                deq_scale = ov::as_type_ptr<op::v0::Constant>(deq_mul_in1);
                if (!deq_scale || deq_scale->get_element_type() != ov::element::f32) {
                    return false;
                }
            } else {
                constw = ov::as_type_ptr<op::v0::Constant>(mm_input1);
                if (!constw) {
                    if (auto cvt = ov::as_type_ptr<op::v0::Convert>(mm_input1)) {
                        constw = ov::as_type_ptr<op::v0::Constant>(cvt->input_value(0).get_node_shared_ptr());
                    } else {
                        return false;
                    }
                }
                if (!constw) {
                    return false;
                }
            }

            // input feature size should be the same
            const auto& wshape = constw->get_shape();
            if (hidden_size == 0) {
                hidden_size = wshape[1];
            } else if (hidden_size != wshape[1]) {
                return false;
            }

            proj_size.push_back(wshape[0]);
            args.emplace_back(constw);
            deq_scales.emplace_back(deq_scale);
            outputs.push_back(mm->get_default_output());
        }

        // make sure just 3 projections are found
        if (outputs.size() != 3) {
            return false;
        }
        if (args.size() != 4) {
            return false;
        }
        // append dequantize scales at the end
        if (is_quantized_int8) {
            for (auto& d : deq_scales) {
                args.push_back(d);
            }
        }

        QKVProjectionNode::Config config{is_quantized_int8,
                                         static_cast<int>(hidden_size),
                                         proj_size[0],
                                         proj_size[1],
                                         proj_size[2],
                                         false};

        auto old_node = root;
        auto new_node = std::make_shared<QKVProjectionNode>(args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({old_node}, new_node);

        // callback is for plugin implementation to check if it can be supported
        if (!transformation_callback(new_node)) {
            return false;
        }

        for (size_t i = 0; i < outputs.size(); i++) {
            auto target = outputs[i].get_node_shared_ptr();
            outputs[i].replace(new_node->output(i));
            new_node->add_node_control_dependents(target);
            new_node->add_node_control_dependencies(target);
            target->clear_control_dependents();
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(q_proj, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::QKVProjFusionPass2::QKVProjFusionPass2() {
    MATCHER_SCOPE(QKVProjFusionPass2);

    auto input = pattern::any_input(pattern::rank_equals(3));

    auto qkv_proj_weight_const = pattern::wrap_const();
    auto qkv_proj_cvt =
        pattern::wrap_type<op::v0::Convert>({qkv_proj_weight_const}, pattern::type_matches(element::f32));

    auto qkv_proj_weight_const_i8 =
        pattern::wrap_type<v0::Constant>(pattern::type_matches(element::i8) && pattern::rank_equals(2));
    auto qkv_proj_weight_f32 =
        pattern::wrap_type<op::v0::Convert>({qkv_proj_weight_const_i8}, {{"destination_type", "f32"}});
    auto qkv_proj_weight_scales_per_OC =
        pattern::wrap_type<v0::Constant>(pattern::type_matches(element::f32) && pattern::shape_matches("[?, 1]"));
    auto qkv_proj_weight_deq =
        pattern::wrap_type<ov::op::v1::Multiply>({qkv_proj_weight_f32, qkv_proj_weight_scales_per_OC},
                                                 {{"auto_broadcast", "numpy"}});

    auto qkv_proj = pattern::wrap_type<op::v0::MatMul>({input, qkv_proj_cvt | qkv_proj_weight_deq},
                                                       {{"transpose_a", false}, {"transpose_b", true}});
    auto qkv_split_lengths =
        pattern::wrap_type<op::v0::Constant>(pattern::type_matches(element::i32) && pattern::shape_matches("[3]"));
    auto qkv_split = pattern::wrap_type<ov::op::v1::VariadicSplit>({qkv_proj, 2, qkv_split_lengths});
    auto result = qkv_split->output(0);

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto node_split_lengths =
            ov::as_type_ptr<op::v0::Constant>(pattern_map.at(qkv_split_lengths).get_node_shared_ptr());
        if (!node_split_lengths) {
            return false;
        }
        auto split_lengths = node_split_lengths->get_vector<int32_t>();
        if (split_lengths.size() != 3) {
            return false;
        }

        auto proj_size = split_lengths[0];
        if (split_lengths[1] != proj_size) {
            return false;
        }
        if (split_lengths[2] != proj_size) {
            return false;
        }

        bool is_quantized_int8 = pattern_map.find(qkv_proj_weight_const_i8) != pattern_map.end();

        std::shared_ptr<op::v0::Constant> qkv_proj_weight_node;
        if (is_quantized_int8) {
            qkv_proj_weight_node =
                ov::as_type_ptr<op::v0::Constant>(pattern_map.at(qkv_proj_weight_const_i8).get_node_shared_ptr());
        } else {
            qkv_proj_weight_node =
                ov::as_type_ptr<op::v0::Constant>(pattern_map.at(qkv_proj_weight_const).get_node_shared_ptr());
        }
        if (!qkv_proj_weight_node) {
            return false;
        }

        auto w_shape = qkv_proj_weight_node->get_shape();
        if (w_shape[0] != static_cast<uint64_t>(proj_size) * 3) {
            return false;
        }

        QKVProjectionNode::Config config{is_quantized_int8,
                                         static_cast<int>(w_shape[1]),
                                         1,
                                         split_lengths[0],
                                         split_lengths[1],
                                         static_cast<bool>(split_lengths[2])};

        OutputVector args = {pattern_map.at(input), qkv_proj_weight_node, qkv_proj_weight_node, qkv_proj_weight_node};
        if (is_quantized_int8) {
            auto scales = pattern_map.at(qkv_proj_weight_scales_per_OC).get_node_shared_ptr();
            args.emplace_back(scales);
            args.emplace_back(scales);
            args.emplace_back(scales);
        }
        auto old_node = root;
        auto new_node = std::make_shared<QKVProjectionNode>(args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({old_node}, new_node);

        // callback is for plugin implementation to check if it can be supported
        if (!transformation_callback(new_node)) {
            return false;
        }

        auto vsplit = pattern_map.at(qkv_split).get_node_shared_ptr();

        for (size_t i = 0; i < vsplit->get_output_size(); i++) {
            vsplit->output(i).replace(new_node->output(i));
        }

        new_node->add_node_control_dependents(vsplit);
        new_node->add_node_control_dependencies(vsplit);
        vsplit->clear_control_dependents();

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

bool ov::intel_cpu::QKVProjFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(QKVProjFusion);

    SymbolicOptimizations symbolic_optimizations(false, get_pass_config());
    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();

    symbolic_ctx_manager->register_pass<QKVProjFusionPass1>();
    symbolic_ctx_manager->register_pass<QKVProjFusionPass2>();

    return symbolic_optimizations.run_on_model(model);
}

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qkv_proj_fusion.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
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
#include "openvino/util/pp.hpp"
#include "transformations/cpu_opset/x64/op/qkv_proj.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov::gen_pattern;
using namespace ov::pass;

ov::intel_cpu::QKVProjFusion::QKVProjFusion() {
    MATCHER_SCOPE(QKVProjFusion);

    auto input = makePattern("[?,?,?]");

    auto q_proj_weight_const = makePattern<op::v0::Constant>({});

    auto q_proj_weight_const_i8 =
        makeConst(ov::element::i8, ov::PartialShape({ov::Dimension(), ov::Dimension()}), nullptr);
    auto q_proj_weight_f32 = makePattern<op::v0::Convert>({q_proj_weight_const_i8}, {{"destination_type", "f32"}});
    auto q_proj_weight_scales_per_OC = makeConst(ov::element::f32, ov::PartialShape({ov::Dimension(), 1}), nullptr);
    auto q_proj_weight_deq = makePattern<ov::op::v1::Multiply>({q_proj_weight_f32, q_proj_weight_scales_per_OC},
                                                               {{"auto_broadcast", "numpy"}});

    auto q_proj_weight_cvt =
        makePattern<op::v0::Convert>({q_proj_weight_const}, {{"destination_type", "f32"}});  //  [4096,4096]
    auto q_proj = makePattern<op::v0::MatMul>({input, q_proj_weight_cvt | q_proj_weight_const | q_proj_weight_deq},
                                              {{"transpose_a", false}, {"transpose_b", true}});  //  [?,?,4096]
    auto result = q_proj;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

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
        for (auto& child : children) {
            auto mm = ov::as_type<op::v0::MatMul>(child.get_node());
            if (!mm) {
                // maybe a ShapeOf
                continue;
            }
            if (mm->get_transpose_a() != false || mm->get_transpose_b() != true) {
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

        QKVProjectionNode::Config config;
        config.quantized = is_quantized_int8;
        config.hidden_size = hidden_size;
        config.weights_combined = false;
        config.proj_size0 = proj_size[0];
        config.proj_size1 = proj_size[1];
        config.proj_size2 = proj_size[2];

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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::QKVProjFusion2::QKVProjFusion2() {
    MATCHER_SCOPE(QKVProjFusion2);

    auto input = makePattern("[?,?,?]");

    auto qkv_proj_weight_const = makePattern<op::v0::Constant>({});
    auto qkv_proj_cvt = makePattern<op::v0::Convert>({qkv_proj_weight_const}, {{"destination_type", "f32"}});

    auto qkv_proj_weight_const_i8 =
        makeConst(ov::element::i8, ov::PartialShape({ov::Dimension(), ov::Dimension()}), nullptr);
    auto qkv_proj_weight_f32 = makePattern<op::v0::Convert>({qkv_proj_weight_const_i8}, {{"destination_type", "f32"}});
    auto qkv_proj_weight_scales_per_OC = makeConst(ov::element::f32, ov::PartialShape({ov::Dimension(), 1}), nullptr);
    auto qkv_proj_weight_deq = makePattern<ov::op::v1::Multiply>({qkv_proj_weight_f32, qkv_proj_weight_scales_per_OC},
                                                                 {{"auto_broadcast", "numpy"}});

    auto qkv_proj = makePattern<op::v0::MatMul>({input, qkv_proj_cvt | qkv_proj_weight_deq},
                                                {{"transpose_a", false}, {"transpose_b", true}});
    auto qkv_split_lengths = makePattern<op::v0::Constant>({}, {}, "i32[3]");
    auto qkv_split = makePattern<ov::op::v1::VariadicSplit>({qkv_proj, 2, qkv_split_lengths});

    auto result = qkv_split->output(0);

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

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

        QKVProjectionNode::Config config;
        config.quantized = is_quantized_int8;
        config.hidden_size = w_shape[1];
        config.weights_combined = true;
        config.proj_size0 = split_lengths[0];
        config.proj_size1 = split_lengths[1];
        config.proj_size2 = split_lengths[2];

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

// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remove_unnecessary_pad.hpp"
#include <memory>

#include "openvino/op/constant.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

RemoveUnnecessaryPad::RemoveUnnecessaryPad() {
    using namespace ov::op;
    using namespace ov::pass;
    auto pads_begin = pattern::wrap_type<v0::Constant>();
    auto pads_end = pattern::wrap_type<v0::Constant>();
    auto arg_pad_value = pattern::wrap_type<v0::Constant>();
    auto pad = pattern::wrap_type<v12::Pad>({ov::pass::pattern::any_input(),
                                                               pads_begin, pads_end, arg_pad_value});
    auto maxpool_v1 = pattern::wrap_type<v1::MaxPool>({pad});
    auto maxpool_v8 = pattern::wrap_type<v8::MaxPool>({pad});
    auto maxpool_v1_or_v8 = std::make_shared<pattern::op::Or>(OutputVector{maxpool_v1, maxpool_v8});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto get_maxpool = [&pattern_map](const std::shared_ptr<Node>& v1_pattern,
                                          const std::shared_ptr<Node>& v8_pattern) -> std::shared_ptr<Node> {
            if (pattern_map.count(v1_pattern))
                return pattern_map.at(v1_pattern).get_node_shared_ptr();
            if (pattern_map.count(v8_pattern))
                return pattern_map.at(v8_pattern).get_node_shared_ptr();
            return nullptr;
        };

        auto mp = get_maxpool(maxpool_v1, maxpool_v8);
        if (!mp) {
            return false;
        }

        auto all_zero = [](std::shared_ptr<ov::op::v0::Constant> constant_node) {
            if (constant_node) {
                auto element_type = constant_node->get_element_type();
                if (element_type == ov::element::i32) {
                    auto vec = constant_node->get_vector<int32_t>();
                    return (std::count_if(vec.begin(), vec.end(), [](int32_t elem) { return elem != 0; }) == 0);
                } else if (element_type == ov::element::i64) {
                    auto vec = constant_node->get_vector<int64_t>();
                    return (std::count_if(vec.begin(), vec.end(), [](int64_t elem) { return elem != 0; }) == 0);
                }
            }
            return false;
        };

        auto pads_begin_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(pads_begin).get_node_shared_ptr());
        auto pads_end_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(pads_end).get_node_shared_ptr());

        if (!all_zero(pads_begin_node) || !all_zero(pads_end_node))
            return false;

        auto pad_node = std::dynamic_pointer_cast<v12::Pad>(pattern_map.at(pad).get_node_shared_ptr());
        auto type_info = mp->get_type_info();

        if (type_info.get_version() == "opset1") {
            auto mp_node = std::dynamic_pointer_cast<v1::MaxPool>(mp);
            auto updated_mp =
                std::make_shared<v1::MaxPool>(pad_node->input_value(0), mp_node->get_strides(),
                                              mp_node->get_pads_begin(), mp_node->get_pads_end(),
                                              mp_node->get_kernel(), mp_node->get_rounding_type(),
                                              mp_node->get_auto_pad());
            copy_runtime_info(mp_node, updated_mp);
            updated_mp->set_friendly_name(mp_node->get_friendly_name());
            replace_node(mp_node, updated_mp);
            return true;
        } else {  // opset8
            auto mp_node = std::dynamic_pointer_cast<v8::MaxPool>(mp);
            auto updated_mp =
                std::make_shared<v8::MaxPool>(pad_node->input_value(0),
                                              mp_node->get_strides(), mp_node->get_dilations(),
                                              mp_node->get_pads_begin(), mp_node->get_pads_end(),
                                              mp_node->get_kernel(), mp_node->get_rounding_type(),
                                              mp_node->get_auto_pad(), mp_node->get_index_element_type(),
                                              mp_node->get_axis());
            copy_runtime_info(mp_node, updated_mp);
            updated_mp->set_friendly_name(mp_node->get_friendly_name());
            replace_node(mp_node, updated_mp);
            return true;
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(maxpool_v1_or_v8, "RemoveUnnecessaryPad");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov

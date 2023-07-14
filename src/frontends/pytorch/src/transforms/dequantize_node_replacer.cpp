// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dequantize_node_replacer.hpp"

#include <list>
#include <memory>
#include <utility>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

#define MAX_SEARCH_NODES 100000

using namespace ov::op;

/**
 * Dequantize Node Replacer
 * Replacer finds the unconverted dequantize ops and converts them 
 * using scale/zero_point from the matching quantized input nodes.
 * To obtain them, a BFS search is performed on the graph structure.
 */
DequantizeNodeReplacer::DequantizeNodeReplacer() {
    auto dequantize_node = ov::pass::pattern::wrap_type<ov::frontend::pytorch::PtFrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto dequantize_node = cast_fw_node(m.get_match_root(), "aten::dequantize");
        if (!dequantize_node)
            return false;

        std::list<std::shared_ptr<ov::Node>> inputs = {dequantize_node->get_input_node_shared_ptr(0)};
        std::shared_ptr<ov::frontend::pytorch::QuantizedPtNode> quantized_pt_node;
        size_t iter = 0;
        while (inputs.size() && iter++ < MAX_SEARCH_NODES) {
            auto node = inputs.front();
            inputs.pop_front();
            if ((quantized_pt_node = cast_quantized_fw_node(node)))
                break;
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                inputs.push_back(node->get_input_node_shared_ptr(i));
            }
        }
        if (quantized_pt_node) {
            ov::pass::NodeRegistry rg;
            const auto input =
                rg.make<v0::Convert>(quantized_pt_node->input_value(0).get_node_shared_ptr(), element::f32);
            // const auto scale = quantized_pt_node->get_scale();
            // const auto zero_point = quantized_pt_node->get_zero_point();
            // const auto scale_convert = rg.make<v0::Convert>(scale, element::f32);
            // const auto zero_point_convert = rg.make<v0::Convert>(zero_point, element::f32);
            // const auto input_sub_zero_pt = rg.make<v1::Subtract>(input, zero_point_convert);
            // const auto dequantized_input = rg.make<v1::Multiply>(input_sub_zero_pt, scale_convert);
            // copy_runtime_info_and_name(dequantize_node, rg.get(), {input});
            // replace_node(dequantize_node, dequantized_input);
            // return true;

            copy_runtime_info_and_name(dequantize_node, rg.get(), {input});
            replace_node(dequantize_node, input);
            return true;
        }
        add_exception_to_fw_node(dequantize_node, "aten::dequantize could not find a matching quantized input.");
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(dequantize_node,
                                                          "ov::frontend::pytorch::pass::DequantizeNodeReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

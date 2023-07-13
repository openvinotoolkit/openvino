// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dequantize_node_replacer.hpp"

#include <memory>
#include <utility>

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

DequantizeNodeReplacer::DequantizeNodeReplacer() {
    auto dequantize_node = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto dequantize_node = cast_fw_node(m.get_match_root(), "aten::dequantize");
        if (!quantized_pt_node)
            return false;

        ov::pass::NodeRegistry rg;
        auto quantized_input = quantized_pt_node->get_input_node_shared_ptr(0);

        copy_runtime_info_and_name(quantized_pt_node, rg.get(), {quantized_input});
        replace_node(quantized_pt_node, quantized_input);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(dequantize_node,
                                                          "ov::frontend::pytorch::pass::DequantizeNodeReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
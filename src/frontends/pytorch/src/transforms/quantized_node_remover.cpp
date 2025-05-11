// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantized_node_remover.hpp"

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

QuantizedNodeRemover::QuantizedNodeRemover() {
    auto quantized_pt_node = ov::pass::pattern::wrap_type<ov::frontend::pytorch::QuantizedPtNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto quantized_pt_node = cast_quantized_fw_node(m.get_match_root());
        if (!quantized_pt_node)
            return false;

        auto quantized_input = quantized_pt_node->input_value(0);
        quantized_pt_node->output(0).replace(quantized_input);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(quantized_pt_node,
                                                          "ov::frontend::pytorch::pass::QuantizedNodeRemover");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

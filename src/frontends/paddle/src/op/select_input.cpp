// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs select_input(const NodeContext& node) {
    const auto x = node.get_ng_inputs("X");
    const auto mask = node.get_input("Mask");

    PADDLE_OP_CHECK(node, x.size() == 2, "select_input needs 2 input nodes.");

    const auto cond = std::make_shared<default_opset::Convert>(mask, element::boolean);
    const auto ps0 = x[0].get_partial_shape();
    const auto ps1 = x[1].get_partial_shape();

    if (ps0.compatible(ps1)) {
        auto placehodler = std::make_shared<default_opset::Select>(cond, x[1], x[0]);
        return node.default_single_output_mapping({placehodler}, {"Out"});
    } else {
        // paddle detection model code is wrong and will result a dynamic rank model:
        //   https://github.com/PaddlePaddle/PaddleDetection/blob/16e3d7408161713c765886cfb952f98d9f68713c/ppdet/modeling/layers.py#L407
        // workaround: check the rank and remove the wrong condition
        if (ps0.rank() != ps1.rank()) {
            const auto fix_idx = [&](int idx) {
                const auto ps = x[idx].get_partial_shape();
                if (ps.is_static()) {
                    const Shape shape(ps.get_shape());
                    const auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
                    if (size == 0)
                        return 1 - idx;
                }
                return idx;
            };
            auto placehodler = std::make_shared<default_opset::Select>(cond, x[fix_idx(1)], x[fix_idx(0)]);
            return node.default_single_output_mapping({placehodler}, {"Out"});
        }
        PADDLE_OP_CHECK(node, false, "input shapes should be compatible.");

        return {};
    }
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
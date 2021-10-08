// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateStridedSliceOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto begin = node.get_ng_input(1);
    auto end = node.get_ng_input(2);
    auto strides = node.get_ng_input(3);

    auto begin_mask = node.get_attribute<int32_t>("begin_mask");
    auto end_mask = node.get_attribute<int32_t>("end_mask");
    auto new_axis_mask = node.get_attribute<int32_t>("new_axis_mask");
    auto ellipsis_mask = node.get_attribute<int32_t>("ellipsis_mask");
    auto shrink_axis_mask = node.get_attribute<int32_t>("shrink_axis_mask");

    // TODO (itikhono): check algorithm
    auto mask_to_vec = [](int32_t mask) {
        auto length = sizeof(mask) * CHAR_BIT;
        std::vector<int64_t> vec(length, 0);
        if (mask == 0) {
            return vec;
        }
        for (auto i = 0; i < length; ++i) {
            if (static_cast<unsigned char>(mask >> i & 0x01) == 1) {
                vec[i] = 1;
            }
        }
        return vec;
    };

    auto strided_slice = make_shared<StridedSlice>(input,
                                                   begin,
                                                   end,
                                                   strides,
                                                   mask_to_vec(begin_mask),
                                                   mask_to_vec(end_mask),
                                                   mask_to_vec(new_axis_mask),
                                                   mask_to_vec(shrink_axis_mask),
                                                   mask_to_vec(ellipsis_mask));
    strided_slice->set_friendly_name(node.get_name());
    return strided_slice->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph

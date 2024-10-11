// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "torchfx_nncf_pattern_replacer.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/convert_promote_types.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;
using namespace ov::pass::pattern;

NNCFDecompressionReplacer::NNCFDecompressionReplacer() {
    const auto& const_3 = wrap_type<v0::Constant>();
    const auto& const_4 = wrap_type<v0::Constant>();
    const auto& convert_4 = wrap_type<v0::Convert>({const_4});
    const auto& bitwise_and = wrap_type<ov::op::v13::BitwiseAnd>({const_3, convert_4});
    const auto& const_unsqueeze_1 = wrap_type<v0::Constant>();
    const auto& unsqueeze_1 = wrap_type<v0::Unsqueeze>({bitwise_and, const_unsqueeze_1});

    const auto& const_1 = wrap_type<v0::Constant>();
    const auto& const_2 = wrap_type<v0::Constant>();
    const auto& bitwise_right_shift = wrap_type<ov::op::util::FrameworkNode>({const_1, const_2});
    const auto& const_unsqueeze_2 = wrap_type<v0::Constant>();
    const auto& unsqueeze_2 = wrap_type<v0::Unsqueeze>({bitwise_right_shift, const_unsqueeze_2});

    const auto& convert_promote = wrap_type<ov::op::v14::ConvertPromoteTypes>({unsqueeze_1, unsqueeze_2});
    const auto& convert_like = wrap_type<ov::op::v1::ConvertLike>({any_input(), convert_promote->output(0)});
    const auto& concat = wrap_type<v0::Concat>({convert_promote->output(0), convert_like});

    const auto& const_reshape = wrap_type<v0::Constant>();
    const auto& reshape = wrap_type<ov::op::v1::Reshape>({concat, const_reshape});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        auto reshape = m.get_match_root();
        if (!reshape) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();   
        const auto& unsqueeze_1_node = pattern_map.at(unsqueeze_1).get_node_shared_ptr();
        const auto& unsqueeze_2_node = pattern_map.at(unsqueeze_2).get_node_shared_ptr();

        auto unsqueeze_1_const = std::dynamic_pointer_cast<v0::Constant>(unsqueeze_1_node->get_input_node_shared_ptr(1));
        auto unsqueeze_2_const = std::dynamic_pointer_cast<v0::Constant>(unsqueeze_2_node->get_input_node_shared_ptr(1));

        if (ov::shape_size(unsqueeze_1_const->get_shape()) != 1 || ov::shape_size(unsqueeze_2_const->get_shape()) != 1)
            return false;

        int32_t axis = unsqueeze_1_const->get_data_ptr<int32_t>()[0];
        if (axis != unsqueeze_2_const->get_data_ptr<int32_t>()[0])
            return false;

        const auto& bitwise_and_node = pattern_map.at(bitwise_and).get_node_shared_ptr();
        const auto& bitwise_right_shift_node = pattern_map.at(bitwise_right_shift).get_node_shared_ptr();

        const auto& u4_const = u4_compression_stack({bitwise_and_node, bitwise_right_shift_node}, (int64_t)axis);
        if (!u4_const)
            return false;

        OutputVector reshape_outputs(1);
        reshape->constant_fold(reshape_outputs, u4_const->outputs());
        const auto& reshaped_u4_const = reshape_outputs[0].get_node_shared_ptr();

        replace_node(reshape, reshaped_u4_const);
        return true;
    };

    auto m = std::make_shared<Matcher>(reshape, "ov::frontend::pytorch::pass::NNCFDecompressionReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

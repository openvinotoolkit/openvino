// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_index_copy_replacer.hpp"

#include <iostream>

#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/pytorch/visibility.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

namespace {}  // namespace

AtenIndexCopyReplacer::AtenIndexCopyReplacer() {
    auto index_op = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto index_op = cast_fw_node(m.get_match_root(), "aten::index_copy_");
        if (!index_op) {
            return false;
        }
        NodeVector rt_copy_from;
        ov::pass::NodeRegistry rg;
        auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
        auto const_1 = v0::Constant::create(element::i32, Shape{}, {1});
        auto const_neg_1 = v0::Constant::create(element::i32, Shape{}, {-1});

        auto input = index_op->input_value(0);
        auto dim = index_op->input_value(1);
        auto index = index_op->input_value(2);
        auto tensor = index_op->input_value(3);

        auto input_shape = rg.make<v3::ShapeOf>(input, element::i32);
        auto input_rank = rg.make<v8::Gather>(input_shape, const_0, const_0);
        auto input_rank_correct_type = rg.make<v1::ConvertLike>(input_rank, index);
        auto positive_dim = rg.make<v1::Add>(dim, input_rank_correct_type);
        auto index_partial_shape = index.get_partial_shape();
        auto index_rank_scalar = index_partial_shape.rank().get_length();
        auto dim_scalar = std::dynamic_pointer_cast<v0::Constant>(dim.get_node_shared_ptr())->cast_vector<int>()[0];
        dim_scalar += index_rank_scalar;
        // check the static shape of index
        // if (!index_partial_shape.rank().is_static()) {
        //     // "We support only index with static rank."
        //     add_exception_to_fw_node(index_op, "aten::index_copy_: dynamic rank for index is not supported.");
        //     return false;
        // }
        // check the rank of index
        if (index_rank_scalar != 0 && index_rank_scalar != 1) {
            add_exception_to_fw_node(index_op, "aten::index_copy_: index should have dimension 1 or 0.");
            return false;
        }

        // begin the computation
        if (dim_scalar == 0) {
            // When dim == 0, the op is equavilent to ScatterNDUpdate
            auto input_shape = rg.make<v3::ShapeOf>(input, element::i32);
            auto selected_dim = rg.make<v8::Gather>(input_shape, positive_dim, const_0);
            auto selected_dim_correct_type = rg.make<v1::ConvertLike>(selected_dim, index);
            Output<Node> correct_index = rg.make<v1::Add>(index, selected_dim_correct_type);
            correct_index = rg.make<v1::Mod>(correct_index, selected_dim_correct_type);
            auto unsqueezed_index = rg.make<v0::Unsqueeze>(correct_index, const_neg_1);
            auto result = rg.make<v3::ScatterNDUpdate>(input, unsqueezed_index, tensor);

            copy_runtime_info_and_name(index_op, rg.get(), rt_copy_from);
            replace_node(index_op, result);
            return true;
        } else {
            // auto partial_tensors = rg.make<v1::Split>(tensor, const_0, const_1);
            return false;
        }
    };

    auto m =
        std::make_shared<ov::pass::pattern::Matcher>(index_op, "ov::frontend::pytorch::pass::AtenIndexCopyReplacer");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

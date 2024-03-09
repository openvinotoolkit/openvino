// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_index_copy_replacer.hpp"

#include <memory>
#include <tuple>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
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

std::tuple<Output<Node>, Output<Node>> get_transpose_perm(ov::pass::NodeRegistry& rg,
                                                          const Output<Node>& tensor_rank,
                                                          const Output<Node>& positive_dim,
                                                          const Output<Node>& positive_dim_plus1,
                                                          const std::shared_ptr<v0::Constant> const_0,
                                                          const std::shared_ptr<v0::Constant> const_1,
                                                          const std::shared_ptr<v0::Constant> const_neg_1) {
    // variable preparation
    auto const_0_vec = v0::Constant::create(element::i32, Shape{1}, {0});
    auto const_1_vec = v0::Constant::create(element::i32, Shape{1}, {1});
    auto positive_dim_vec = rg.make<v1::Reshape>(positive_dim, const_1_vec, false);
    auto positive_dim_plus1_vec = rg.make<v1::Reshape>(positive_dim_plus1, const_1_vec, false);

    auto tensor_rank_correct_type = rg.make<v1::ConvertLike>(tensor_rank, positive_dim);
    OutputVector shapes_list;
    shapes_list.push_back(positive_dim_vec);
    auto internal_range = rg.make<v4::Range>(const_0, positive_dim, const_1, element::i32);  // [0]
    auto internal_range_correct_type = rg.make<v1::ConvertLike>(internal_range, positive_dim);
    shapes_list.push_back(internal_range_correct_type);
    Output<Node> perm = rg.make<v0::Concat>(shapes_list, 0);                              // [1, 0]
    auto diff = rg.make<v1::Subtract>(tensor_rank_correct_type, positive_dim_plus1_vec);  // 1

    // add the suffix if exists
    perm = rg.make<v1::Pad>(perm, const_0_vec, diff, const_0, PadMode::CONSTANT);  // [1, 0, 0]
    auto negative_range =
        rg.make<v4::Range>(const_0, rg.make<v0::Negative>(positive_dim_plus1), const_neg_1, element::i32);  // [0, -1]
    Output<Node> negative_range_correct_type = rg.make<v1::ConvertLike>(negative_range, positive_dim);
    negative_range_correct_type =
        rg.make<v1::Pad>(negative_range_correct_type, const_0_vec, diff, const_0, PadMode::CONSTANT);  // [0, -1, 0]
    auto full_range = rg.make<v4::Range>(const_0, tensor_rank_correct_type, const_1, element::i32);    // [0, 1, 2]
    auto full_range_correct_type = rg.make<v1::ConvertLike>(full_range, positive_dim);
    perm = rg.make<v1::Add>(perm, negative_range_correct_type);
    perm = rg.make<v1::Add>(perm, full_range_correct_type);  // [1, 0, 2]

    // compute the reverse perm
    OutputVector reverse_shapes_list;
    auto reverse_prefix = rg.make<v4::Range>(const_1, positive_dim_plus1, const_1, element::i32);  // [1]
    auto reverse_prefix_correct_type = rg.make<v1::ConvertLike>(reverse_prefix, positive_dim);
    reverse_shapes_list.push_back(reverse_prefix_correct_type);
    reverse_shapes_list.push_back(const_0_vec);
    Output<Node> reverse_perm = rg.make<v0::Concat>(reverse_shapes_list, 0);                       // [1, 0]
    reverse_perm = rg.make<v1::Pad>(reverse_perm, const_0_vec, diff, const_0, PadMode::CONSTANT);  // [1, 0, 0]
    reverse_perm = rg.make<v1::Add>(reverse_perm, negative_range_correct_type);
    reverse_perm = rg.make<v1::Add>(reverse_perm, full_range_correct_type);  // [1, 0, 2]

    return std::make_tuple(perm, reverse_perm);
};

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

        /*
         * Here we use an example to express the shape for each step
         * input = torch.ranint([4, 3, 5])
         * dim = 1
         * index = torch.tensor([1, 0])
         * tensor = torch.randint([4, 2, 5])
         */

        auto dim_scalar = std::dynamic_pointer_cast<v0::Constant>(dim.get_node_shared_ptr())->cast_vector<int>()[0];
        auto tensor_shape = rg.make<v3::ShapeOf>(tensor, element::i32);               // [4, 2, 5]
        Output<Node> tensor_rank = rg.make<v3::ShapeOf>(tensor_shape, element::i32);  // [3]
        tensor_rank = rg.make<v8::Gather>(tensor_rank, const_0, const_0);             // 3
        auto tensor_rank_correct_type = rg.make<v1::ConvertLike>(tensor_rank, dim);
        Output<Node> positive_dim = rg.make<v1::Add>(dim, tensor_rank_correct_type);
        positive_dim = rg.make<v1::Mod>(positive_dim, tensor_rank_correct_type);
        Output<Node> positive_dim_plus1 = rg.make<v1::Add>(positive_dim, const_1);

        // get the correct index
        auto input_shape = rg.make<v3::ShapeOf>(input, element::i32);                 // [4, 3, 5]
        auto selected_dim = rg.make<v8::Gather>(input_shape, positive_dim, const_0);  // 3
        auto selected_dim_correct_type = rg.make<v1::ConvertLike>(selected_dim, index);
        Output<Node> correct_index = rg.make<v1::Add>(index, selected_dim_correct_type);
        correct_index = rg.make<v1::Mod>(correct_index, selected_dim_correct_type);
        auto unsqueezed_index = rg.make<v0::Unsqueeze>(correct_index, const_neg_1);  // [[1], [0]]

        // begin the computation
        if (dim_scalar == 0) {
            // When dim == 0, the op is equavilent to ScatterNDUpdate
            auto result = rg.make<v3::ScatterNDUpdate>(input, unsqueezed_index, tensor);

            copy_runtime_info_and_name(index_op, rg.get(), rt_copy_from);
            replace_node(index_op, result);
            return true;
        } else {
            // When dim > 0, we need to get correct tensors to use ScatterNDUpdate
            Output<Node> perm, reverse_perm;
            std::tie(perm, reverse_perm) =
                get_transpose_perm(rg, tensor_rank, positive_dim, positive_dim_plus1, const_0, const_1, const_neg_1);
            auto transposed_tensor = rg.make<v1::Transpose>(tensor, perm);  // [2, 4, 5]
            auto transposed_input = rg.make<v1::Transpose>(input, perm);    // [3, 4, 5]
            auto result =
                rg.make<v3::ScatterNDUpdate>(transposed_input, unsqueezed_index, transposed_tensor);  // [3, 4, 5]
            auto transposed_result = rg.make<v1::Transpose>(result, reverse_perm);                    // [4, 3, 5]

            copy_runtime_info_and_name(index_op, rg.get(), rt_copy_from);
            replace_node(index_op, transposed_result);
            return true;
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

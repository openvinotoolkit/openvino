// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/tensor_array_v3_replacer.hpp"

#include "helper_ops/tensor_array.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::pass;

ov::frontend::tensorflow::pass::TensorArrayV3Replacer::TensorArrayV3Replacer() {
    auto tensor_array_v3 = pattern::wrap_type<TensorArrayV3>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry rg;

        auto tensor_array_v3 = ov::as_type_ptr<TensorArrayV3>(m.get_match_root());
        if (!tensor_array_v3) {
            return false;
        }

        int32_t tensor_element_rank = static_cast<int32_t>(tensor_array_v3->get_element_rank());
        if (tensor_element_rank < 0) {
            return false;
        }

        // retrieve all TensorArrayV3 inputs
        auto size = tensor_array_v3->input_value(0);
        auto element_type = tensor_array_v3->get_element_type();

        // adjust size to have it of shape [1] for further concatenation with element shape
        auto new_size_shape = rg.make<v0::Constant>(element::i32, Shape{1}, 1);
        auto new_size = rg.make<v1::Reshape>(size, new_size_shape, false);

        // create a vector of size element_shape.rank() with ones
        // and compute a shape of initial tensor array [size, 1, ..., 1]
        Output<Node> target_shape;
        if (tensor_element_rank == 0) {
            target_shape = new_size->output(0);
        } else {
            vector<int32_t> ones(tensor_element_rank, 1);
            auto ones_const = rg.make<v0::Constant>(element::i32, Shape{ones.size()}, ones);
            target_shape = rg.make<v0::Concat>(OutputVector{new_size, ones_const}, 0)->output(0);
        }

        // create initial tensor array
        auto scalar_value = make_shared<v0::Constant>(element_type, Shape{}, vector<int32_t>{0});
        auto initial_tensor_array = make_shared<v3::Broadcast>(scalar_value, target_shape);

        // preserve names of the node and the output tensor
        initial_tensor_array->set_friendly_name(tensor_array_v3->get_friendly_name());
        copy_runtime_info(tensor_array_v3, rg.get());

        ov::replace_node(tensor_array_v3,
                         ov::OutputVector{initial_tensor_array->output(0), initial_tensor_array->output(0)});
        return true;
    };

    auto m =
        std::make_shared<pattern::Matcher>(tensor_array_v3, "ov::frontend::tensorflow::pass::TensorArrayV3Replacer");
    register_matcher(m, callback);
}

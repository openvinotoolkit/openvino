// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "index_loop_getitem_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

using namespace ov::pass::pattern;

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::pass;
using namespace ov::op;

IndexLoopGetitemReplacer::IndexLoopGetitemReplacer() {
    auto loop_pattern = pattern::wrap_type<v5::Loop>();
    ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto loop_op = ov::as_type_ptr<v5::Loop>(m.get_match_root());
        bool check_len_input = false;
        if (auto len_reduce = ov::as_type_ptr<v1::ReduceSum>(loop_op->input_value(0).get_node_shared_ptr())) {
            if (auto len_slice = ov::as_type_ptr<v8::Slice>(len_reduce->input_value(0).get_node_shared_ptr())) {
                if (auto len_shape_of = ov::as_type_ptr<v3::ShapeOf>(len_slice->input_value(0).get_node_shared_ptr())) {
                    check_len_input = true;
                }
            }
        }
        if (!check_len_input)
            return false;

        std::shared_ptr<Node> chunk_op;
        size_t chunk_idx = 0;
        auto loop_inputs = loop_op->input_values();
        for (size_t i = 1; i < loop_inputs.size(); i++) {
            if (cast_fw_node(loop_inputs.at(i).get_node_shared_ptr(), "aten::chunk")) {
                chunk_op = loop_inputs.at(i).get_node_shared_ptr();
                chunk_idx = i;
            }
        }
        if (!chunk_op)
            return false;

        auto body = loop_op->get_function();
        size_t chunk_param_idx;
        bool found_param_idx = false;
        for (auto input_desc : loop_op->get_input_descriptions()) {
            if (input_desc->m_input_index == chunk_idx) {
                chunk_param_idx = input_desc->m_body_parameter_index;
                found_param_idx = true;
            }
        }
        if (!found_param_idx)
            return false;

        auto chunk_param = body->get_parameters().at(chunk_param_idx);
        auto param_targets = chunk_param->get_output_target_inputs(0);
        if (param_targets.size() != 1)
            return false;

        auto getitem = param_targets.begin()->get_node()->shared_from_this();
        if (!ov::as_type_ptr<v8::Gather>(getitem))
            return false;

        auto dim = chunk_op->input_value(2);
        if (!ov::as_type_ptr<v0::Constant>(dim.get_node_shared_ptr()))
            return false;

        // connect chunk input directly to loop
        auto chunk_input = chunk_op->input_value(0);
        chunk_op->output(0).replace(chunk_input);
        // len(chunks) is number of iterations
        auto chunks_outside = chunk_op->input_value(1);
        loop_op->input_value(0).replace(chunks_outside);

        auto chunk_counter = getitem->input_value(1);

        auto tensor_0 = v0::Constant::create(element::i32, Shape{1}, {0});
        auto one_1d = v0::Constant::create(element::i32, Shape{1}, {1});

        auto input_shape = std::make_shared<v3::ShapeOf>(chunk_input, element::i32);
        auto input_dimension = std::make_shared<v8::Gather>(input_shape, dim, tensor_0);
        auto init_chunk_size = std::make_shared<v1::Divide>(input_dimension, chunks_outside, true);

        // Add 1 if input is not evenly divisible by chunks
        auto last_chunk_size = std::make_shared<v1::Mod>(input_dimension, chunks_outside);
        auto is_last_nonzero = std::make_shared<v1::Greater>(last_chunk_size, tensor_0);
        auto is_last_nonzero_int = std::make_shared<v0::Convert>(is_last_nonzero, element::i32);
        auto chunk_size = std::make_shared<v1::Add>(init_chunk_size, is_last_nonzero_int);
        auto dim_1d = std::make_shared<v1::Reshape>(dim, one_1d, false);

        // Add new inputs in Loop: chunk_size and dim_1d
        auto inp_descs = loop_op->get_input_descriptions();
        auto chunks_size_body = std::make_shared<v0::Parameter>(element::i32, Shape{1});
        auto dim_body = std::make_shared<v0::Parameter>(dim.get_element_type(), Shape{1});
        body->add_parameters({chunks_size_body, dim_body});
        loop_op->set_argument(loop_op->get_input_size(), chunk_size);
        loop_op->set_argument(loop_op->get_input_size(), dim_1d);
        inp_descs.push_back(std::make_shared<ov::op::util::MultiSubGraphOp::InvariantInputDescription>(
            loop_op->get_input_size() - 2,
            body->get_parameters().size() - 2));
        inp_descs.push_back(std::make_shared<ov::op::util::MultiSubGraphOp::InvariantInputDescription>(
            loop_op->get_input_size() - 1,
            body->get_parameters().size() - 1));
        loop_op->set_input_descriptions(0, inp_descs);

        auto start = std::make_shared<v1::Multiply>(chunk_counter, chunks_size_body);
        auto stop = std::make_shared<v1::Add>(start, chunks_size_body);
        auto curr_chunk = std::make_shared<v8::Slice>(chunk_param, start, stop, one_1d, dim_body);
        replace_node(getitem, curr_chunk);
        copy_runtime_info({chunk_op, getitem},
                          {input_shape,
                           input_dimension,
                           init_chunk_size,
                           last_chunk_size,
                           is_last_nonzero,
                           is_last_nonzero_int,
                           chunk_size,
                           start,
                           stop,
                           dim_1d,
                           curr_chunk});
        curr_chunk->set_friendly_name(getitem->get_friendly_name());
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(loop_pattern, "ov::frontend::pytorch::pass::IndexLoopGetitemReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
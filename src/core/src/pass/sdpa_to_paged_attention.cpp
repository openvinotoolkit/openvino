// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

bool ov::pass::SDPAToPagedAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAToPagedAttention);

    // писать сюда
    // yes sir
    std::cout << "[ STUB ] ov::pass::SDPAToPagedAttention::run_on_model\n";

    //Should I use PartialShape or Shape??
    //Do I need to specify name because in Python it looks like the name argument is ignored completely
    // Maybe I should specify it explicitly
    auto max_context_len = std::make_shared<v0::Parameter>(element::i64, PartialShape{}); // max_context_len
    auto model_remaining_param = {
        std::make_shared<v0::Parameter>(element::boolean, PartialShape{}), // is_prompt
        std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1}),
        max_context_len,
        std::make_shared<v0::Parameter>(element::i64, PartialShape{-1}), // context_lens
        std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1}), // block_tables
    };

    for (auto& parameter : model_remaining_param) {
        parameter->get_output_tensor(0).set_names({parameter->get_friendly_name()});
    }
    // Is it a right way to create a sliding window?
    auto sliding_window = v0::Constant::create(element::i32, Shape{}, {0}); // sliding_window

    auto cur_seq_len = v1::Gather(v3::ShapeOf(model->input("input_ids")).output(0),
                                  v0::Constant::create(element::i32, Shape{}, {1}),
                                  v0::Constant::create(element::i32, Shape{}, {0}));
    cur_seq_len.set_friendly_name("my_curr_seq_len");
    // auto prev_max_seq_len = max_context_len - cur_seq_len; //??

    auto has_parameter = [OV_CAPTURE_CPY_AND_THIS](const std::shared_ptr<ov::Model>& model, const std::string& name) -> bool {
        for (auto& t : model->inputs()) {
            const auto& names = t.get_names();
            if (names.find(name) != names.end()) {
                return true;
            }
        }

        return false;
    };

    if (!has_parameter(model, "position_id")) {
        auto position_ids = std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1}); // position_ids SHOULD I SET NAME?
        position_ids->get_output_tensor(0).set_names({position_ids->get_friendly_name()});
        model->add_parameters({position_ids});
        std::cout << "CREATED A NEW position_ids PARAMETER" << std::endl;
    }

    auto position_ids_input = model->input("position_id"); // Why is it needed?

    return true;
}

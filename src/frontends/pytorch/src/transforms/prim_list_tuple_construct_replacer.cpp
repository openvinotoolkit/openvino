// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prim_list_tuple_construct_replacer.hpp"

#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

namespace {
bool decompose_list_tuple_results(const std::shared_ptr<Model>& model) {
    bool at_least_one_decomposed = false;

    ResultVector results = model->get_results();

    for (size_t i = 0; i < results.size(); ++i) {
        auto result = results[i];
        auto input_node = result->get_input_node_shared_ptr(0);
        auto tuple_construct = cast_fw_node(input_node, "prim::TupleConstruct");
        auto list_construct = cast_fw_node(input_node, "prim::ListConstruct");
        if (!tuple_construct && !list_construct) {
            continue;
        }
        for (const auto& input : input_node->inputs()) {
            const auto& out = input.get_source_output();
            if (const auto& fw_node = cast_fw_node(out.get_node_shared_ptr(), "prim::Constant")) {
                const auto& attrs = fw_node->get_attrs();
                if (attrs.find("none_value") != attrs.end()) {
                    // This is None constant, we skip None if it goes to output of the model. It can be embedding loss
                    // function calculation in model, which used only in training stage. When we move model to eval mode
                    // and does not provide annotation, it is not calculated and return by default None.
                    continue;
                }
            }
            model->add_results({std::make_shared<ov::op::v0::Result>(out)});
        }
        at_least_one_decomposed = true;
        model->remove_result(result);
    }
    return at_least_one_decomposed;
}
};  // namespace

bool DecomposeListTupleResults::run_on_model(const std::shared_ptr<Model>& model) {
    bool at_least_one_decomposed = false;
    bool current_decomposed = false;
    current_decomposed = decompose_list_tuple_results(model);
    if (current_decomposed) {
        at_least_one_decomposed = current_decomposed;
    }
    while (current_decomposed) {
        current_decomposed = decompose_list_tuple_results(model);
    }

    return at_least_one_decomposed;
};
}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

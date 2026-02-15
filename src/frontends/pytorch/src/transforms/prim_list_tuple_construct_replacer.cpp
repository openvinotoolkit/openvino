// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "prim_list_tuple_construct_replacer.hpp"

#include <deque>

#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

namespace {
bool is_index(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), isdigit);
}
}  // namespace

bool DecomposeListTupleResults::run_on_model(const std::shared_ptr<Model>& model) {
    bool at_least_one_decomposed = false;
    const auto& orig_results = model->get_results();
    std::deque<std::shared_ptr<ov::op::v0::Result>> results(orig_results.begin(), orig_results.end());
    ov::ResultVector updated_results;  // will hold final fully unpacked results list

    while (!results.empty()) {
        auto result = results.front();
        results.pop_front();
        auto input_node = result->get_input_node_shared_ptr(0);
        auto tuple_construct = cast_fw_node(input_node, "prim::TupleConstruct");
        auto list_construct = cast_fw_node(input_node, "prim::ListConstruct");
        if (!tuple_construct && !list_construct) {
            updated_results.push_back(result);
            continue;
        }
        const auto& inputs = input_node->inputs();
        // enumerating inputs in reverse order because of results.push_front below
        for (auto pinput = inputs.rbegin(); pinput != inputs.rend(); ++pinput) {
            auto out = pinput->get_source_output();
            if (const auto& fw_node = cast_fw_node(out.get_node_shared_ptr(), "prim::Constant")) {
                const auto& attrs = fw_node->get_attrs();
                if (attrs.find("none_value") != attrs.end()) {
                    // This is None constant, we skip None if it goes to output of the model. It can be embedding loss
                    // function calculation in model, which used only in training stage. When we move model to eval mode
                    // and does not provide annotation, it is not calculated and return by default None.
                    continue;
                }
            }
            auto names = out.get_names();
            out.set_names({});
            for (auto& name : names) {
                if (!is_index(name)) {
                    // Set first found non-index name as output name. If such name exist it will be debug name
                    out.set_names({name});
                    break;
                }
            }
            auto new_result = std::make_shared<ov::op::v0::Result>(out);
            results.push_front(new_result);
            at_least_one_decomposed = true;
        }
    }

    if (at_least_one_decomposed) {
        // remove all results
        while (!model->get_results().empty())
            model->remove_result(model->get_results()[0]);
        // and replace them all by updated list of results
        model->add_results(updated_results);
    }

    return at_least_one_decomposed;
};
}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

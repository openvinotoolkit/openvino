// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/uninitialized_variable_resolve.hpp"

#include "openvino/frontend/tensorflow/hash_table.hpp"
#include "openvino/frontend/tensorflow/variable.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::pass;

ov::frontend::tensorflow::pass::UninitializedVariableResolver::UninitializedVariableResolver() {
    auto unitialized_variable = pattern::wrap_type<ov::frontend::tensorflow::Variable>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry rg;

        auto unitialized_hash_table = dynamic_pointer_cast<ov::frontend::tensorflow::HashTable>(m.get_match_root());
        if (!unitialized_hash_table) {
            return false;
        }

        auto keys = unitialized_hash_table->get_keys();
        auto values = unitialized_hash_table->get_values();

        if (ov::as_type_ptr<HashTable>(keys.get_node_shared_ptr()) ||
            ov::as_type_ptr<HashTable>(values.get_node_shared_ptr())) {
            // keys and values producer is still unitialized variable
            return false;
        }

        rg.add(keys.get_node_shared_ptr());
        rg.add(values.get_node_shared_ptr());

        copy_runtime_info(unitialized_hash_table, rg.get());

        ov::replace_node(unitialized_hash_table, ov::OutputVector{unitialized_hash_table->output(0), keys, values});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(unitialized_variable,
                                                "ov::frontend::tensorflow::pass::UninitializedVariableResolver");
    register_matcher(m, callback);
}

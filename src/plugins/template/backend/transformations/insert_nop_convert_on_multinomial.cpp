// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/insert_nop_convert_on_multinomial.hpp"

#include "openvino/core/constant_fold_utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multinomial.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::runtime::interpreter::pass::InsertNopConvertOnMultinomial::InsertNopConvertOnMultinomial() {
    auto root = ov::pass::pattern::wrap_type<op::v13::Multinomial>();

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto multinomial = as_type_ptr<op::v13::Multinomial>(m.get_match_root());
        if (!multinomial)
            return false;

        auto types = util::unsupported_types();
        if (std::find(types.begin(), types.end(), multinomial->get_input_element_type(0)) == types.end())
            return false;

        ov::pass::NodeRegistry node_registry;
        auto convert =
            std::make_shared<op::v0::Convert>(multinomial->input_value(0), multinomial->get_input_element_type(0));
        convert->get_rt_info()["keep_convert_precision"] = true;
        node_registry.add(convert);
        auto new_multinomial = std::make_shared<op::v13::Multinomial>(convert,
                                                                      multinomial->input_value(1),
                                                                      multinomial->get_convert_type(),
                                                                      multinomial->get_with_replacement(),
                                                                      multinomial->get_log_probs(),
                                                                      multinomial->get_global_seed(),
                                                                      multinomial->get_op_seed());
        new_multinomial->set_friendly_name(multinomial->get_friendly_name());
        node_registry.add(new_multinomial);

        copy_runtime_info(multinomial, node_registry.get());
        replace_node(multinomial, new_multinomial);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(root, "InsertNopConvertOnMultinomial");
    register_matcher(m, callback);
}

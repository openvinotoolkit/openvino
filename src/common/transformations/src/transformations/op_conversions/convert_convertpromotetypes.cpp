// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_convertpromotetypes.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_promote_types.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertConvertPromoteTypes::ConvertConvertPromoteTypes() {
    MATCHER_SCOPE(ConvertConvertPromoteTypes);

    auto has_static_defined_type = [](const Output<Node>& output) -> bool {
        return !pattern::type_matches_any({element::dynamic, element::undefined})(output);
    };
    auto convert_promote_types = pattern::wrap_type<ov::op::v14::ConvertPromoteTypes>(has_static_defined_type);

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto convert_promote_types = ov::as_type_ptr<ov::op::v14::ConvertPromoteTypes>(m.get_match_root());
        if (!convert_promote_types) {
            return false;
        }
        const element::Type& dest_type = convert_promote_types->get_output_element_type(0);
        const auto friendly_name = convert_promote_types->get_friendly_name();
        NodeRegistry node_registry;
        const auto out0 = node_registry.make<ov::op::v0::Convert>(convert_promote_types->input_value(0), dest_type);
        const auto out1 = node_registry.make<ov::op::v0::Convert>(convert_promote_types->input_value(1), dest_type);
        out0->set_friendly_name(convert_promote_types->get_input_node_shared_ptr(0)->get_friendly_name() + "/" +
                                friendly_name + ".0");
        out1->set_friendly_name(convert_promote_types->get_input_node_shared_ptr(1)->get_friendly_name() + "/" +
                                friendly_name + ".1");
        copy_runtime_info(convert_promote_types, node_registry.get());
        replace_node(convert_promote_types, {out0, out1});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(convert_promote_types, matcher_name);
    this->register_matcher(m, callback);
}

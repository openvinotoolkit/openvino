// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_convertaligntypes.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_align_types.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;

ov::pass::ConvertConvertAlignTypes::ConvertConvertAlignTypes() {
    MATCHER_SCOPE(ConvertConvertAlignTypes);

    auto convert_align_types = pattern::wrap_type<ov::op::v14::ConvertAlignTypes>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto convert = std::dynamic_pointer_cast<ov::op::v14::ConvertAlignTypes>(m.get_match_root());
        if (!convert) {
            return false;
        }
        const element::Type& dest_type = convert->get_output_element_type(0);
        if (dest_type == element::dynamic || dest_type == element::undefined) {
            return false;
        }
        auto out0 = std::make_shared<ov::op::v0::Convert>(convert->input_value(0), dest_type);
        auto out1 = std::make_shared<ov::op::v0::Convert>(convert->input_value(1), dest_type);
        out0->set_friendly_name(convert->get_friendly_name() + ".0");
        out1->set_friendly_name(convert->get_friendly_name() + ".1");
        copy_runtime_info(convert, {out0, out1});
        replace_node(convert, {out0, out1});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(convert_align_types, matcher_name);
    this->register_matcher(m, callback);
}

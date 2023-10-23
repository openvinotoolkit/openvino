// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "align_types_removal.hpp"

#include <memory>
#include <utility>

#include "helper_ops/align_types.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

AlignTypesRemoval::AlignTypesRemoval() {
    auto align_types_pattern = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto align_types = std::dynamic_pointer_cast<AlignTypes>(m.get_match_root());
        if (!align_types)
            return false;
        auto lhs_itype = align_types->get_input_element_type(0);
        auto rhs_itype = align_types->get_input_element_type(1);
        auto lhs_otype = align_types->get_output_element_type(0);
        auto rhs_otype = align_types->get_output_element_type(1);
        if (lhs_otype.is_static() && rhs_otype.is_static()) {
            auto out1 = align_types->input_value(0);
            auto out2 = align_types->input_value(1);
            if (lhs_itype != lhs_otype)
                out1 = std::make_shared<v0::Convert>(align_types->input_value(0), lhs_otype);
            if (rhs_itype != rhs_otype)
                out2 = std::make_shared<v0::Convert>(align_types->input_value(1), rhs_otype);
            align_types->output(0).replace(out1);
            align_types->output(1).replace(out2);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(align_types_pattern,
                                                          "ov::frontend::pytorch::pass::AlignTypesRemoval");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

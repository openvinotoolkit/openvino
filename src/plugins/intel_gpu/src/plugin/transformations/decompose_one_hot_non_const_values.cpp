// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decompose_one_hot_non_const_values.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/util/one_hot_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu {

DecomposeOneHotNonConstValues::DecomposeOneHotNonConstValues() {
    using namespace ov::pass::pattern;

    // OneHotBase is the parent of both v1::OneHot and v16::OneHot, so this matches either version.
    auto one_hot_m = wrap_type<ov::op::util::OneHotBase>();

    ov::matcher_pass_callback callback = [this](Matcher& m) {
        auto one_hot = ov::as_type_ptr<ov::op::util::OneHotBase>(m.get_match_root());
        if (!one_hot || transformation_callback(one_hot))
            return false;

        const auto on_value = one_hot->input_value(2);
        const auto off_value = one_hot->input_value(3);

        // Compile time values are baked into the kernel by the one_hot primitive, leave them alone.
        if (ov::is_type<ov::op::v0::Constant>(on_value.get_node()) &&
            ov::is_type<ov::op::v0::Constant>(off_value.get_node()))
            return false;

        auto mask_on = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, true);
        auto mask_off = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, false);
        // Cloning keeps the axis and, for v16, the negative indices mode.
        auto mask = one_hot->clone_with_new_inputs({one_hot->input_value(0),
                                                    one_hot->input_value(1),
                                                    mask_on,
                                                    mask_off});
        mask->set_friendly_name(one_hot->get_friendly_name() + "/mask");

        // on/off are scalars by the op specification, so the Select keeps the OneHot shape and type.
        auto select = std::make_shared<ov::op::v1::Select>(mask, on_value, off_value);
        select->set_friendly_name(one_hot->get_friendly_name());

        ov::copy_runtime_info(one_hot, {mask_on, mask_off, mask, select});
        ov::replace_node(one_hot, select);
        return true;
    };

    auto m = std::make_shared<Matcher>(one_hot_m, "DecomposeOneHotNonConstValues");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu

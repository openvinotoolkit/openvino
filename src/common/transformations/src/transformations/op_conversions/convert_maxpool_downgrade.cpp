// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;

pass::ConvertMaxPool8ToMaxPool1::ConvertMaxPool8ToMaxPool1() {
    MATCHER_SCOPE(ConvertMaxPool8ToMaxPool1);

    auto maxpool_v8_pattern = pattern::wrap_type<ov::op::v8::MaxPool>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto maxpool_v8_node = std::dynamic_pointer_cast<ov::op::v8::MaxPool>(m.get_match_root());

        if (!maxpool_v8_node || maxpool_v8_node->get_output_target_inputs(1).size() != 0)
            return false;

        for (auto dilation : maxpool_v8_node->get_dilations())
            if (dilation != 1)
                return false;

        auto maxpool_v1_node = make_shared<ov::op::v1::MaxPool>(maxpool_v8_node->input_value(0),
                                                                maxpool_v8_node->get_strides(),
                                                                maxpool_v8_node->get_pads_begin(),
                                                                maxpool_v8_node->get_pads_end(),
                                                                maxpool_v8_node->get_kernel(),
                                                                maxpool_v8_node->get_rounding_type(),
                                                                maxpool_v8_node->get_auto_pad());

        auto out_name = ov::op::util::create_ie_output_name(maxpool_v8_node->output(0));

        maxpool_v1_node->set_friendly_name(maxpool_v8_node->get_friendly_name());
        maxpool_v8_node->output(0).replace(maxpool_v1_node->output(0));
        ov::copy_runtime_info(maxpool_v8_node, maxpool_v1_node);
        maxpool_v8_node->clear_control_dependencies();

        OPENVINO_SUPPRESS_DEPRECATED_START
        ov::descriptor::set_ov_tensor_legacy_name(maxpool_v1_node->output(0).get_tensor(), out_name);
        OPENVINO_SUPPRESS_DEPRECATED_END

        return true;
    };

    auto m = make_shared<pattern::Matcher>(maxpool_v8_pattern, matcher_name);
    register_matcher(m, callback);
}

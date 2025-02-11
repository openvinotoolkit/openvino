// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

pass::ConvertMulticlassNmsToMulticlassNmsIE::ConvertMulticlassNmsToMulticlassNmsIE(bool force_i32_output_type) {
    MATCHER_SCOPE(ConvertMulticlassNmsToMulticlassNmsIE);
    auto nms = pattern::wrap_type<op::util::MulticlassNmsBase>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto nms = ov::as_type_ptr<op::util::MulticlassNmsBase>(m.get_match_root());
        if (!nms || transformation_callback(nms)) {
            return false;
        }

        const auto new_args = nms->input_values();

        // if input shape is dynamic force the output shape must be dynamic too
        if (nms->get_input_partial_shape(0).is_dynamic() || nms->get_input_partial_shape(1).is_dynamic()) {
            return false;
        }
        if (new_args.size() > 2) {
            if (nms->get_input_partial_shape(2).is_dynamic()) {
                return false;
            }
        }

        // vector of new openvino operations
        NodeVector new_ops;
        auto attrs = nms->get_attrs();
        attrs.output_type = force_i32_output_type ? element::i32 : attrs.output_type;

        std::shared_ptr<op::internal::MulticlassNmsIEInternal> nms_new;
        if (new_args.size() > 2) {
            nms_new = std::make_shared<op::internal::MulticlassNmsIEInternal>(new_args.at(0),
                                                                              new_args.at(1),
                                                                              new_args.at(2),
                                                                              attrs);
        } else {
            nms_new = std::make_shared<op::internal::MulticlassNmsIEInternal>(new_args.at(0), new_args.at(1), attrs);
        }
        new_ops.emplace_back(nms_new);

        Output<Node> output_0 = nms_new->output(0);
        Output<Node> output_1 = nms_new->output(1);
        Output<Node> output_2 = nms_new->output(2);

        if (nms->output(1).get_element_type() != output_1.get_element_type()) {
            output_1 = std::make_shared<ov::op::v0::Convert>(output_1, nms->output(1).get_element_type());
            new_ops.emplace_back(output_1.get_node_shared_ptr());
        }

        if (nms->output(2).get_element_type() != output_2.get_element_type()) {
            output_2 = std::make_shared<ov::op::v0::Convert>(output_2, nms->output(2).get_element_type());
            new_ops.emplace_back(output_2.get_node_shared_ptr());
        }

        nms_new->set_friendly_name(nms->get_friendly_name());
        copy_runtime_info(nms, new_ops);
        replace_node(nms, {output_0, output_1, output_2});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}

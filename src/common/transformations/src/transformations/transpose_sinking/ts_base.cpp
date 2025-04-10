// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_base.hpp"

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

void TSForwardBase::transpose_sinking(const std::string& pass_name,
                                      const TSForwardBase::sinking_function& sinking_transformation) {
    ov::matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto main_node = pattern_to_output.at(m_pattern).get_node_shared_ptr();
        utils::TransposeInputsInfo transpose_input_info =
            utils::GetFirstTransposeInput(main_node, m_transpose_indices, m_if_transpose_sinkable);

        if (transformation_callback(main_node)) {
            mark_as_no_sinking_node(transpose_input_info.transpose);
            return false;
        }

        bool res;
        if (sinking_transformation) {
            // use custom function to sink transpose
            res = sinking_transformation(main_node, transpose_input_info);
        } else {
            // default transpose sinking function:
            res = default_inputs_update(main_node, transpose_input_info);
            if (res) {
                default_outputs_update(main_node, transpose_input_info);
            }
        }
        if (!res) {
            mark_as_no_sinking_node(transpose_input_info.transpose);
        }
        return res;
    };

    auto m = std::make_shared<pattern::Matcher>(m_pattern, pass_name);
    register_matcher(m, matcher_pass_callback);
}

bool TSForwardBase::default_inputs_update(const std::shared_ptr<Node>& main_node,
                                          const TransposeInputsInfo& transpose_info) {
    return utils::sink_forward::UpdateInputTransposes(main_node, transpose_info);
}

void TSForwardBase::default_outputs_update(const std::shared_ptr<Node>& main_node,
                                           const TransposeInputsInfo& transpose_info) {
    main_node->validate_and_infer_types();
    for (auto& new_node : utils::sink_forward::InsertOutputTransposes(main_node, transpose_info)) {
        register_new_node(new_node);
        mark_as_no_sinking_node(new_node);
    }
}

bool TSForwardBase::if_node_has_transpose_inputs(
    const Output<Node>& output,
    const std::vector<size_t>& transpose_indices,
    const std::function<bool(const std::shared_ptr<ov::op::v1::Transpose>& transpose,
                             const std::shared_ptr<ov::op::v0::Constant>& transpose_order)>& if_transpose_sinkable) {
    utils::TransposeInputsInfo inputs_info =
        utils::GetFirstTransposeInput(output.get_node_shared_ptr(), transpose_indices, if_transpose_sinkable);
    return !inputs_info.isEmpty();
}

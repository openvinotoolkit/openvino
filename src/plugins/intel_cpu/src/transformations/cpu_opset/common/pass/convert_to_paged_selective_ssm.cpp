// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_paged_selective_ssm.hpp"

#include <memory>

#include "nodes/paged_selective_ssm.h"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"

namespace ov::intel_cpu {

ConvertToPagedSelectiveSSM::ConvertToPagedSelectiveSSM() {
    MATCHER_SCOPE(ConvertToPagedSelectiveSSM);

    const auto paged_ssm_pattern = ov::pass::pattern::wrap_type<ov::op::internal::PagedSelectiveSSM>();
    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& matcher) {
        const auto paged_ssm = ov::as_type_ptr<ov::op::internal::PagedSelectiveSSM>(matcher.get_match_root());
        if (!paged_ssm || std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(paged_ssm)) {
            return false;
        }

        const auto data_precision = paged_ssm->get_input_element_type(input_port_index(PagedSelectiveSSMInputPort::A));
        const auto state_precision =
            paged_ssm->get_input_element_type(input_port_index(PagedSelectiveSSMInputPort::State));
        if (data_precision == state_precision ||
            (state_precision != ov::element::f32 && state_precision != ov::element::f16 &&
             state_precision != ov::element::bf16)) {
            return false;
        }

        ov::element::TypeVector validation_input_types(paged_ssm_input_count, ov::element::dynamic);
        validation_input_types[input_port_index(PagedSelectiveSSMInputPort::State)] = data_precision;
        ov::element::TypeVector output_types(paged_ssm_output_count, paged_ssm->get_output_element_type(0));
        const auto replacement =
            std::make_shared<ov::op::TypeRelaxed<ov::op::internal::PagedSelectiveSSM>>(*paged_ssm,
                                                                                       validation_input_types,
                                                                                       output_types);
        replacement->set_friendly_name(paged_ssm->get_friendly_name());
        ov::copy_runtime_info(paged_ssm, replacement);
        ov::replace_node(paged_ssm, replacement);
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(paged_ssm_pattern, matcher_name), callback);
}

}  // namespace ov::intel_cpu

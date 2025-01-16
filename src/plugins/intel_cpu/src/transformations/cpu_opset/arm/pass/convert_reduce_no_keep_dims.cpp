// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_reduce_no_keep_dims.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset8.hpp"

template <class T>
ov::matcher_pass_callback ov::intel_cpu::ConvertReduceNoKeepDimsBase::convert_reduce() {
    return [&](ov::pass::pattern::Matcher& m) {
        auto reduce = std::dynamic_pointer_cast<T>(m.get_match_root());
        if (!reduce || reduce->get_keep_dims()) {
            return false;
        }

        reduce->set_keep_dims(true);
        const auto reduce_new = reduce->clone_with_new_inputs({reduce->input_value(0), reduce->input_value(1)});
        std::shared_ptr<ov::Node> squeeze = std::make_shared<ov::op::v0::Squeeze>(reduce_new, reduce->input_value(1));
        squeeze->set_friendly_name(reduce_new->get_friendly_name());
        ov::copy_runtime_info(reduce, {reduce_new, squeeze});
        ov::replace_node(reduce, squeeze);

        return true;
    };
}

template <typename ReductionType>
ov::intel_cpu::ConvertReduction<ReductionType>::ConvertReduction() {
    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        ov::pass::pattern::wrap_type<ReductionType>({ov::pass::pattern::any_input(),
                                                     ov::pass::pattern::wrap_type<ov::opset8::Constant>()}), "ConvertReduction");
     register_matcher(m, convert_reduce<ReductionType>());
}

template class ov::intel_cpu::ConvertReduction<ov::op::util::LogicalReductionKeepDims>;
template class ov::intel_cpu::ConvertReduction<ov::op::util::ArithmeticReductionKeepDims>;

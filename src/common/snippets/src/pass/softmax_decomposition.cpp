// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/softmax_decomposition.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::pass {
using namespace lowered;

SoftmaxDecomposition::SoftmaxDecomposition() {
    MATCHER_SCOPE(SoftmaxDecomposition);
    auto softmax_v1_m = ov::pass::pattern::wrap_type<ov::op::v1::Softmax>();
    auto softmax_v8_m = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>();
    auto softmax_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{softmax_v1_m, softmax_v8_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::SoftmaxDecomposition")
        auto softmax = m.get_match_root();

        const auto& pshape = softmax->get_input_partial_shape(0);
        OPENVINO_ASSERT(!pshape.rank().is_dynamic(), "SoftmaxDecomposition doesn't support dynamic ranks");
        const auto rank = pshape.size();

        size_t axis = 0;
        if (const auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(softmax)) {
            axis = ov::util::try_normalize_axis(softmax_v8->get_axis(), rank, *softmax);
        } else if (const auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(softmax)) {
            axis = softmax_v1->get_axis();
        } else {
            OPENVINO_THROW("Unexpected node matched");
        }

        const auto& softmax_input = softmax->input_value(0);
        const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(softmax_input, axis);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_max);
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(softmax_input, reduce_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);

        const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, axis);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_sum);
        const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.F);
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);

        OPENVINO_ASSERT(axis < rank, "Softmax has incorrect axis");
        std::vector<size_t> subtensor(rank, 1);
        for (size_t i = axis; i < rank; ++i) {
            subtensor[i] = utils::get_full_dim_value();
        }

        PortDescriptorUtils::set_port_descriptor(power->input(0), subtensor);
        PortDescriptorUtils::set_port_descriptor(power->output(0), std::move(subtensor));

        copy_runtime_info(softmax, {reduce_max, subtract, exp, reduce_sum, power, multiply});
        return ov::replace_node_update_name(softmax, multiply);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(softmax_m, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::snippets::pass

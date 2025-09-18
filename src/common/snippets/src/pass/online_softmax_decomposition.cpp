// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/online_softmax_decomposition.hpp"

#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/online_softmax.hpp"
#include "snippets/op/online_softmax_update_max.hpp"
#include "snippets/op/online_softmax_update_sum.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/reduce.hpp"

namespace ov::snippets::pass {

OnlineSoftmaxDecomposition::OnlineSoftmaxDecomposition() {
    MATCHER_SCOPE(OnlineSoftmaxDecomposition);

    using namespace ov::pass::pattern;
    const auto online_softmax_m = wrap_type<ov::snippets::op::OnlineSoftmax>({any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::OnlineSoftmaxDecomposition")
        const auto& online_softmax = ov::as_type_ptr<ov::snippets::op::OnlineSoftmax>(m.get_match_root());

        OPENVINO_ASSERT(online_softmax->get_input_element_type(0).is_real(),
                        "OnlineSoftmaxDecomposition currently only works for real data type");

        const auto& pshape = online_softmax->get_input_partial_shape(0);
        OPENVINO_ASSERT(!pshape.rank().is_dynamic(), "OnlineSoftmaxDecomposition doesn't support dynamic ranks");
        const auto& rank = pshape.size();
        OPENVINO_ASSERT(rank != 0, "OnlineSoftmaxDecomposition doesn't support input tesnor with rank smaller than 1");

        const auto& axis = rank - 1;

        const auto& softmax_input = online_softmax->input_value(0);
        const auto& reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(softmax_input, axis);

        // input is max_local, first output is max_current, second ouput is (max_past - max_current)
        const auto& updated_max = std::make_shared<ov::snippets::op::OnlineSoftmaxUpdateMax>(reduce_max);

        // sum_local
        const auto& subtract = std::make_shared<ov::op::v1::Subtract>(softmax_input, updated_max->output(0));
        const auto& exp = std::make_shared<ov::op::v0::Exp>(subtract);
        const auto& reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, axis);

        // std::exp(max_past - max)
        const auto& coeff_exp = std::make_shared<ov::op::v0::Exp>(updated_max->output(1));
        // sum_current = std::exp(max_past - max) * sum_past + sum_local
        // output 0 is sum_current.
        // output 1 is coeff(std::exp(max_past - max) * sum_past) to second brgemm, not used in single online softmax.
        const auto& updated_sum = std::make_shared<ov::snippets::op::OnlineSoftmaxUpdateSum>(reduce_sum, coeff_exp);
        const auto& power = std::make_shared<ov::snippets::op::PowerStatic>(updated_sum->output(0), -1.F);
        const auto& multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);

        // This is coeff for second brgemm in flash attention scenario (it is applied on the K loops exception
        // the first iteration).
        // It is not used in single online softmax scenario.
        // std::exp(max_past - max) * sum_past / sum_current
        const auto& brgemm_coeff = std::make_shared<ov::op::v1::Divide>(updated_sum->output(1), updated_sum->output(0));

        copy_runtime_info(online_softmax,
                          {reduce_max,
                           updated_max,
                           subtract,
                           exp,
                           reduce_sum,
                           coeff_exp,
                           updated_sum,
                           power,
                           multiply,
                           brgemm_coeff});
        multiply->set_friendly_name(online_softmax->get_friendly_name());
        online_softmax->output(0).replace(multiply->output(0));
        online_softmax->output(1).replace(brgemm_coeff->output(0));
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(online_softmax_m, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::snippets::pass
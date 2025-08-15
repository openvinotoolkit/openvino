// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/fa_decomposition.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::pass {
using namespace lowered;

FADecomposition::FADecomposition() {
    MATCHER_SCOPE(FADecomposition);

    using namespace ov::pass::pattern;
    auto single_consumer_f32 = [](const ov::Output<ov::Node>& out) {
        return consumers_count(1)(out) && type_matches(ov::element::f32)(out);
    };
    const auto brgemm0_m = wrap_type<ov::snippets::op::Brgemm>(single_consumer_f32);
    const auto softmax_m = wrap_type<ov::op::v1::Softmax, ov::op::v8::Softmax>({brgemm0_m}, single_consumer_f32);
    const auto brgemm1_m = wrap_type<ov::snippets::op::Brgemm>({softmax_m, any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::FADecomposition")

        auto& pm = m.get_pattern_value_map();
        const auto brgemm0 = as_type_ptr<ov::snippets::op::Brgemm>(pm.at(brgemm0_m).get_node_shared_ptr());
        const auto softmax = pm.at(softmax_m).get_node_shared_ptr();
        const auto brgemm1 = as_type_ptr<ov::snippets::op::Brgemm>(pm.at(brgemm1_m).get_node_shared_ptr());

        const auto& q_shape = brgemm0->get_input_partial_shape(0);
        OPENVINO_ASSERT(!q_shape.rank().is_dynamic(), "FADecomposition doesn't support dynamic ranks");
        const auto& rank = q_shape.size();

        // extend to dynmaic
        if (!q_shape.is_static()) {
            return false;
        }
        auto buffer_shape = q_shape.get_shape();
        buffer_shape[rank - 1] = 1;

        const auto max_buffer = std::make_shared<snippets::op::Buffer>(buffer_shape, ov::element::f32);
        const auto sum_buffer = std::make_shared<snippets::op::Buffer>(buffer_shape, ov::element::f32);
        const auto past_max_buffer = std::make_shared<snippets::op::Buffer>(buffer_shape, ov::element::f32);
        const auto past_sum_buffer = std::make_shared<snippets::op::Buffer>(buffer_shape, ov::element::f32);

        size_t axis;
        if (const auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(softmax)) {
            axis = ov::util::try_normalize_axis(softmax_v8->get_axis(), rank, *softmax);
        } else if (const auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(softmax)) {
            axis = softmax_v1->get_axis();
        } else {
            OPENVINO_THROW("Unexpected node matched");
        }
        if (axis != static_cast<size_t>(-1) && axis != rank - 1) {
            return false;
        }

        const auto& softmax_input = softmax->input_value(0);
        // local max
        const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(softmax_input, axis);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_max);
        // update max
        const auto new_max = std::make_shared<ov::op::v1::Maximum>(past_max_buffer, reduce_max);
        OutputVector max_buf_args({new_max});
        max_buffer->set_arguments({max_buf_args});

        // local sum
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(softmax_input, new_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);
        const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, axis);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_sum);
        // update sum
        const auto sub_coef = std::make_shared<ov::op::v1::Subtract>(past_max_buffer, new_max);
        const auto exp_coef = std::make_shared<ov::op::v0::Exp>(sub_coef);
        const auto mul_coef = std::make_shared<ov::op::v1::Multiply>(exp_coef, past_sum_buffer);
        const auto new_sum = std::make_shared<ov::op::v1::Add>(mul_coef, reduce_sum);
        OutputVector sum_buf_args({new_sum});
        sum_buffer->set_arguments({sum_buf_args});

        // softmax final result
        const auto power = std::make_shared<ov::snippets::op::PowerStatic>(new_sum, -1.f);
        const auto softmax_out = std::make_shared<ov::op::v1::Multiply>(exp, power);

        // calibration coef
        const auto coef = std::make_shared<ov::op::v1::Divide>(mul_coef, new_sum);

        // const auto brgemm1_new = std::make_shared<op::Brgemm>(softmax_out, brgemm1->input_value(1));
        const auto brgemm1_new = std::make_shared<op::Brgemm>(softmax_out, brgemm1->input_value(1), coef);

        // remove softmax
        copy_runtime_info(softmax, {reduce_max, subtract, exp, reduce_sum, power, softmax_out});
        softmax->output(0).replace(softmax->input_value(0));
        // replace brgemm1
        copy_runtime_info(brgemm1, brgemm1_new);
        return ov::replace_node_update_name(brgemm1, brgemm1_new);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(brgemm1_m, matcher_name);
    register_matcher(m, callback);

}

}  // namespace ov::snippets::pass
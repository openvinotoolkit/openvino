// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/tensor_iterator_transformations/apply_transformations_to_ti_body.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/specialize_function.hpp>

void ngraph::pass::ApplyTransformationsToTIBody::apply_transformations_to_ti_body(ngraph::pass::Manager& manager) {
    auto tensor_iterator = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32,
                                                                        ngraph::Shape{}, ngraph::pattern::has_class<ngraph::opset3::TensorIterator>());

    ngraph::graph_rewrite_callback callback = [this, &manager](pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset3::TensorIterator>(m.get_match_root());
        if (!ti) {
            return false;
        }

        auto body = ti->get_body();
        const auto function = std::make_shared<ngraph::Function>(body->get_results(),
                                                                 ngraph::ParameterVector{body->get_parameters()});

        manager.run_passes(function);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "ApplyTransformationsToTIBody");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
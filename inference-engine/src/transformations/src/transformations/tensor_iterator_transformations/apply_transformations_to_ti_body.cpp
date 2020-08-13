// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/tensor_iterator_transformations/apply_transformations_to_ti_body.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::pass::ApplyTransformationsToTIBody::ApplyTransformationsToTIBody(ngraph::pass::Manager & manager) : MatcherPass() {
    auto tensor_iterator = ngraph::pattern::wrap_type<ngraph::opset4::TensorIterator>();

    ngraph::matcher_pass_callback callback = [this, &manager](pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator>(m.get_match_root());
        if (!ti) {
            return false;
        }

        manager.run_passes(ti->get_body()->to_function());
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "ApplyTransformationsToTIBody");
    register_matcher(m, callback);
}

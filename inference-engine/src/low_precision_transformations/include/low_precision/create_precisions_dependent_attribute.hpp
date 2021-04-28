// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

template <typename AttributeType, typename OperationType>
class TRANSFORMATIONS_API CreatePrecisionsDependentAttribute;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

template <typename AttributeType, typename OperationType>
class ngraph::pass::low_precision::CreatePrecisionsDependentAttribute : public ngraph::pass::MatcherPass {
public:
    CreatePrecisionsDependentAttribute() {
        auto operation = pattern::wrap_type<OperationType>();

        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto op = m.get_match_root();
            if (!op || transformation_callback(op)) {
                return false;
            }

            auto& rt = op->get_rt_info();

            const auto precisionPreservedAttribute = std::make_shared<ngraph::VariantWrapper<PrecisionPreservedAttribute>>(PrecisionPreservedAttribute(false));
            rt[ngraph::VariantWrapper<PrecisionPreservedAttribute>::type_info.name] = precisionPreservedAttribute;

            const auto& sharedValue = precisionPreservedAttribute->get().sharedValue;
            const auto attribute = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>(
                std::make_shared<AttributeType>(sharedValue));
            rtInfo[ngraph::VariantWrapper<std::shared_ptr<AttributeType>>::type_info.name] = attribute;

            return true;
        };

        auto matcher = std::make_shared<ngraph::pattern::Matcher>(operation, "CreateAttribute");
        this->register_matcher(matcher, callback);
    }
};

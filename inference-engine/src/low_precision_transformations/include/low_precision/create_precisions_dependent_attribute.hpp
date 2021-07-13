// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "rt_info/precision_preserved_attribute.hpp"
#include "network_helper.hpp"
#include "lpt_itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

template <typename AttributeType, typename OperationType>
class CreatePrecisionsDependentAttribute;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

template <typename AttributeType, typename OperationType>
class ngraph::pass::low_precision::CreatePrecisionsDependentAttribute : public ngraph::pass::MatcherPass {
public:
    CreatePrecisionsDependentAttribute() {
        auto operation = pattern::wrap_type<OperationType>();

        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto node = m.get_match_root();
            if (transformation_callback(node)) {
                return false;
            }

            {
                OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "CreatePrecisionsDependentAttribute");
                auto &rt = node->get_rt_info();

                const auto precisionPreservedAttribute = std::make_shared<ngraph::VariantWrapper<PrecisionPreservedAttributePtr>>(
                    std::make_shared<PrecisionPreservedAttribute>(false));
                rt[ngraph::VariantWrapper<PrecisionPreservedAttributePtr>::type_info.name] = precisionPreservedAttribute;
                const auto &targetSharedValue = precisionPreservedAttribute->get()->sharedValue;

                const auto attribute = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>(
                    std::make_shared<AttributeType>());
                rt[ngraph::VariantWrapper<std::shared_ptr<AttributeType>>::type_info.name] = attribute;

                ngraph::pass::low_precision::NetworkHelper::reassign<PrecisionPreservedSharedValue, PrecisionPreservedAttribute>(
                    targetSharedValue,
                    {
                        std::dynamic_pointer_cast<PrecisionPreservedAttribute>(attribute->get()),
                        std::dynamic_pointer_cast<PrecisionPreservedAttribute>(precisionPreservedAttribute->get())
                    });
            }
            return true;
        };

        auto matcher = std::make_shared<ngraph::pattern::Matcher>(operation, "CreatePrecisionsDependentAttribute");
        this->register_matcher(matcher, callback);
    }
};

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/lpt_visibility.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/opsets/opset1.hpp"
#include "rt_info/precision_preserved_attribute.hpp"
#include "network_helper.hpp"
#include "lpt_itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

template <typename AttributeType, typename OperationType>
class CreatePrecisionsDependentAttribute;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief CreatePrecisionsDependentAttribute transformation marks OperationType operations by
 * PrecisionPreservedAttribute and AttributeType attributes with the same shared part.
 *
 * For more details about the transformation, refer to
 * [CreatePrecisionsDependentAttribute](@ref openvino_docs_OV_UG_lpt_CreatePrecisionsDependentAttribute) page
 * in the OpenVINO Developer Guide.
 */
template <typename AttributeType, typename OperationType>
class ov::pass::low_precision::CreatePrecisionsDependentAttribute : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("low_precision::CreatePrecisionsDependentAttribute");
    CreatePrecisionsDependentAttribute() {
        auto operation = pattern::wrap_type<OperationType>();

        ov::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto node = m.get_match_root();
            if (transformation_callback(node)) {
                return false;
            }

            {
                OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "CreatePrecisionsDependentAttribute");
                auto &rt = node->get_rt_info();

                // The goal is definition if an operation precision preserved or not. As result here we should make 3 steps:
                // Step #1: create PrecisionPreservedAttribute instance obviously,
                // which will be used as result (will be used for future precision propagation)
                const auto precisionPreservedAttribute = PrecisionPreservedAttribute(false);
                rt[PrecisionPreservedAttribute::get_type_info_static()] = precisionPreservedAttribute;
                const auto &targetSharedValue = precisionPreservedAttribute.attribute->sharedValue;

                // Step #2: create AttributeType attribute instance for OperationType operation to propagate the instance
                const auto attribute = AttributeType{};
                rt[AttributeType::get_type_info_static()] = attribute;

                // Step #3: assign the same shared value to enable PrecisionPreservedAttribute update during AttributeType propagation
                ov::pass::low_precision::NetworkHelper::reassign<AttributeType>(
                    targetSharedValue,
                    {
                        attribute.attribute,
                        precisionPreservedAttribute.attribute
                    });
            }
            return true;
        };

        auto matcher = std::make_shared<ov::pass::pattern::Matcher>(operation, "CreatePrecisionsDependentAttribute");
        this->register_matcher(matcher, callback);
    }
};

// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "low_precision/lpt_visibility.hpp"
#include "low_precision/base_matcher_pass.hpp"
#include "low_precision/lpt_itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

template <typename AttributeType, typename OperationType>
class CreateAttribute;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

enum class AttributeSource {
    Node,
    OutputPort
};

/**
 * @ingroup ov_transformation_common_api
 * @brief CreateAttribute transformation marks OperationType operations by AttributeType attribute.
 *
 * For more details about the transformation, refer to
 * [CreateAttribute](@ref openvino_docs_OV_UG_lpt_CreateAttribute) page in the OpenVINO Developer Guide.
 */
template <typename AttributeType, typename OperationType = ov::pass::pattern::op::Label>
class ov::pass::low_precision::CreateAttribute : public ov::pass::low_precision::BaseMatcherPass {
public:
    CreateAttribute(const AttributeParameters& params = AttributeParameters(), const AttributeSource source = AttributeSource::Node) : BaseMatcherPass(params) {
        assert((source == AttributeSource::Node) || (source == AttributeSource::OutputPort));
        auto operation = std::is_same<OperationType, pattern::op::Label>::value ?
            pattern::any_input() :
            pattern::wrap_type<OperationType>();

        ov::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto op = m.get_match_root();
            if (transformation_callback(op)) {
                return false;
            }
            {
                OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "CreateAttribute");
                const auto attribute = AttributeType::create(op, this->params);
                if (attribute.empty()) {
                    return false;
                }
            }
            return true;
        };

        auto matcher = std::make_shared<ov::pass::pattern::Matcher>(operation, "CreateAttribute");
        this->register_matcher(matcher, callback);
    }
};

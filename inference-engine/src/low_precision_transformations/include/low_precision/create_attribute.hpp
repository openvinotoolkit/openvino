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
class TRANSFORMATIONS_API CreateAttribute;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

template <typename AttributeType, typename OperationType>
class ngraph::pass::low_precision::CreateAttribute : public ngraph::pass::MatcherPass {
public:
    typedef enum CreateAttributeSource {
        Node,
        OutputPort
    } Source;

    CreateAttribute(const Source source = Source::Node) {
        assert((source == Source::Node) || (source == Source::OutputPort));
        auto operation = pattern::wrap_type<OperationType>();

        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto op = m.get_match_root();
            assert((source == Source::Node) || ((source == Source::OutputPort) && (op->get_output_size() == 1ul)));

            if (!op || transformation_callback(op)) {
                return false;
            }

            auto attribute = make_shared_attribute<AttributeType>();
            auto attributeWrapper = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>(attribute);

            auto& rt = source == Source::Node ? op->get_rt_info() : op->output(0).get_rt_info();
            rt[ngraph::VariantWrapper<std::shared_ptr<AttributeType>>::type_info.name] = attributeWrapper;

            return true;
        };

        auto matcher = std::make_shared<ngraph::pattern::Matcher>(operation, "CreateAttribute");
        this->register_matcher(matcher, callback);
    }
};

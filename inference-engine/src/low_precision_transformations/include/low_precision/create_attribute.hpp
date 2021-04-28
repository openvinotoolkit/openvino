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

// Attribute::init(Node&) <= can we reuse?
enum class AttributeSource {
    Node,
    OutputPort
};

template <typename AttributeType, typename OperationType>
class ngraph::pass::low_precision::CreateAttribute : public ngraph::pass::MatcherPass {
public:
    CreateAttribute(const AttributeSource source = AttributeSource::Node) {
        assert((source == AttributeSource::Node) || (source == AttributeSource::OutputPort));
        auto operation = pattern::wrap_type<OperationType>();

        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto op = m.get_match_root();
            //assert((source == AttributeSource::Node) || ((source == AttributeSource::OutputPort) && (op->get_output_size() == 1ul)));
            if ((source == AttributeSource::OutputPort) && (op->get_output_size() != 1ul)) {
                std::cout << "CreateAttribute" << std::endl;
            }

            if (!op || transformation_callback(op)) {
                return false;
            }

            auto attribute = make_shared_attribute<AttributeType>();
            auto attributeWrapper = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<AttributeType>>>(attribute);

            auto& rt = source == AttributeSource::Node ? op->get_rt_info() : op->output(0).get_rt_info();
            rt[ngraph::VariantWrapper<std::shared_ptr<AttributeType>>::type_info.name] = attributeWrapper;

            return true;
        };

        auto matcher = std::make_shared<ngraph::pattern::Matcher>(operation, "CreateAttribute");
        this->register_matcher(matcher, callback);
    }
};

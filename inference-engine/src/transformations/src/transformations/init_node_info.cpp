// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::InitNodeInfo, "InitNodeInfo", 0);

bool ngraph::pass::InitNodeInfo::run_on_function(std::shared_ptr<ngraph::Function> f) {
    std::vector<std::shared_ptr<Variant> > attributes {
        std::make_shared<VariantWrapper<FusedNames> >(FusedNames())
    };

    using VariantCreator = std::function<std::shared_ptr<Variant>(const std::string&)>;
    std::map<std::string, VariantCreator> update_attributes {
            {"PrimitivesPriority",
                [](const std::string & value) -> std::shared_ptr<Variant> {
                    return std::make_shared<VariantWrapper<PrimitivesPriority> >(PrimitivesPriority(value));
                }
            }
    };

    for (auto & node : f->get_ops()) {
        auto & rtInfo = node->get_rt_info();
        // Default attributes initialization
        for (auto & attr : attributes) {
            // Skip initialization if attribute has been already set
            if (rtInfo.count(attr->get_type_info().name)) continue;
            if (auto init_attr = attr->init(node)) {
                rtInfo[attr->get_type_info().name] = init_attr;
            }
        }
        // Convert manually set attributes to appropriate VariantWraper class instances
        // all manually set attributes must belong to VariantWrapper<std::string> class
        for (auto & attr : update_attributes) {
            if (rtInfo.count(attr.first)) {
                if (auto variant_string = std::dynamic_pointer_cast<VariantWrapper<std::string> >(rtInfo[attr.first])) {
                    rtInfo.erase(attr.first);
                    auto res = attr.second(variant_string->get());
                    rtInfo[res->get_type_info().name] = res;
                }
            }
        }
    }
    return false;
}

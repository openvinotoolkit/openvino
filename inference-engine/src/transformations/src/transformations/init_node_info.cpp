// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

bool ngraph::pass::InitNodeInfo::run_on_function(std::shared_ptr<ngraph::Function> f) {
    std::vector<std::shared_ptr<Variant> > attributes {
        std::make_shared<VariantWrapper<FusedNames> >(FusedNames())
    };

    for (auto & node : f->get_ops()) {
        auto & rtInfo = node->get_rt_info();
        for (auto & attr : attributes) {
            // Skip initialization if attribute has been already set
            if (rtInfo.count(attr->get_type_info().name)) continue;
            if (auto init_attr = attr->init(node)) {
                rtInfo[attr->get_type_info().name] = init_attr;
            }
        }
    }
    return false;
}

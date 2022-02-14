// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/vector_to_scalar.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::ReplaceLoadsWithScalarLoads::ReplaceLoadsWithScalarLoads() {
    MATCHER_SCOPE(ReplaceLoadsWithScalarLoads);
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::snippets::op::Load>()),
            [this](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ReplaceLoadsWithScalarLoads_callback")
            auto root = m.get_match_root();
            if (transformation_callback(root))
                return false;
            auto load = std::make_shared<ngraph::snippets::op::ScalarLoad> (root->input_value(0));
            load->set_friendly_name(root->get_friendly_name());
            ngraph::copy_runtime_info(root, load);
            ngraph::replace_node(root, load);
            return true;
        });
}

ngraph::snippets::pass::ReplaceStoresWithScalarStores::ReplaceStoresWithScalarStores() {
    MATCHER_SCOPE(ReplaceStoresWithScalarStores);
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::snippets::op::Store>()),
            [this](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ReplaceStoresWithScalarStores_callback")
            auto root = m.get_match_root();
            if (transformation_callback(root))
                return false;
            auto store = std::make_shared<ngraph::snippets::op::ScalarStore> (root->input_value(0));
            store->set_friendly_name(root->get_friendly_name());
            ngraph::copy_runtime_info(root, store);
            ngraph::replace_node(root, store);
            return true;
        });
}

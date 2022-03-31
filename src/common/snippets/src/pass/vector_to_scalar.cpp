// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/vector_to_scalar.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::SetScalarCountForLoad::SetScalarCountForLoad() {
    MATCHER_SCOPE(SetScalarCountForLoad);
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::snippets::op::Load>()),
            [this](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::SetScalarCountForLoad_callback")
            auto root = m.get_match_root();
            if (transformation_callback(root))
                return false;

            const auto load = ov::as_type_ptr<ngraph::snippets::op::Load>(root);
            if (!load)
                return false;

            load->set_count(1lu);
            return true;
        });
}

ngraph::snippets::pass::SetScalarCountForStore::SetScalarCountForStore() {
    MATCHER_SCOPE(SetScalarCountForStore);
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::snippets::op::Store>()),
            [this](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::SetScalarCountForStore_callback")
            auto root = m.get_match_root();
            if (transformation_callback(root))
                return false;

            const auto store = ov::as_type_ptr<ngraph::snippets::op::Store>(root);
            if (!store)
                return false;

            store->set_count(1lu);
            return true;
        });
}

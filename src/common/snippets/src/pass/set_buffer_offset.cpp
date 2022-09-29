// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "snippets/snippets_isa.hpp"
#include "snippets/pass/set_buffer_offset.hpp"
#include "snippets/op/subgraph.hpp"


ngraph::snippets::pass::SetBufferOffset::SetBufferOffset() {
    MATCHER_SCOPE(SetBufferOffset);
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<op::Buffer>(), matcher_name),
        [&](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::SetBufferOffset")
        auto root = m.get_match_root();
        const auto buffer = ov::as_type_ptr<op::Buffer>(root);
        buffer->set_offset(current_offset);
        current_offset += ngraph::shape_size(buffer->get_shape()) * buffer->get_element_type().size();
        return true;
    });
}

// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "snippets/snippets_isa.hpp"
#include "snippets/pass/convert_constants.hpp"
#include "snippets/op/subgraph.hpp"


ngraph::snippets::pass::ConvertConstantsToScalars::ConvertConstantsToScalars() {
    MATCHER_SCOPE(ConvertConstantsToScalars);
    auto constants = std::make_shared<pattern::op::Label>(pattern::any_input(),
                                                    [](std::shared_ptr<Node> n) {
                                                        return ngraph::is_type<ov::op::v0::Constant>(n);
                                                    });
    ngraph::graph_rewrite_callback callback = [this](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ConvertConstantsToScalars")
        auto constant = as_type_ptr<ov::op::v0::Constant>(m.get_match_root());
        if (ov::shape_size(constant->get_output_shape(0)) != 1)
            return false;
        //  Note that all Constants {1,1,1,1} are converted to Scalar {1} here
        //  This is needed to simplify shape inference, otherwise {1,1,1,1} Constants can increase output rank
        //  Also some operations support only scalar shapes, so we need separate scalars and shape [1]
        const auto shape = constant->get_output_shape(0).size() == 0 ? ov::Shape{} : ov::Shape{1};
        auto scalar = std::make_shared<snippets::op::Scalar>(ov::op::v0::Constant(*constant, shape));
        scalar->set_friendly_name(constant->get_friendly_name());
        ngraph::copy_runtime_info(constant, scalar);
        ngraph::replace_node(constant, scalar);
        return true;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(constants), callback);
}

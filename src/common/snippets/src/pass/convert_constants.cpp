// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/convert_constants.hpp"

#include "openvino/core/rt_info.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/op/subgraph.hpp"

ov::snippets::pass::ConvertConstantsToScalars::ConvertConstantsToScalars() {
    MATCHER_SCOPE(ConvertConstantsToScalars);
    auto constants =
        std::make_shared<ov::pass::pattern::op::Label>(ov::pass::pattern::any_input(), [](std::shared_ptr<Node> n) {
            return ov::is_type<ov::op::v0::Constant>(n);
        });
    ov::graph_rewrite_callback callback = [](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::ConvertConstantsToScalars")
        auto constant = as_type_ptr<ov::op::v0::Constant>(m.get_match_root());
        if (ov::shape_size(constant->get_output_partial_shape(0).to_shape()) != 1)
            return false;
        //  Note that all Constants {1,1,1,1} are converted to Scalar {1} here
        //  This is needed to simplify shape inference, otherwise {1,1,1,1} Constants can increase output rank
        //  Also some operations support only scalar shapes, so we need separate scalars and shape [1]
        auto scalar = std::make_shared<snippets::op::Scalar>(ov::op::v0::Constant(*constant, ov::Shape{1}));
        scalar->set_friendly_name(constant->get_friendly_name());
        ov::copy_runtime_info(constant, scalar);
        ov::replace_node(constant, scalar);
        return true;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(constants, matcher_name), callback);
}

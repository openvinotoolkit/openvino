// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/pass/reduce_to_snippets_reduce.hpp"
#include "snippets/op/reduce.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "openvino/opsets/opset1.hpp"
#include "openvino/core/rt_info.hpp"

namespace ov {

snippets::pass::ReduceToSnippetsReduce::ReduceToSnippetsReduce() {
    MATCHER_SCOPE(ReduceToSnippetsReduce);
    auto reduce_pattern = ov::pass::pattern::wrap_type<ov::op::v1::ReduceSum, ov::op::v1::ReduceMax>();

    auto callback = [](ov::pass::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::ReduceToSnippetsReduce")
        auto root = m.get_match_root();
        const auto& reduce_base = ov::as_type_ptr<const ov::op::util::ArithmeticReductionKeepDims>(root);
        OPENVINO_ASSERT(reduce_base, "Failed to cast Reduce operation to ArithmeticReductionKeepDims");
        const auto& axis_constant = ov::as_type_ptr<const ov::op::v0::Constant>(root->get_input_node_shared_ptr(1));
        // Note: we do not check the Constant value here. If the Reduce was tokenized, then we assume that it is supported
        OPENVINO_ASSERT(reduce_base->get_keep_dims() && axis_constant, "Unspported Reduce was tokenized by Snippets");
        const auto& data_input = root->get_input_source_output(0);
        const auto axis_value = axis_constant->cast_vector<int32_t>(1)[0];
        std::shared_ptr<snippets::op::ReduceBase> snippets_reduce = nullptr;
        if (ov::is_type<ov::op::v1::ReduceSum>(root))
            snippets_reduce = std::make_shared<snippets::op::ReduceSum>(data_input, axis_value);
        else
            snippets_reduce = std::make_shared<snippets::op::ReduceMax>(data_input, axis_value);

        replace_node(root, snippets_reduce);
        snippets_reduce->set_friendly_name(root->get_friendly_name());
        copy_runtime_info(root, snippets_reduce);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reduce_pattern, matcher_name);
    register_matcher(m, callback);
}

} // namespace ov
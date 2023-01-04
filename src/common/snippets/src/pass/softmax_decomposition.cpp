// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/remarks.hpp"
#include <snippets/itt.hpp>

#include "snippets/pass/softmax_decomposition.hpp"
#include "snippets/pass/reset_buffer.hpp"
#include "snippets/pass/insert_loops.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/validation_util.hpp>

namespace ngraph {
namespace snippets {
pass::SoftmaxDecomposition::SoftmaxDecomposition(const size_t vector_size, const int32_t buffer_allocation_rank) {
    MATCHER_SCOPE(SoftmaxDecomposition);

    auto m_softmax = ngraph::pattern::wrap_type<ngraph::op::v1::Softmax, ngraph::op::v8::Softmax>();

    auto callback = [=](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::SoftmaxDecomposition")
        auto root = m.get_match_root();
        const auto master_pshape = root->get_input_partial_shape(0);
        const auto rank = master_pshape.rank();
        if (rank.is_dynamic() || master_pshape.is_dynamic())
            return false;

        int64_t axis = 0;
        if (const auto softmax_v8 = ngraph::as_type_ptr<const ov::op::v8::Softmax>(root)) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            axis = ngraph::normalize_axis(root->get_friendly_name(), softmax_v8->get_axis(), rank);
            OPENVINO_SUPPRESS_DEPRECATED_END
        } else if (const auto& softmax_v1 = ngraph::as_type_ptr<const ov::op::v1::Softmax>(root)) {
            axis = static_cast<int64_t>(softmax_v1->get_axis());
        } else {
            return false;
        }

        const auto shape_rank = rank.get_length();
        if (axis != shape_rank - 1)
            return false;

        const auto& load = std::make_shared<op::Load>(root->get_input_source_output(0), vector_size);
        const auto& softmax = std::make_shared<snippets::op::Softmax>(load, axis);
        ngraph::copy_runtime_info(root, softmax);
        const auto& store = std::make_shared<op::Store>(softmax, vector_size);

        const std::vector<size_t> tensor = root->get_input_shape(0);
        const std::vector<size_t> subtensor {1, tensor.back()};
        TensorDescriptor td(tensor, subtensor);
        set_tensor_descriptor_ptr(root->get_input_source_output(0), std::make_shared<TensorDescriptor>(td));
        set_tensor_descriptor_ptr(load, std::make_shared<TensorDescriptor>(td));
        set_tensor_descriptor_ptr(softmax, std::make_shared<TensorDescriptor>(td));
        ngraph::replace_node(root, store);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_softmax, matcher_name);
    register_matcher(m, callback);
}
} // namespace snippets
} // namespace ngraph
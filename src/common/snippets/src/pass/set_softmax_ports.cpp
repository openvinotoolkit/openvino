// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/set_softmax_ports.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"

#include "openvino/op/softmax.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/core/validation_util.hpp"


ov::snippets::pass::SetSoftmaxPorts::SetSoftmaxPorts() {
    MATCHER_SCOPE(SetSoftmaxPorts);

    auto m_softmax_v1 = ov::pass::pattern::wrap_type<ov::op::v1::Softmax>();
    auto m_softmax_v8 = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>();
    auto m_softmax = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{m_softmax_v1, m_softmax_v8});

    auto callback = [](ov::pass::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::SetSoftmaxPorts")
        auto root = m.get_match_root();

        const auto& pshape = root->get_input_partial_shape(0);

        OPENVINO_ASSERT(!pshape.rank().is_dynamic(), "SetSoftmaxPorts doesn't support dynamic ranks");
        const auto rank =  pshape.rank().get_length();

        int64_t axis;
        if (const auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(root)) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            axis = ov::normalize_axis(root->get_friendly_name(), softmax_v8->get_axis(), rank);
            OPENVINO_SUPPRESS_DEPRECATED_END
        } else if (const auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(root)) {
            axis = softmax_v1->get_axis();
        } else {
            return false;
        }

        OPENVINO_ASSERT(axis < static_cast<int64_t>(rank), "Softmax has incorrect axis");
        std::vector<size_t> subtensor(rank, 1);
        for (auto i = axis; i < rank; ++i)
            subtensor[i] = lowered::PortDescriptor::ServiceDimensions::FULL_DIM;

        lowered::PortDescriptorUtils::set_port_descriptor_ptr(root->input(0), std::make_shared<lowered::PortDescriptor>(root->input(0), subtensor));
        lowered::PortDescriptorUtils::set_port_descriptor_ptr(root->output(0), std::make_shared<lowered::PortDescriptor>(root->output(0), subtensor));

        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_softmax, matcher_name), callback);
}

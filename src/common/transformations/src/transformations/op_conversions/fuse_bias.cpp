// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/fuse_bias.hpp"

#include <algorithm>
#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/convolution_biased.hpp"
#include "ov_ops/fully_connected.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::FuseBias::FuseBias() {
    MATCHER_SCOPE(FuseBias);
    using namespace ov::pass::pattern;

    auto data_batch = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto filters = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto m_conv =
        ov::pass::pattern::wrap_type<ov::op::internal::ConvolutionBiased>({data_batch, filters},
                                                                          ov::pass::pattern::consumers_count(1));
    auto m_bias = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_conv, m_bias});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();

        auto conv =
            ov::as_type_ptr<ov::op::internal::ConvolutionBiased>(pattern_to_output[m_conv].get_node_shared_ptr());
        if (!conv || transformation_callback(conv)) {
            return false;
        }

        const auto& bias = pattern_to_output[m_bias].get_node_shared_ptr();
        if (!bias || transformation_callback(bias)) {
            return false;
        }

        auto add = ov::as_type_ptr<ov::op::v1::Add>(pattern_to_output[m_add].get_node_shared_ptr());
        if (!add || transformation_callback(add)) {
            return false;
        }

        const ov::PartialShape& output_shape = conv->get_output_partial_shape(0);
        auto rank = output_shape.size();
        if (rank == 0) {
            return false;
        }
        ov::NodeVector new_ops;

        std::shared_ptr<ov::Node> final_bias = bias;
        auto add_shape = add->get_output_partial_shape(0);

        if (add_shape.rank().is_dynamic()) {
            return false;
        }

        if (add_shape.size() >= 2) {
            auto reshape_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
            final_bias = ov::op::util::make_try_fold<ov::op::v1::Reshape>(final_bias, reshape_const, true);
            new_ops.push_back(final_bias);
        }

        auto new_conv = std::make_shared<ov::op::internal::ConvolutionBiased>(conv->input_value(0),
                                                                              conv->input_value(1),
                                                                              final_bias,
                                                                              conv->get_strides(),
                                                                              conv->get_pads_begin(),
                                                                              conv->get_pads_end(),
                                                                              conv->get_dilations(),
                                                                              conv->get_auto_pad());

        new_ops.push_back(new_conv);

        new_conv->set_friendly_name(add->get_friendly_name());
        ov::copy_runtime_info({conv, add}, new_ops);
        ov::replace_node(add, new_conv);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_add, "FuseBias");
    this->register_matcher(m, callback);
}

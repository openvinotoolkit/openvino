// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/constant_folding.hpp"
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <transformations/utils/utils.hpp>
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"

#include "split_fc_m.hpp"

#include "itt.hpp"

static size_t weightsThreshold() {
    static size_t result = std::getenv("SPLIT_THRESHOLD") ? static_cast<size_t>(std::stoi(std::getenv("SPLIT_THRESHOLD"))) : 6600000;
    return result;
}

ov::intel_cpu::SplitFCbyM::SplitFCbyM(int sub_stream_num) {
    MATCHER_SCOPE(SplitFCbyM);
    auto fc_m = ov::pass::pattern::wrap_type<ov::intel_cpu::FullyConnectedNode>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& fc_node = pattern_map.at(fc_m).get_node_shared_ptr();
        auto& rt_info = fc_node->get_rt_info();
        if (rt_info.count("split")) {
            return false;
        }

        const auto src_item = fc_node->get_input_node_shared_ptr(0);
        const auto fc_weight_node = fc_node->get_input_node_shared_ptr(1);

        // TODO: support transpose
        if (ov::is_type<ov::op::v1::Transpose>(fc_weight_node)) {
            return false;
        }

        // needn't to split fc when the dim is 0.
        const auto& wgt_shape = fc_weight_node->get_shape();
        const size_t split_dim = wgt_shape.size() - 1;
        // split happens on the second dimension.
        auto split_dim_node = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, split_dim);

        // std::cout << "SplitFCbyM transformation check" << "\n";
        // weight shape size 660000 is a trade-off value, which is summarized and verified by LLMs.
        if (wgt_shape[split_dim] <= 1 || ov::shape_size(wgt_shape) < weightsThreshold()) {
            return false;
        }

        // parts will be splited according the sub stream num.
        int split_num = sub_stream_num + 1;

        // auto split_parts = [](int len, int n) {
        //     int average = len / n;
        //     std::vector<int> parts(n, average);
        //     parts.back() = len - average * (n - 1);
        //     return parts;
        // };

        const PartialShape shape = fc_node->input(0).get_partial_shape();
        auto rank = shape.get_max_shape().size();
        // @todo check if static
        const Dimension K = shape[rank - 1];
        Dimension K_split_dim = K / split_num;

        std::vector<Dimension> split_fc_dims;
        for (size_t i = 0; i < rank; i++) {
            split_fc_dims.push_back(shape[i]);
        }
        split_fc_dims.push_back(K_split_dim);
        const PartialShape split_fc_shape(split_fc_dims);

        // const auto K_split = K_split_dim.get_length();
        std::vector<std::shared_ptr<Node>> fc_node_vec(split_num);
        for (int i = 0; i < split_num; i++) {
            // int64_t offset = K_split * (i + 1);
            fc_node_vec[i] = std::make_shared<ov::intel_cpu::FullyConnectedNode>(
                src_item,
                fc_weight_node,
                fc_node->get_output_partial_shape(0).rank(),
                ov::element::undefined);
                // activation_stride,
                // activation_offset);

            fc_node_vec[i]->set_friendly_name(fc_node->get_friendly_name() + "_split_" + std::to_string(i));
            fc_node_vec[i]->get_rt_info()["split_part"] = true;
            if (i > 0)
                fc_node_vec[i]->get_rt_info()["other_split"] = true;
        }
        fc_node_vec[0]->get_rt_info()["main_split_root"] = true;
        fc_node_vec[0]->get_rt_info()["duplicates"] = fc_node_vec;

        // @todo currently works only for split = 2.
        // split > 2 requires a chain of Add nodes
        auto add_node = std::make_shared<ov::op::v1::Add>(fc_node_vec[0], fc_node_vec[1]);
        add_node->get_rt_info()["sync_point"] = true;

        ov::replace_node_update_name(fc_node, add_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_m, matcher_name);
    this->register_matcher(m, callback);
}

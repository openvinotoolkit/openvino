// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_reduce_multi_axis.hpp"

#include <ngraph/rt_info.hpp>

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>

template <class T>
ngraph::matcher_pass_callback ov::intel_cpu::ConvertReduceMultiAxisBase::convert_reduce() {
    return [&](ngraph::pattern::Matcher& m) {
        auto reduce = m.get_match_root();
        if (!std::dynamic_pointer_cast<T>(reduce)) {
            return false;
        }

        const auto& input0 = reduce->input_value(0);
        const auto& input1 = reduce->input_value(1);
        const auto& data_shape0 = input0.get_partial_shape();
        const auto& data_shape1 = input1.get_partial_shape();
        if (data_shape0.is_dynamic() ||
            data_shape1.is_dynamic()) {
            return false;
        }

        if (ngraph::shape_size(input1.get_shape()) <= 1) {
            return false;
        }
        auto reduction_axes = std::dynamic_pointer_cast<ov::opset8::Constant>(input1.get_node_shared_ptr());
        if (!reduction_axes) {
            return false;
        }
        auto axes = reduction_axes->cast_vector<int64_t>();
        ngraph::NodeVector new_ops;
        std::shared_ptr<ngraph::Node> node = input0.get_node_shared_ptr();
        for (auto axis : axes) {
            auto reduction_axis = ov::opset8::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{}, {axis});
            node = std::make_shared<T>(node, reduction_axis, true);
            new_ops.push_back(node);
        }

        auto out_shape = reduce->get_output_shape(0);
        auto dst_shape = std::make_shared<ov::opset8::Constant>(ngraph::element::i64, ngraph::Shape{out_shape.size()},
                    std::vector<int64_t>(out_shape.begin(), out_shape.end()));
        auto reshape = std::make_shared<ov::opset8::Reshape>(node, dst_shape, true);

        reshape->set_friendly_name(reduce->get_friendly_name());
        ngraph::copy_runtime_info(reduce, new_ops);
        ngraph::replace_node(reduce, reshape);
        return true;
    };
}

ov::intel_cpu::ConvertReduceProd::ConvertReduceProd() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::opset8::ReduceProd>({ngraph::pattern::any_input(),
                                                           ngraph::pattern::wrap_type<ov::opset8::Constant>()}), "ConvertReduceProd");
    register_matcher(m, convert_reduce<ov::opset8::ReduceProd>());
}

ov::intel_cpu::ConvertReduceMin::ConvertReduceMin() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::opset8::ReduceMin>({ngraph::pattern::any_input(),
                                                          ngraph::pattern::wrap_type<ov::opset8::Constant>()}), "ConvertReduceMin");
    register_matcher(m, convert_reduce<ov::opset8::ReduceMin>());
}

ov::intel_cpu::ConvertReduceMax::ConvertReduceMax() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::opset8::ReduceMax>({ngraph::pattern::any_input(),
                                                          ngraph::pattern::wrap_type<ov::opset8::Constant>()}), "ConvertReduceMax");
    register_matcher(m, convert_reduce<ov::opset8::ReduceMax>());
}

ov::intel_cpu::ConvertReduceSum::ConvertReduceSum() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::opset8::ReduceSum>({ngraph::pattern::any_input(),
                                                          ngraph::pattern::wrap_type<ov::opset8::Constant>()}), "ConvertReduceSum");
    register_matcher(m, convert_reduce<ov::opset8::ReduceSum>());
}

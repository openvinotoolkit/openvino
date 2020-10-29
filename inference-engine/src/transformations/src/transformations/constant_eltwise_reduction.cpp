// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/constant_eltwise_reduction.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

template <typename T>
std::shared_ptr<ngraph::op::Constant> constant_reduction(const std::shared_ptr<ngraph::op::Constant>& const_node) {
    std::vector<T> data = const_node->get_vector<T>();
    // TODO: implement this function after eltwise broadcast support will be added
    return nullptr;
}

ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
    // Check that eltwise operation either Add or Multiply
    auto eltwise_node = m.get_match_root();
    if (!std::dynamic_pointer_cast<ngraph::opset1::Add>(eltwise_node) &&
        !std::dynamic_pointer_cast<ngraph::opset1::Multiply>(eltwise_node)) {
        return false;
    }

    for (const auto& input : eltwise_node->get_inputs()) {
        const auto& inputLayer = input.get_output().get_node();
        auto const_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(inputLayer);
        if (!const_node) continue;

        std::shared_ptr<ngraph::opset1::Constant> result = nullptr;

        ngraph::element::Type elem_type = const_node->get_element_type();
        switch (elem_type) {
        case ngraph::element::Type_t::f32:
            result = constant_reduction<float>(const_node);
            break;
        case ngraph::element::Type_t::i16:
        case ngraph::element::Type_t::f16:
            result = constant_reduction<short>(const_node);
            break;
        case ngraph::element::Type_t::u8:
            result = constant_reduction<uint8_t>(const_node);
            break;
        case ngraph::element::Type_t::i8:
            result = constant_reduction<int8_t>(const_node);
            break;
        case ngraph::element::Type_t::i32:
            result = constant_reduction<int32_t>(const_node);
            break;
        default:
            return false;
        }
        if (result) {
            ngraph::replace_node(inputLayer, std::dynamic_pointer_cast<ngraph::Node>(result));
            std::cout << "Successful constant_eltwise reduction" << std::endl;
        }
    }

    return true;
};

void ngraph::pass::ConstantEltwiseReduction::constant_multiply_reduction() {
    Shape shape {2, 2, 1, 1};
    auto constant1 = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto constant2 = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto eltwise = std::make_shared<ngraph::opset1::Multiply>(constant1, constant2);

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, "CPUFusion.ConstantMultiplyReduction");
    this->add_matcher(m, callback);
}

void ngraph::pass::ConstantEltwiseReduction::constant_add_reduction() {
    Shape shape {2, 2, 1, 1};
    auto constant1 = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto constant2 = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto eltwise = std::make_shared<ngraph::opset1::Add>(constant1, constant2);

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, "CPUFusion.ConstantAddReduction");
    this->add_matcher(m, callback);
}

// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeConversion(const ngraph::Output<Node>& in,
                                             const element::Type& output_type,
                                             const ngraph::helpers::ConversionTypes& conversionType) {
    if (conversionType == ngraph::helpers::ConversionTypes::CONVERT) {
        if (in.get_element_type() == ov::element::f32) {
            auto add_const = std::make_shared<ngraph::opset1::Constant>(ov::element::f32, Shape{1},
                                                                        std::vector<float>{0.8});
            auto add = std::make_shared<ngraph::opset1::Add>(in, add_const);
            return std::make_shared<ngraph::opset1::Convert>(add, output_type);
        } else if (in.get_element_type() == ov::element::f16) {
            auto add_const = std::make_shared<ngraph::opset1::Constant>(ov::element::f16, Shape{1},
                                                                        std::vector<float>{0.8});
            auto add = std::make_shared<ngraph::opset1::Add>(in, add_const);
            return std::make_shared<ngraph::opset1::Convert>(add, output_type);
        } else if (in.get_element_type() == ov::element::f64) {
            auto add_const = std::make_shared<ngraph::opset1::Constant>(ov::element::f64, Shape{1},
                                                                        std::vector<float>{0.8});
            auto add = std::make_shared<ngraph::opset1::Add>(in, add_const);
            return std::make_shared<ngraph::opset1::Convert>(add, output_type);
        } else if (in.get_element_type() == ov::element::bf16) {
            auto add_const = std::make_shared<ngraph::opset1::Constant>(ov::element::bf16, Shape{1},
                                                                        std::vector<float>{0.8});
            auto add = std::make_shared<ngraph::opset1::Add>(in, add_const);
            return std::make_shared<ngraph::opset1::Convert>(add, output_type);
        } else {
            return std::make_shared<ngraph::opset1::Convert>(in, output_type);
        }
    } else if (conversionType == ngraph::helpers::ConversionTypes::CONVERT_LIKE) {
        const auto like = std::make_shared<op::Constant>(output_type, ngraph::Shape{1});
        return std::make_shared<ngraph::opset1::ConvertLike>(in, like);
    } else {
        throw std::runtime_error("Incorrect type of Conversion operation");
    }
}

}  // namespace builder
}  // namespace ngraph

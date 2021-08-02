// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeConvertion(const ngraph::Output<Node>& in,
                                             const element::Type& output_type,
                                             const ngraph::helpers::ConvertionTypes& convertionType) {
    if (convertionType == ngraph::helpers::ConvertionTypes::CONVERT) {
        return std::make_shared<ngraph::opset1::Convert>(in, output_type);
    } else if (convertionType == ngraph::helpers::ConvertionTypes::CONVERT_LIKE) {
        const auto like = std::make_shared<op::Constant>(output_type, ngraph::Shape{1});
        return std::make_shared<ngraph::opset1::ConvertLike>(in, like);
    } else {
        throw std::runtime_error("Incorrect type of Convertion operation");
    }
}

}  // namespace builder
}  // namespace ngraph

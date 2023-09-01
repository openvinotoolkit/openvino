// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/opsets/opset1.hpp>
#include "ov_models/utils/ov_helpers.hpp"

namespace ov {
namespace builder {

std::shared_ptr<ov::Node> makeConversion(const ov::Output<Node>& in,
                                             const element::Type& output_type,
                                             const ov::helpers::ConversionTypes& conversionType) {
    if (conversionType == ov::helpers::ConversionTypes::CONVERT) {
        return std::make_shared<ov::opset1::Convert>(in, output_type);
    } else if (conversionType == ov::helpers::ConversionTypes::CONVERT_LIKE) {
        const auto like = std::make_shared<op::v0::Constant>(output_type, ov::Shape{1});
        return std::make_shared<ov::opset1::ConvertLike>(in, like);
    } else {
        throw std::runtime_error("Incorrect type of Conversion operation");
    }
}

}  // namespace builder
}  // namespace ov

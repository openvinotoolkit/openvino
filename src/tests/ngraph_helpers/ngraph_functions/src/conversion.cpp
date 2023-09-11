// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "common_test_utils/test_enums.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeConversion(const ov::Output<Node>& in,
                                         const element::Type& output_type,
                                         const ov::test::utils::ConversionTypes& conversionType) {
    if (conversionType == ov::test::utils::ConversionTypes::CONVERT) {
        return std::make_shared<ov::op::v0::Convert>(in, output_type);
    } else if (conversionType == ov::test::utils::ConversionTypes::CONVERT_LIKE) {
        const auto like = std::make_shared<op::Constant>(output_type, ov::Shape{1});
        return std::make_shared<ov::op::v1::ConvertLike>(in, like);
    } else {
        throw std::runtime_error("Incorrect type of Conversion operation");
    }
}

}  // namespace builder
}  // namespace ngraph

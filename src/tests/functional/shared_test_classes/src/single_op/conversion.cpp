// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/conversion.hpp"

namespace ov {
namespace test {
namespace {
std::map<ov::test::utils::ConversionTypes, std::string> conversionNames = {
    {ov::test::utils::ConversionTypes::CONVERT, "Convert"},
    {ov::test::utils::ConversionTypes::CONVERT_LIKE, "ConvertLike"}};
}

std::string ConversionLayerTest::getTestCaseName(const testing::TestParamInfo<ConversionParamsTuple>& obj) {
    ov::test::utils::ConversionTypes conversion_type;
    ov::element::Type input_type, convert_type;
    std::string device_name;
    std::vector<InputShape> shapes;
    std::tie(conversion_type, shapes, input_type, convert_type, device_name) =
        obj.param;
    std::ostringstream result;
    result << "conversionOpType=" << conversionNames[conversion_type] << "_";
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "inputPRC=" << input_type.get_type_name() << "_";
    result << "targetPRC=" << convert_type.get_type_name() << "_";
    result << "trgDev=" << device_name;
    return result.str();
}

void ConversionLayerTest::SetUp() {
    ov::test::utils::ConversionTypes conversion_type;
    ov::element::Type input_type, convert_type;
    std::vector<InputShape> shapes;
    std::tie(conversion_type, shapes, input_type, convert_type, targetDevice) = GetParam();
    init_input_shapes(shapes);

    ov::ParameterVector params;
    for (const auto& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(input_type, shape));
    }

    std::shared_ptr<ov::Node> conversion;
    if (conversion_type == ov::test::utils::ConversionTypes::CONVERT) {
        conversion = std::make_shared<ov::op::v0::Convert>(params.front(), convert_type);
    } else /*CONVERT_LIKE*/ {
        auto like = std::make_shared<ov::op::v0::Constant>(convert_type, ov::Shape{1});
        conversion = std::make_shared<ov::op::v1::ConvertLike>(params.front(), like);
    }

    auto result = std::make_shared<ov::op::v0::Result>(conversion);
    function = std::make_shared<ov::Model>(result, params, "Conversion");
}
}  // namespace test
}  // namespace ov

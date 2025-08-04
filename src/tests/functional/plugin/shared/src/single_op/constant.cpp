// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/constant.hpp"

namespace ov {
namespace test {
namespace {
template <size_t N>
std::vector<std::string> getElements(const std::vector<std::string>& v) {
    const auto new_size = std::min(N, v.size());
    return {begin(v), std::next(begin(v), new_size)};
}
}  // namespace

std::string ConstantLayerTest::getTestCaseName(
    const testing::TestParamInfo<constantParamsTuple>& obj) {
    ov::Shape shape;
    ov::element::Type model_type;
    std::vector<std::string> data_elements;
    std::string target_device;

    std::tie(shape, model_type, data_elements, target_device) = obj.param;

    std::ostringstream result;
    result << "TS={" << ov::test::utils::vec2str(shape) << "}_";
    result << "dataPRC=" << model_type.get_type_name() << "_";
    result << "dataValue=" << ov::test::utils::vec2str(getElements<5>(data_elements)) << "_";
    result << "targetDevice=" << target_device << "_";
    return result.str();
}

void ConstantLayerTest::SetUp() {
    ov::Shape shape;
    ov::element::Type model_type;
    std::vector<std::string> data_elements;

    std::tie(shape, model_type, data_elements, targetDevice) = this->GetParam();

    auto constant = ov::op::v0::Constant::create(model_type, shape, data_elements);
    auto result = std::make_shared<ov::op::v0::Result>(constant);

    function = std::make_shared<ov::Model>(result, ov::ParameterVector{}, "constant");
}
}  // namespace test
}  // namespace ov

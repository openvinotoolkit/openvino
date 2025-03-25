// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/range_add.hpp"

#include "common_test_utils/node_builders/eltwise.hpp"

namespace ov {
namespace test {

// ------------------------------ V0 ------------------------------

std::string RangeAddSubgraphTest::getTestCaseName(const testing::TestParamInfo<RangeParams>& obj) {
    ov::element::Type input_type;
    float start, stop, step;
    std::string targetDevice;
    std::tie(start, stop, step, input_type, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "Start=" << start << separator;
    result << "Stop=" << stop << separator;
    result << "Step=" << step << separator;
    result << "ET=" << input_type << separator;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void RangeAddSubgraphTest::SetUp() {
    ov::element::Type element_type;
    float start, stop, step;
    std::tie(start, stop, step, element_type, targetDevice) = GetParam();

    auto startConstant = std::make_shared<ov::op::v0::Constant>(element_type, ov::Shape{}, start);
    auto stopConstant = std::make_shared<ov::op::v0::Constant>(element_type, ov::Shape{}, stop);
    auto stepConstant = std::make_shared<ov::op::v0::Constant>(element_type, ov::Shape{}, step);
    auto range = std::make_shared<ov::op::v0::Range>(startConstant, stopConstant, stepConstant);

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(element_type, range->get_shape())};
    auto eltwise = ov::test::utils::make_eltwise(params.front(), range, ov::test::utils::EltwiseTypes::ADD);
    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwise)};
    function = std::make_shared<ov::Model>(results, params, "RangeEltwise");
}

// ------------------------------ V4 ------------------------------

std::string RangeNumpyAddSubgraphTest::getTestCaseName(const testing::TestParamInfo<RangeParams>& obj) {
    ov::element::Type element_type;
    float start, stop, step;
    std::string targetDevice;
    std::tie(start, stop, step, element_type, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "Start=" << start << separator;
    result << "Stop=" << stop << separator;
    result << "Step=" << step << separator;
    result << "ET=" << element_type << separator;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void RangeNumpyAddSubgraphTest::SetUp() {
    ov::element::Type element_type;
    float start, stop, step;
    std::tie(start, stop, step, element_type, targetDevice) = GetParam();

    auto startConstant = std::make_shared<ov::op::v0::Constant>(element_type, ov::Shape{}, start);
    auto stopConstant = std::make_shared<ov::op::v0::Constant>(element_type, ov::Shape{}, stop);
    auto stepConstant = std::make_shared<ov::op::v0::Constant>(element_type, ov::Shape{}, step);
    auto range = std::make_shared<ov::op::v4::Range>(startConstant, stopConstant, stepConstant, element_type);

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(element_type, range->get_shape())};

    auto eltwise = ov::test::utils::make_eltwise(params.front(), range, ov::test::utils::EltwiseTypes::ADD);
    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwise)};
    function = std::make_shared<ov::Model>(results, params, "RangeEltwise");
}

}  // namespace test
}  // namespace ov

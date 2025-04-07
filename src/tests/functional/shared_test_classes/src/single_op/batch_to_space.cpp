// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/batch_to_space.hpp"

namespace ov {
namespace test {
std::string BatchToSpaceLayerTest::getTestCaseName(const testing::TestParamInfo<batchToSpaceParamsTuple> &obj) {
    std::vector<InputShape> shapes;
    std::vector<int64_t> block_shape, crops_begin, crops_end;
    ov::element::Type model_type;
    std::string target_name;
    std::tie(block_shape, crops_begin, crops_end, shapes, model_type, target_name) = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "inT=" << model_type.get_type_name() << "_";
    result << "BS=" << ov::test::utils::vec2str(block_shape) << "_";
    result << "CB=" << ov::test::utils::vec2str(crops_begin) << "_";
    result << "CE=" << ov::test::utils::vec2str(crops_end) << "_";
    result << "trgDev=" << target_name << "_";
    return result.str();
}

void BatchToSpaceLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    std::vector<int64_t> block_shape, crops_begin, crops_end;
    ov::element::Type model_type;
    std::tie(block_shape, crops_begin, crops_end, shapes, model_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front())};

    auto const_shape = ov::Shape{inputDynamicShapes.front().size()};
    auto block_shape_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, block_shape.data());
    auto crops_begin_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, crops_begin.data());
    auto crops_end_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, crops_end.data());

    auto b2s = std::make_shared<ov::op::v1::BatchToSpace>(params[0], block_shape_node, crops_begin_node, crops_end_node);
    ov::OutputVector results{std::make_shared<ov::op::v0::Result>(b2s)};
    function = std::make_shared<ov::Model>(results, params, "BatchToSpace");
}
}  // namespace test
}  // namespace ov

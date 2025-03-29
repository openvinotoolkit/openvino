// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/tile.hpp"

namespace ov {
namespace test {
std::string TileLayerTest::getTestCaseName(const testing::TestParamInfo<TileLayerTestParamsSet>& obj) {
    TileSpecificParams tile_params;
    ov::element::Type model_type;
    std::vector<InputShape> input_shapes;
    std::string target_device;
    std::tie(tile_params, model_type, input_shapes, target_device) = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < input_shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({input_shapes[i].first})
               << (i < input_shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < input_shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < input_shapes.size(); j++) {
            result << ov::test::utils::vec2str(input_shapes[j].second[i]) << (j < input_shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "Repeats=" << ov::test::utils::vec2str(tile_params) << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void TileLayerTest::SetUp() {
    TileSpecificParams tile_params;
    ov::element::Type model_type;
    std::vector<InputShape> input_shapes;
    std::tie(tile_params, model_type, input_shapes, targetDevice) = this->GetParam();
    init_input_shapes({input_shapes});

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto repeats = std::make_shared<ov::op::v0::Constant>(ov::element::i64, std::vector<size_t>{tile_params.size()}, tile_params);
    auto tile = std::make_shared<ov::op::v0::Tile>(param, repeats);
    function = std::make_shared<ov::Model>(tile->outputs(), ov::ParameterVector{param}, "tile");
}
}  // namespace test
}  // namespace ov

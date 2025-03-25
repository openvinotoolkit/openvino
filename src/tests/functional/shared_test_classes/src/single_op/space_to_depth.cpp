// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/space_to_depth.hpp"

namespace ov {
namespace test {

using ov::op::v0::SpaceToDepth;

static inline std::string SpaceToDepthModeToString(const SpaceToDepth::SpaceToDepthMode& mode) {
    static std::map<SpaceToDepth::SpaceToDepthMode, std::string> names = {
        {SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
        {SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, "DEPTH_FIRST"},
    };

    auto i = names.find(mode);
    if (i != names.end())
        return i->second;
    else
        throw std::runtime_error("Unsupported SpaceToDepthMode");
}

std::string SpaceToDepthLayerTest::getTestCaseName(const testing::TestParamInfo<spaceToDepthParamsTuple> &obj) {
    std::vector<InputShape> input_shapes;
    SpaceToDepth::SpaceToDepthMode mode;
    std::size_t block_size;
    ov::element::Type model_type;
    std::string target_device;
    std::tie(input_shapes, model_type, mode, block_size, target_device) = obj.param;
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
    result << "modelType=" << model_type.to_string() << "_";
    result << "M=" << SpaceToDepthModeToString(mode) << "_";
    result << "BS=" << block_size << "_";
    result << "targetDevice=" << target_device << "_";
    return result.str();
}

void SpaceToDepthLayerTest::SetUp() {
    std::vector<InputShape> input_shapes;
    SpaceToDepth::SpaceToDepthMode mode;
    std::size_t block_size;
    ov::element::Type model_type;
    std::tie(input_shapes, model_type, mode, block_size, targetDevice) = this->GetParam();

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto s2d = std::make_shared<ov::op::v0::SpaceToDepth>(param, mode, block_size);
    function = std::make_shared<ov::Model>(s2d->outputs(), ov::ParameterVector{param}, "SpaceToDepth");
}
}  // namespace test
}  // namespace ov

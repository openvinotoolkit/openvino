// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/depth_to_space.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/depth_to_space.hpp"

namespace ov {
namespace test {
using ov::op::v0::DepthToSpace;

static inline std::string DepthToSpaceModeToString(const DepthToSpace::DepthToSpaceMode& mode) {
    static std::map<DepthToSpace::DepthToSpaceMode, std::string> names = {
        {DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
        {DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, "DEPTH_FIRST"},
    };

    auto i = names.find(mode);
    if (i != names.end())
        return i->second;
    else
        throw std::runtime_error("Unsupported DepthToSpaceMode");
}

std::string DepthToSpaceLayerTest::getTestCaseName(const testing::TestParamInfo<depthToSpaceParamsTuple> &obj) {
    std::vector<InputShape> shapes;
    DepthToSpace::DepthToSpaceMode mode;
    std::size_t block_size;
    ov::element::Type model_type;
    std::string device_name;
    std::tie(shapes, model_type, mode, block_size, device_name) = obj.param;

    std::ostringstream result;
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
    result << "inPrc=" << model_type.get_type_name() << "_";
    result << "M=" << DepthToSpaceModeToString(mode) << "_";
    result << "BS=" << block_size << "_";
    result << "targetDevice=" << device_name << "_";
    return result.str();
}

void DepthToSpaceLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    DepthToSpace::DepthToSpaceMode mode;
    std::size_t block_size;
    ov::element::Type model_type;
    std::tie(shapes, model_type, mode, block_size, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto d2s = std::make_shared<ov::op::v0::DepthToSpace>(param, mode, block_size);
    auto result = std::make_shared<ov::op::v0::Result>(d2s);

    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "DepthToSpace");
}
}  // namespace test
}  // namespace ov

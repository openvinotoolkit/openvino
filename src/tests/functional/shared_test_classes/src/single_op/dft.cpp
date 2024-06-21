// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/dft.hpp"
#include "common_test_utils/node_builders/dft.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"

namespace ov {
namespace test {
static inline void set_real_number_generation_data(utils::InputGenerateData& inGenData) {
    inGenData.range = 8;
    inGenData.resolution = 32;
}

void DFTLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::test::utils::DFTOpType op_type;
    std::string target_device;
    std::tie(shapes, model_type, axes, signal_size, op_type, target_device) = GetParam();

    auto elemType = model_type;
    bool inPrcSigned = elemType.is_signed();

    ov::test::utils::InputGenerateData inGenData;
    if (elemType.is_real()) {
        set_real_number_generation_data(inGenData);
    }

    const auto& funcInputs = function->inputs();
    auto funcInput = funcInputs.begin();

    const auto node = funcInput->get_node_shared_ptr();
    const size_t inNodeCnt = node->get_input_size();

    auto it = ov::test::utils::inputRanges.find(ov::op::v7::DFT::get_type_info_static());
    if (op_type == ov::test::utils::DFTOpType::INVERSE) {
        it = ov::test::utils::inputRanges.find(ov::op::v7::IDFT::get_type_info_static());
    }

    if (it != ov::test::utils::inputRanges.end()) {
        ov::test::utils::Range ranges = it->second;
        inGenData = ranges.get_data(0, elemType);
    }

    inputs.clear();
    Tensor data_tensor = ov::test::utils::create_and_fill_tensor_act_dft(funcInput->get_element_type(),
                                            targetInputStaticShapes[0],
                                            inGenData.range, inGenData.start_from, inGenData.resolution, inGenData.seed);
    inputs.insert({funcInput->get_node_shared_ptr(), data_tensor});
}

std::string DFTLayerTest::getTestCaseName(const testing::TestParamInfo<DFTParams>& obj) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::test::utils::DFTOpType op_type;
    std::string target_device;
    std::tie(shapes, model_type, axes, signal_size, op_type, target_device) = obj.param;

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
    result << "Precision=" << model_type.get_type_name() << "_";
    result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "signal_size=" << ov::test::utils::vec2str(signal_size) << "_";
    result << "Inverse=" << (op_type == ov::test::utils::DFTOpType::INVERSE) << "_";
    result << "TargetDevice=" << target_device;
    return result.str();
}

void DFTLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::test::utils::DFTOpType op_type;
    std::tie(shapes, model_type, axes, signal_size, op_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto dft = ov::test::utils::make_dft(param, axes, signal_size, op_type);

    auto result = std::make_shared<ov::op::v0::Result>(dft);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "DFT");

    if (model_type == ov::element::f32) {
        abs_threshold = 8e-5;
    } else if (model_type == ov::element::bf16) {
        abs_threshold = 5e-7;
    }
}
}  // namespace test
}  // namespace ov

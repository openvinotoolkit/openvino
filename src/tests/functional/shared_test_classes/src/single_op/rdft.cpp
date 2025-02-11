// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/rdft.hpp"
#include "common_test_utils/node_builders/rdft.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"

namespace ov {
namespace test {
static inline void set_real_number_generation_data(utils::InputGenerateData& inGenData) {
    inGenData.range = 8;
    inGenData.resolution = 32;
}

void RDFTLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    std::vector<size_t> input_shape;
    ov::element::Type model_type;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::test::utils::DFTOpType op_type;
    std::string target_device;
    std::tie(input_shape, model_type, axes, signal_size, op_type, targetDevice) = this->GetParam();

    auto elemType = model_type;

    ov::test::utils::InputGenerateData inGenData;
    if (elemType.is_real()) {
        set_real_number_generation_data(inGenData);
    }

    const auto& funcInputs = function->inputs();
    auto funcInput = funcInputs.begin();

    const auto node = funcInput->get_node_shared_ptr();
    const size_t inNodeCnt = node->get_input_size();

    auto it = ov::test::utils::inputRanges.find(ov::op::v9::RDFT::get_type_info_static());
    if (op_type == ov::test::utils::DFTOpType::INVERSE) {
        it = ov::test::utils::inputRanges.find(ov::op::v9::IRDFT::get_type_info_static());
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

std::string RDFTLayerTest::getTestCaseName(const testing::TestParamInfo<RDFTParams>& obj) {
    std::vector<size_t> input_shape;
    ov::element::Type model_type;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::test::utils::DFTOpType op_type;
    std::string target_device;
    std::tie(input_shape, model_type, axes, signal_size, op_type, target_device) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "SignalSize=" << ov::test::utils::vec2str(signal_size) << "_";
    result << "Inverse=" << (op_type == ov::test::utils::DFTOpType::INVERSE) << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void RDFTLayerTest::SetUp() {
    std::vector<size_t> input_shape;
    ov::element::Type model_type;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::test::utils::DFTOpType op_type;
    std::tie(input_shape, model_type, axes, signal_size, op_type, targetDevice) = this->GetParam();

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape));
    auto rdft = ov::test::utils::make_rdft(param, axes, signal_size, op_type);
    function = std::make_shared<ov::Model>(rdft->outputs(), ov::ParameterVector{param}, "RDFT");

    if (model_type == ov::element::f32) {
        abs_threshold = 1e-4;
    }
}
}  // namespace test
}  // namespace ov
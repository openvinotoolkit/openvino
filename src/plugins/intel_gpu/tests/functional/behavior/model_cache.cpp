// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/2_input_subtract.hpp"
#include "common_test_utils/subgraph_builders/concat_with_params.hpp"
#include "common_test_utils/subgraph_builders/conv_bias.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu_no_reshapes.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu_non_zero.hpp"
#include "common_test_utils/subgraph_builders/convert_transpose.hpp"
#include "common_test_utils/subgraph_builders/detection_output.hpp"
#include "common_test_utils/subgraph_builders/kso_func.hpp"
#include "common_test_utils/subgraph_builders/matmul_bias.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/subgraph_builders/multiple_input_outpput_double_concat.hpp"
#include "common_test_utils/subgraph_builders/nested_branch_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/nested_split_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/read_concat_split_assign.hpp"
#include "common_test_utils/subgraph_builders/single_concat_with_constant.hpp"
#include "common_test_utils/subgraph_builders/single_conv.hpp"
#include "common_test_utils/subgraph_builders/single_split.hpp"
#include "common_test_utils/subgraph_builders/split_concat.hpp"
#include "common_test_utils/subgraph_builders/split_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/split_multi_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/ti_with_lstm_cell.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/pass/serialize.hpp"

namespace {
class CheckWeightlessCacheAccuracy : public ::testing::Test {
protected:
    std::shared_ptr<ov::Model> model;
    std::string xml_path;
    std::string bin_path;
    std::string cache_path;

    void SetUp() override;
    void TearDown() override;
    void run();
};

void CheckWeightlessCacheAccuracy::SetUp() {
    std::string filePrefix = ov::test::utils::generateTestFilePrefix();
    xml_path = filePrefix + ".xml";
    bin_path = filePrefix + ".bin";
    cache_path = filePrefix + ".blob";
}

void CheckWeightlessCacheAccuracy::TearDown() {
    std::remove(xml_path.c_str());
    std::remove(bin_path.c_str());
    std::remove(cache_path.c_str());
}

void CheckWeightlessCacheAccuracy::run() {
    ov::AnyMap config = { ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE) };
    ov::AnyMap config_with_weights_path = { ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE), ov::weights_path(bin_path) };
    auto core = ov::test::utils::PluginCache::get().core();
    ov::pass::Serialize(xml_path, bin_path).run_on_model(model);

    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(xml_path, ov::test::utils::DEVICE_GPU, config));

    auto ofstr = std::ofstream(cache_path, std::ofstream::binary);
    OV_ASSERT_NO_THROW(compiled_model.export_model(ofstr));
    ofstr.close();

    auto ifstr = std::ifstream(cache_path, std::ifstream::binary);
    ov::CompiledModel imported_model;
    OV_ASSERT_NO_THROW(imported_model = core->import_model(ifstr, ov::test::utils::DEVICE_GPU, config_with_weights_path));
    ifstr.close();

    auto orig_req = compiled_model.create_infer_request();
    auto new_req = imported_model.create_infer_request();

    for (size_t param_idx = 0; param_idx < model->get_parameters().size(); ++param_idx) {
        auto input = model->get_parameters().at(param_idx);
        auto tensor = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
        orig_req.set_tensor(input, tensor);
        new_req.set_tensor(input, tensor);
    }

    OV_ASSERT_NO_THROW(orig_req.infer());
    OV_ASSERT_NO_THROW(new_req.infer());

    auto result_vector = model->get_results();
    for (auto& res : result_vector) {
        auto orig_out = orig_req.get_tensor(res);
        auto new_out = new_req.get_tensor(res);
        ov::test::utils::compare(orig_out, new_out);
    }
}

TEST_F(CheckWeightlessCacheAccuracy, ReadConcatSplitAssign) {
    model = ov::test::utils::make_read_concat_split_assign({1, 1, 2, 4}, ov::element::f16);
    run();
}

TEST_F(CheckWeightlessCacheAccuracy, SingleConcatWithConstant) {
    model = ov::test::utils::make_single_concat_with_constant({1, 1, 2, 4}, ov::element::f16);
    run();
}

TEST_F(CheckWeightlessCacheAccuracy, TiWithLstmCell) {
    model = ov::test::utils::make_ti_with_lstm_cell(ov::element::f16);
    run();
}

}  // namespace

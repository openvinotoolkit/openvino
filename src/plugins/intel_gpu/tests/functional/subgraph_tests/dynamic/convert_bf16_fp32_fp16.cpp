// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"

namespace {
using ov::test::InputShape;

using BF16WeightsDecompressionParams = std::tuple<int, float>;                    // per-tensor zero-point

class BF16WeightsDecompression : public testing::WithParamInterface<BF16WeightsDecompressionParams>, public ov::test::TestsCommon {
public:
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    std::string targetDevice = ov::test::utils::DEVICE_GPU;

    static std::string get_test_case_name(testing::TestParamInfo<BF16WeightsDecompressionParams> obj) {
        int given;
        float expected;
        std::tie(given, expected) = obj.param;

        std::ostringstream result;
        result << " given=" << given << "expected=" << expected;

        return result.str();
    }

protected:
    std::shared_ptr<ov::Node> init_compressed_weights_subgraph(ov::element::Type_t typ, int val) {
        auto weights_tensor_left = ov::test::utils::create_and_fill_tensor(typ, ov::Shape{{2, 2}});
        auto weights_left = std::make_shared<ov::op::v0::Constant>(weights_tensor_left);
        weights_left->fill_data(typ, val);
        return weights_left;
    }

    std::shared_ptr<ov::Model> init_subgraph() {
        const auto weights_left = init_compressed_weights_subgraph(ov::element::f32, 2);
        const auto weights_right = init_compressed_weights_subgraph(ov::element::bf16, 3);
        auto r = std::make_shared<ov::op::v0::Convert>(weights_right, ov::element::f32);
        r->set_friendly_name("konwersja_bf16_na_f32");
        auto matmul_res = std::make_shared<ov::op::v0::MatMul>(weights_left, r);
        return std::make_shared<ov::Model>(ov::NodeVector{matmul_res}, ov::ParameterVector{}, "BF16WeightsDecompression");
    }

    std::shared_ptr<ov::Model> init_subgraph_without_convert(int right_val) {
        const auto weights_left = init_compressed_weights_subgraph(ov::element::bf16, 2);
        const auto weights_right = init_compressed_weights_subgraph(ov::element::bf16, right_val);
        auto matmul_res = std::make_shared<ov::op::v0::MatMul>(weights_left, weights_right);
        return std::make_shared<ov::Model>(ov::NodeVector{matmul_res}, ov::ParameterVector{}, "BF16WeightsDecompression");
    }

    std::shared_ptr<ov::Model> init_subgraph_without_convert_add() {
        const auto weights_left = init_compressed_weights_subgraph(ov::element::bf16, 2);
        const auto weights_right = init_compressed_weights_subgraph(ov::element::bf16, 3);
        auto matmul_res = std::make_shared<ov::op::v1::Add>(weights_left, weights_right);
        return std::make_shared<ov::Model>(ov::NodeVector{matmul_res}, ov::ParameterVector{}, "BF16WeightsDecompression");
    }
};

TEST_F(BF16WeightsDecompression, TypeCheck) {
    std::shared_ptr<ov::Model> subgraph = init_subgraph();
    ov::CompiledModel model = core->compile_model(subgraph, targetDevice, {ov::hint::inference_precision(ov::element::f32)});
    bool convert_found = false;
    std::cout << "entering " << model.get_runtime_model()->get_ops().size() << std::endl;
    for ( const auto& n : model.get_runtime_model()->get_ordered_ops() ) {
        std::cout << "znaleziono " << n->get_friendly_name() << std::endl;
        if ( n->get_friendly_name() == "konwersja_bf16_na_f32" ) {
            ASSERT_EQ(n->get_input_element_type(0),  ov::element::bf16);
            ASSERT_EQ(n->get_output_element_type(0),  ov::element::f32);
            convert_found = true;
        }
    }
    ASSERT_TRUE(convert_found);
}

TEST_F(BF16WeightsDecompression, Inference_input_3) {
    std::shared_ptr<ov::Model> function = init_subgraph();
    ov::CompiledModel cM = core->compile_model(function, targetDevice, {ov::hint::inference_precision(ov::element::f32)});
    auto request = cM.create_infer_request();
    request.infer();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], 12);
    }
}

TEST_P(BF16WeightsDecompression, Inference_without_convert) {
    int given;
    float expected;
    std::tie(given, expected) = GetParam();
    std::shared_ptr<ov::Model> function_without_convert = init_subgraph_without_convert(given);
    ov::CompiledModel model_without_convert = core->compile_model(function_without_convert, targetDevice, {ov::hint::inference_precision(ov::element::f32)});
    auto request = model_without_convert.create_infer_request();
    request.infer();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_NEAR(reinterpret_cast<unsigned short*>(out_tensor.data())[i], expected, 0.2f); //3072
    }
}

INSTANTIATE_TEST_SUITE_P(Inference_without_convert,
                         BF16WeightsDecompression,
                         testing::Values(BF16WeightsDecompressionParams(768, 17728.0f), BF16WeightsDecompressionParams(3, 16704.0f) ));


TEST_F(BF16WeightsDecompression, Inference_input_Add_without_convert) {
    std::shared_ptr<ov::Model> function_without_convert_add = init_subgraph_without_convert_add();
    ov::CompiledModel model_without_convert_add = core->compile_model(function_without_convert_add, targetDevice, \
    {ov::hint::inference_precision(ov::element::f32)});

    auto request = model_without_convert_add.create_infer_request();
    request.infer();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_NEAR(reinterpret_cast<unsigned short*>(out_tensor.data())[i], 16544, 0.2f);
    }
}
} // namespace

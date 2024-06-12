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

struct ShapeParams {
    ShapeParams() = default;
    ShapeParams(InputShape data_shape, ov::Shape weights_shape, int weights_group_size = -1)
        : data_shape(std::move(data_shape)),
          weights_shape(std::move(weights_shape)),
          weights_group_size(weights_group_size) {}

    InputShape data_shape;
    ov::Shape weights_shape;
    // Decompression group size. If the value is equal to -1, ordinary decompression is used
    int weights_group_size;
};

using MatmulWeightsDecompressionParams = std::tuple<ShapeParams,              // input shapes
                                                    ov::element::Type,        // weights type
                                                    ov::element::Type,        // activations type
                                                    bool,                     // transpose on weights
                                                    bool,                     // decompression subtract
                                                    bool,                     // reshape on decompression constants
                                                    bool>;                    // per-tensor zero-point

class BF16WeightsDecompression : public testing::WithParamInterface<MatmulWeightsDecompressionParams>,
                                   virtual public ov::test::SubgraphBaseTest {
public:
    ov::CompiledModel cM;
    ov::CompiledModel cMF16;
    std::shared_ptr<ov::Model> functionF16;
    ov::ParameterVector paramsf16{std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::Shape{{2, 2}})};
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::Shape{{2, 2}})};
    static std::string get_test_case_name(testing::TestParamInfo<MatmulWeightsDecompressionParams> obj) {
        ShapeParams shape_params;
        ov::element::Type weights_precision;
        ov::element::Type activations_precision;
        bool transpose;
        bool decompression_sub;
        bool reshape_on_decompression;
        bool per_tensor_zp;

        std::tie(shape_params,
                 weights_precision,
                 activations_precision,
                 transpose,
                 decompression_sub,
                 reshape_on_decompression,
                 per_tensor_zp) = obj.param;

        std::ostringstream result;
        result << "data_shape=";
        result << ov::test::utils::partialShape2str({shape_params.data_shape.first}) << "_";
        for (const auto& actual_shape : shape_params.data_shape.second) {
            result << ov::test::utils::partialShape2str({actual_shape}) << "_";
        }
        result << "_" << "weights_shape=" << shape_params.weights_shape << "_";
        result << "group_size=" << shape_params.weights_group_size << "_";
        result << "weights_precision=" << weights_precision << "_";
        result << "activations_precision=" << activations_precision << "_";
        result << "transpose_weights=" << transpose << "_";
        result << "decompression_subtract=" << decompression_sub << "_";
        result << "reshape_on_decompression=" << reshape_on_decompression << "_";
        result << "per_tensor_zp=" << per_tensor_zp;

        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph() {
        params.front()->output(0).get_tensor().set_names({"scalar1"});
        auto r = std::make_shared<ov::op::v0::Convert>(params[0], ov::element::f32);
        r->set_friendly_name("konwersja_bf16_na_f32");
        const auto weights_subgraph = init_compressed_weights_subgraph();
        auto matmul_res = std::make_shared<ov::op::v0::MatMul>(weights_subgraph, r);
        //auto end = std::make_shared<ov::op::v0::Convert>(matmul_res, ov::element::f16);
        //end->set_friendly_name("konwersja_fp32_na_f16");
        return std::make_shared<ov::Model>(ov::NodeVector{matmul_res}, params, "BF16WeightsDecompression");
    }

    std::shared_ptr<ov::Model> init_subgraph_f16() {
        paramsf16.front()->output(0).get_tensor().set_names({"scalar1"});
        auto r = std::make_shared<ov::op::v0::Convert>(paramsf16[0], ov::element::f16);
        r->set_friendly_name("konwersja_bf16_na_f16");
        const auto weights_subgraph = init_compressed_weights_subgraph_f16();
        auto matmul_res = std::make_shared<ov::op::v0::MatMul>(weights_subgraph, r);
        auto end = std::make_shared<ov::op::v0::Convert>(matmul_res, ov::element::f32);
        end->set_friendly_name("konwersja_fp16_na_f32");
        return std::make_shared<ov::Model>(ov::NodeVector{end}, paramsf16, "BF16WeightsDecompression");
    }


    std::shared_ptr<ov::Node> init_compressed_weights_subgraph() {
        auto weights_tensor_left = ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{{2, 2}});
        auto weights_left = std::make_shared<ov::op::v0::Constant>(weights_tensor_left);
        weights_left->fill_data(ov::element::f32, 2);
        return weights_left;
    }

    std::shared_ptr<ov::Node> init_compressed_weights_subgraph_f16() {
        auto weights_tensor_left = ov::test::utils::create_and_fill_tensor(ov::element::f16, ov::Shape{{2, 2}});
        auto weights_left = std::make_shared<ov::op::v0::Constant>(weights_tensor_left);
        weights_left->fill_data(ov::element::f16, 2);
        return weights_left;
    }

    void SetUp() override {
        init_input_shapes({ov::test::InputShape{ov::PartialShape{3, 3}, 3}, ov::test::InputShape{ov::PartialShape{3, 3}, 5}});
        targetDevice = ov::test::utils::DEVICE_GPU;
        ShapeParams shape_params;
        ov::element::Type weights_precision;
        ov::element::Type activations_precision;
        inType = outType = activations_precision;

        function = init_subgraph();
        functionF16 = init_subgraph_f16();
        if (activations_precision == ov::element::f16) {
            abs_threshold = 1.0f;
        } else {
            abs_threshold = 1e-4f;
        }
        cM = core->compile_model(function, targetDevice, {ov::hint::inference_precision(ov::element::f32)});
        cMF16 = core->compile_model(functionF16, targetDevice, {ov::hint::inference_precision(ov::element::f16)});
    }

    void check_results() {
        bool convert_found = false;
        for ( const auto& n : cM.get_runtime_model()->get_ordered_ops() ) {
            if ( n->get_friendly_name() == "konwersja_bf16_na_f32" ) {
                ASSERT_EQ(n->get_input_element_type(0),  ov::element::bf16);
                ASSERT_EQ(n->get_output_element_type(0),  ov::element::f32);
                convert_found = true;
            }
        }
        ASSERT_TRUE(convert_found);
    }
};

TEST_F(BF16WeightsDecompression, TypeCheck) {
    check_results();
}

TEST_F(BF16WeightsDecompression, Inference_input_2) {
    const auto& model_inputs = cM.inputs();
    const auto& model_input = model_inputs[0];
    ov::Tensor tensor = ov::Tensor(model_input.get_element_type(), model_input.get_shape());
    for (size_t j =0 ; j < tensor.get_size(); j++) {
        (reinterpret_cast<unsigned short*>(tensor.data())) [j] = 16384; //2 in bf16
    }
    auto request = cM.create_infer_request();
    request.set_tensor("scalar1", tensor);
    request.infer();
    request.wait();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], 8);
    }
}

TEST_F(BF16WeightsDecompression, Inference_input_3) {
    const auto& model_inputs = cM.inputs();
    const auto& model_input = model_inputs[0];
    ov::Tensor tensor = ov::Tensor(model_input.get_element_type(), model_input.get_shape());
    for (size_t j =0 ; j < tensor.get_size(); j++) {
        (reinterpret_cast<unsigned short*>(tensor.data())) [j] = 16448; //3 inbf16
    }
    auto request = cM.create_infer_request();
    request.set_tensor("scalar1", tensor);
    request.infer();
    request.wait();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], 12);
    }
}

TEST_F(BF16WeightsDecompression, Inference_input_768) {
    const auto& model_inputs = cM.inputs();
    const auto& model_input = model_inputs[0];
    ov::Tensor tensor = ov::Tensor(model_input.get_element_type(), model_input.get_shape());
    for (size_t j =0 ; j < tensor.get_size(); j++) {
        (reinterpret_cast<unsigned short*>(tensor.data())) [j] = 16448+1024; //3*256 inbf16
    }
    auto request = cM.create_infer_request();
    request.set_tensor("scalar1", tensor);
    request.infer();
    request.wait();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], 3072);
    }
}

TEST_F(BF16WeightsDecompression, Inference_input_768x16) {
    const auto& model_inputs = cM.inputs();
    const auto& model_input = model_inputs[0];
    ov::Tensor tensor = ov::Tensor(model_input.get_element_type(), model_input.get_shape());
    for (size_t j =0 ; j < tensor.get_size(); j++) {
        (reinterpret_cast<unsigned short*>(tensor.data())) [j] = 16448+1024+512; //3*256*16 inbf16
    }
    auto request = cM.create_infer_request();
    request.set_tensor("scalar1", tensor);
    request.infer();
    request.wait();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], 3072*16);
    }
}

TEST_F(BF16WeightsDecompression, Inference_input_768x64) {
    const auto& model_inputs = cM.inputs();
    const auto& model_input = model_inputs[0];
    ov::Tensor tensor = ov::Tensor(model_input.get_element_type(), model_input.get_shape());
    for (size_t j =0 ; j < tensor.get_size(); j++) {
        (reinterpret_cast<unsigned short*>(tensor.data())) [j] = 16448+1024+512+256; //3*256*64 inbf16
    }
    auto request = cM.create_infer_request();
    request.set_tensor("scalar1", tensor);
    request.infer();
    request.wait();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], 3072*64);
    }
}

TEST_F(BF16WeightsDecompression, Inference_input_768x128) {
    const auto& model_inputs = cM.inputs();
    const auto& model_input = model_inputs[0];
    ov::Tensor tensor = ov::Tensor(model_input.get_element_type(), model_input.get_shape());
    for (size_t j =0 ; j < tensor.get_size(); j++) {
        (reinterpret_cast<unsigned short*>(tensor.data())) [j] = 16448+1024+512+256+128; //3*256*128 inbf16
    }
    auto request = cM.create_infer_request();
    request.set_tensor("scalar1", tensor);
    request.infer();
    request.wait();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], 3072*128);
    }
}

TEST_F(BF16WeightsDecompression, Inference_input_768x256) {
    const auto& model_inputs = cM.inputs();
    const auto& model_input = model_inputs[0];
    ov::Tensor tensor = ov::Tensor(model_input.get_element_type(), model_input.get_shape());
    for (size_t j =0 ; j < tensor.get_size(); j++) {
        (reinterpret_cast<unsigned short*>(tensor.data())) [j] = 16448+1024+1024; //3*256*256 inbf16
    }
    auto request = cM.create_infer_request();
    request.set_tensor("scalar1", tensor);
    request.infer();
    request.wait();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], 3072*256);
    }
}

TEST_F(BF16WeightsDecompression, Inference_input_16448plus8192) {
    const auto& model_inputs = cM.inputs();
    const auto& model_input = model_inputs[0];
    ov::Tensor tensor = ov::Tensor(model_input.get_element_type(), model_input.get_shape());
    for (size_t j =0 ; j < tensor.get_size(); j++) {
        (reinterpret_cast<unsigned short*>(tensor.data())) [j] = 16448+8192;
    }
    auto request = cM.create_infer_request();
    request.set_tensor("scalar1", tensor);
    request.infer();
    request.wait();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        float expected_val = 221360928884514619392.0f;
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], expected_val);
    }
}

TEST_F(BF16WeightsDecompression, Inference_input_16448plus8192_but_minus) {
    const auto& model_inputs = cM.inputs();
    const auto& model_input = model_inputs[0];
    ov::Tensor tensor = ov::Tensor(model_input.get_element_type(), model_input.get_shape());
    for (size_t j =0 ; j < tensor.get_size(); j++) {
        (reinterpret_cast<unsigned short*>(tensor.data())) [j] = 16448+8192+32768;
    }
    auto request = cM.create_infer_request();
    request.set_tensor("scalar1", tensor);
    request.infer();
    request.wait();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        float expected_val = -221360928884514619392.0f;
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], expected_val);
    }
}

TEST_F(BF16WeightsDecompression, Inference_input_minus3) {
    const auto& model_inputs = cM.inputs();
    const auto& model_input = model_inputs[0];
    ov::Tensor tensor = ov::Tensor(model_input.get_element_type(), model_input.get_shape());
    for (size_t j =0 ; j < tensor.get_size(); j++) {
        (reinterpret_cast<unsigned short*>(tensor.data())) [j] = 49216; //-3 inbf16
    }
    auto request = cM.create_infer_request();
    request.set_tensor("scalar1", tensor);
    request.infer();
    request.wait();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], -12);
    }
}

TEST_F(BF16WeightsDecompression, Inference_input_3_f16) {
    const auto& model_inputs = cMF16.inputs();
    const auto& model_input = model_inputs[0];
    ov::Tensor tensor = ov::Tensor(model_input.get_element_type(), model_input.get_shape());
    for (size_t j =0 ; j < tensor.get_size(); j++) {
        (reinterpret_cast<unsigned short*>(tensor.data())) [j] = 16448; //3 inbf16
    }
    auto request = cMF16.create_infer_request();
    request.set_tensor("scalar1", tensor);
    request.infer();
    request.wait();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], 12);
    }
}
} // namespace

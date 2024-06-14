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
    ov::CompiledModel model_without_convert;
    ov::CompiledModel model_without_convert_add;
    std::shared_ptr<ov::Model> function_without_convert;
    std::shared_ptr<ov::Model> function_without_convert_add;
    static std::string get_test_case_name(testing::TestParamInfo<MatmulWeightsDecompressionParams> obj) {
        ShapeParams shape_params;
        ov::element::Type activations_precision;
        ov::element::Type weights_precision;
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

    std::shared_ptr<ov::Model> init_subgraph_without_convert() {
        const auto weights_left = init_compressed_weights_subgraph(ov::element::bf16, 2);
        const auto weights_right = init_compressed_weights_subgraph(ov::element::bf16, 3);
        auto matmul_res = std::make_shared<ov::op::v0::MatMul>(weights_left, weights_right);
        return std::make_shared<ov::Model>(ov::NodeVector{matmul_res}, ov::ParameterVector{}, "BF16WeightsDecompression");
    }

    std::shared_ptr<ov::Model> init_subgraph_without_convert_add() {
        const auto weights_left = init_compressed_weights_subgraph(ov::element::bf16, 2);
        const auto weights_right = init_compressed_weights_subgraph(ov::element::bf16, 3);
        auto matmul_res = std::make_shared<ov::op::v1::Add>(weights_left, weights_right);
        return std::make_shared<ov::Model>(ov::NodeVector{matmul_res}, ov::ParameterVector{}, "BF16WeightsDecompression");
    }

    void SetUp() override {
        init_input_shapes({ov::test::InputShape{ov::PartialShape{3, 3}, 3}, ov::test::InputShape{ov::PartialShape{3, 3}, 5}});
        targetDevice = ov::test::utils::DEVICE_GPU;
        ShapeParams shape_params;
        ov::element::Type activations_precision;
        inType = outType = activations_precision;

        function = init_subgraph();
        function_without_convert = init_subgraph_without_convert();
        function_without_convert_add = init_subgraph_without_convert_add();
        if (activations_precision == ov::element::f16) {
            abs_threshold = 1.0f;
        } else {
            abs_threshold = 1e-4f;
        }
        cM = core->compile_model(function, targetDevice, {ov::hint::inference_precision(ov::element::f32)});
        model_without_convert = core->compile_model(function_without_convert, targetDevice, {ov::hint::inference_precision(ov::element::f32)});
        model_without_convert_add = core->compile_model(function_without_convert_add, targetDevice, {ov::hint::inference_precision(ov::element::f32)});
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

TEST_F(BF16WeightsDecompression, Inference_input_3) {
    auto request = cM.create_infer_request();
    request.infer();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_FLOAT_EQ(reinterpret_cast<float*>(out_tensor.data())[i], 12);
    }
}

TEST_F(BF16WeightsDecompression, Inference_input_3_without_convert) {
    auto request = model_without_convert.create_infer_request();
    request.infer();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_NEAR(reinterpret_cast<ushort*>(out_tensor.data())[i], 16704, 0.2f);
    }
}

TEST_F(BF16WeightsDecompression, Inference_input_Add_without_convert) {
    auto request = model_without_convert_add.create_infer_request();
    request.infer();
    auto out_tensor = request.get_output_tensor();
    for ( size_t i = 0 ; i < out_tensor.get_size(); i++ ) {
        ASSERT_NEAR(reinterpret_cast<ushort*>(out_tensor.data())[i], 16544, 0.2f);
    }
}
} // namespace

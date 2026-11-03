// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using rope_params = std::tuple<ov::element::Type, std::string>;
using rope_params_2 = std::tuple<bool, ov::element::Type, std::string>;
using rope_params_qwenvit = std::tuple<ov::element::Type, std::string, std::string>;
using rope_params_chatglm = std::tuple<ov::element::Type, std::string, bool>;
using rope_params_cohere = std::tuple<ov::element::Type, std::string, bool>;

class RoPETestFlux : public SubgraphBaseTest, public testing::WithParamInterface<rope_params> {
private:
    std::shared_ptr<ov::Model> build_rope_flux(int batch,
                                               int seq_length,
                                               int num_head,
                                               int ndims,
                                               ov::element::Type element_type);

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params>& obj);
};

class RoPETestQwenVL : public SubgraphBaseTest, public testing::WithParamInterface<rope_params_qwenvit> {
private:
    std::shared_ptr<ov::Model> buildROPE_QwenVL(ov::element::Type element_type,
                                                ov::PartialShape input_shape,
                                                ov::PartialShape cos_shape,
                                                ov::PartialShape sin_shape,
                                                std::string split_op_type);

protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params_qwenvit>& obj);
};

class RoPETestLlama2StridedSlice : public SubgraphBaseTest, public testing::WithParamInterface<rope_params> {
private:
    std::shared_ptr<ov::Model> buildROPE_Llama2(int batch,
                                                int seq_length,
                                                int max_position_embeddings,
                                                int num_head,
                                                int ndims);

protected:
    ov::OutputVector makeCosSinCache(int max_position_embeddings, int rotary_ndims);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params>& obj);
};

class RoPETestChatGLMStridedSlice : public SubgraphBaseTest, public testing::WithParamInterface<rope_params> {
private:
    std::shared_ptr<ov::Model> buildROPE_ChatGLM(int batch,
                                                 int head_cnt,
                                                 int rotary_dims,
                                                 ov::element::Type element_type);

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params>& obj);
};

class RoPETestQwen7bStridedSlice : public SubgraphBaseTest, public testing::WithParamInterface<rope_params_2> {
private:
    std::shared_ptr<ov::Model> buildROPE_QWen7b(bool specialReshape, ov::element::Type element_type);

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params_2>& obj);
};

class RoPETestGPTJStridedSlice : public SubgraphBaseTest, public testing::WithParamInterface<rope_params_2> {
private:
    std::shared_ptr<ov::Model> buildROPE_GPTJ(int num_head,
                                              int hidden_dims,
                                              int rotary_dims,
                                              bool hasShapeOf,
                                              ov::element::Type element_type);

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params_2>& obj);
};

class RoPETestRotateHalfWithoutTranspose : public SubgraphBaseTest, public testing::WithParamInterface<rope_params> {
private:
    ov::OutputVector makeCosSinCache(int max_position_embeddings, int rotary_ndims);
    std::shared_ptr<ov::Model> buildROPE_RotateHalfWithoutTranspose(int batch,
                                                                    int seq_length,
                                                                    int max_position_embeddings,
                                                                    int num_head,
                                                                    int ndims);

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params>& obj);
};

// Tests RoPE fusion for an RMS-normalized input whose cos/sin are computed from position ids inside the model.
// Params: <inference precision and model element type, target device>
//
// Model (shapes: batch = 1, num_head = 4, ndims = 128, dynamic sequence length):
//
//   input [1, 4, ?, 128]                                 pos_ids [1, ?] (i32)
//        |                                                    |
//   RMS norm: input * 1 / sqrt(mean(input^2) + eps)      Unsqueeze -> Convert (element type)
//        |                                                    |
//   Multiply (scale = 1.25)                              MatMul (inv_freq [1, 64, 1])
//        |                                                    |
//    rope_input                                          Transpose -> Concat(x, x)
//        |                                                    |
//        |                                               +----+----+
//        |                                               |         |
//        |                                              Cos       Sin
//        |                                               |         |
//        |                                           Unsqueeze  Unsqueeze
//        |                                               |         |
//        |                                           cos_values  sin_values
//        |                                               |         |
//        +-----------------------------------------------+---------+
//                                                        v
//   output = rope_input * cos_values + rotate_half(rope_input) * sin_values
//
//   where rotate_half(x) = Concat(-x[..., 64:], x[..., :64])
class RoPETestRotateHalfRMS : public SubgraphBaseTest, public testing::WithParamInterface<rope_params> {
private:
    std::shared_ptr<ov::Model> buildROPE_RotateHalfRMS(int batch, int num_head, int ndims);

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params>& obj);
};

class RoPETestLlama2Slice : public RoPETestLlama2StridedSlice {
private:
    std::shared_ptr<ov::Model> buildROPE_Llama2(int batch,
                                                int seq_length,
                                                int max_position_embeddings,
                                                int num_head,
                                                int ndims);

protected:
    void SetUp() override;
};

class RoPETestChatGLMSlice : public RoPETestChatGLMStridedSlice {
private:
    std::shared_ptr<ov::Model> buildROPE_ChatGLM(int batch,
                                                 int head_cnt,
                                                 int rotary_dims,
                                                 ov::element::Type element_type);

protected:
    void SetUp() override;
};

class RoPETestQwen7bSlice : public RoPETestQwen7bStridedSlice {
private:
    std::shared_ptr<ov::Model> buildROPE_Qwen7b(bool specialReshape, ov::element::Type element_type);

protected:
    void SetUp() override;
};

class RoPETestGPTJSlice : public RoPETestGPTJStridedSlice {
private:
    std::shared_ptr<ov::Model> buildROPE_GPTJ(int num_head,
                                              int hidden_dims,
                                              int rotary_dims,
                                              bool hasShapeOf,
                                              ov::element::Type element_type);

protected:
    void SetUp() override;
};

class RoPETestChatGLM2DRoPEStridedSlice : public SubgraphBaseTest, public testing::WithParamInterface<rope_params> {
private:
    std::shared_ptr<ov::Model> buildROPE_ChatGLM(int batch,
                                                 int head_cnt,
                                                 int rotary_dims,
                                                 ov::element::Type element_type);

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params>& obj);
};

class RoPETestChatGLMHF : public SubgraphBaseTest, public testing::WithParamInterface<rope_params_chatglm> {
private:
    std::shared_ptr<ov::Model> buildROPE_ChatGLMHF(int seq_length,
                                                   int num_heads,
                                                   int rotary_ndims,
                                                   ov::element::Type element_type,
                                                   bool use_split);

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params_chatglm>& obj);
};

class RoPETestGPTOSS : public SubgraphBaseTest, public testing::WithParamInterface<rope_params> {
private:
    std::shared_ptr<ov::Model> buildROPE_GPTOSS(int num_head,
                                              int rotary_dims,
                                              ov::element::Type element_type);

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params>& obj);
};

class RoPETestLtxVideo : public SubgraphBaseTest, public testing::WithParamInterface<rope_params> {
private:
    std::shared_ptr<ov::Model> buildROPE_LtxVideo(int batch,
                                                  int seq_length,
                                                  int rotary_dims,
                                                  ov::element::Type element_type);

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params>& obj);
};

class RoPETestCohere : public SubgraphBaseTest, public testing::WithParamInterface<rope_params_cohere> {
private:
    std::shared_ptr<ov::Model> buildROPE_Cohere(int batch,
                                                int seq_length,
                                                int num_head,
                                                int rotary_dims,
                                                ov::element::Type element_type,
                                                bool use_pa_pattern);

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params_cohere>& obj);
};

}  // namespace test
}  // namespace ov

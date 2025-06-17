// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using rope_params = std::tuple<ov::element::Type, std::string>;
using rope_params_2 = std::tuple<bool, ov::element::Type, std::string>;

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

class RoPETestLlama2StridedSlice : public SubgraphBaseTest, public testing::WithParamInterface<rope_params> {
private:
    std::shared_ptr<ov::Model> buildROPE_Llama2(int batch,
                                                int seq_length,
                                                int max_position_embeddings,
                                                int num_head,
                                                int ndims);
protected:
    ov::Tensor create_i32_tensor(const ov::Shape& shape, int start, int step = 1);
    ov::OutputVector makeCosSinCache(int max_position_embeddings, int rotary_ndims);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params>& obj);
};

class RoPETestChatGLMStridedSlice : public SubgraphBaseTest, public testing::WithParamInterface<rope_params> {
private:
    std::shared_ptr<ov::Model> buildROPE_ChatGLM(int batch, int head_cnt, int rotary_dims, ov::element::Type element_type);
protected:
    ov::Tensor create_i32_tensor(const ov::Shape& shape, int start, int step = 1);
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
    ov::Tensor create_i32_tensor(const ov::Shape& shape, int start, int step = 1);
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
    std::shared_ptr<ov::Model> buildROPE_ChatGLM(int batch, int head_cnt, int rotary_dims, ov::element::Type element_type);
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
    std::shared_ptr<ov::Model> buildROPE_ChatGLM(int batch, int head_cnt, int rotary_dims, ov::element::Type element_type);
protected:
    ov::Tensor create_i32_tensor(const ov::Shape& shape, int start, int step = 1);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<rope_params>& obj);
};

}  // namespace test
}  // namespace ov

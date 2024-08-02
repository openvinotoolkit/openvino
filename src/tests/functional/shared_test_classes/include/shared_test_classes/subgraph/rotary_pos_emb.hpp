// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

class RoPETestLlama2 : public SubgraphBaseTest, public testing::WithParamInterface<std::string> {
private:
    ov::OutputVector makeCosSinCache(int max_position_embeddings, int rotary_ndims);
    std::shared_ptr<ov::Model> buildROPE_Llama2(int batch,
                                                int seq_length,
                                                int max_position_embeddings,
                                                int num_head,
                                                int ndims);
    ov::Tensor create_i32_tensor(const ov::Shape& shape, int start, int step = 1);
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::string>& obj);
};

class RoPETestChatGLM : public SubgraphBaseTest, public testing::WithParamInterface<std::string> {
private:
    std::shared_ptr<ov::Model> buildROPE_ChatGLM(int batch, int head_cnt, int rotary_dims);
    ov::Tensor create_i32_tensor(const ov::Shape& shape, int start, int step = 1);
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::string>& obj);
};

class RoPETestQwen7b : public SubgraphBaseTest, public testing::WithParamInterface<std::tuple<bool, std::string>> {
private:
    std::shared_ptr<ov::Model> buildROPE_QWen7b(bool specialReshape);
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<bool, std::string>>& obj);
};

class RoPETestGPTJ : public SubgraphBaseTest, public testing::WithParamInterface<std::tuple<bool, std::string>> {
private:
    std::shared_ptr<ov::Model> buildROPE_GPTJ(int num_head,
                                              int hidden_dims,
                                              int rotary_dims,
                                              bool hasShapeOf);
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<bool, std::string>>& obj);
};

}  // namespace test
}  // namespace ov

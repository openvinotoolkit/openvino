// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<std::vector<InputShape>,                       // Input shapes
                   std::vector<ov::element::Type>,                // Input Element types
                   ov::element::Type,                             // Inference precision
                   size_t,                                        // Thread count
                   std::string,                                   // Target Device
                   ov::AnyMap,                                    // Config
                   std::pair<size_t, std::pair<size_t, size_t>>,  // {number of hidden layers, {expected number of
                                                                  // subgraphs, expected number of nodes}}
                   size_t                                         // hidden matmul layers size
                   >
    MLPSeqParams;

class MLPSeqBase :  virtual public SnippetsTestsCommon {
public:
    constexpr static size_t default_thread_count = 0;

protected:
    void SetUp() override;
    void compile_model() override;
    virtual std::shared_ptr<SnippetsFunctionBase> get_subgraph(size_t num_hidden_layers, size_t hidden_matmul_size) const = 0;
    virtual void init_params(std::vector<InputShape>& input_shapes, ov::element::Type& prc, ov::AnyMap& additional_config) = 0;

    size_t m_thread_count;
    std::vector<ov::element::Type> m_input_types;
    size_t m_num_hidden_layers, m_hidden_matmul_size;
};

class MLPSeq : public testing::WithParamInterface<ov::test::snippets::MLPSeqParams>,
            virtual public MLPSeqBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::test::snippets::MLPSeqParams>& obj);

protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph(size_t num_hidden_layers, size_t hidden_matmul_size) const override;
    void init_params(std::vector<InputShape>& input_shapes, ov::element::Type& prc, ov::AnyMap& additional_config) override;
};

class MLPSeqQuantized : public MLPSeq {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph(size_t num_hidden_layers, size_t hidden_matmul_size) const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov

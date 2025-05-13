// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<std::vector<InputShape>,         // Input shapes
                   std::vector<ov::element::Type>,  // Input Element types
                   ov::element::Type,               // Inference precision
                   size_t,                          // Thread count
                   size_t,                          // Expected num nodes
                   size_t,                          // Expected num subgraphs
                   std::string,                     // Target Device
                   ov::AnyMap,                      // Config
                   size_t                           // Expected num hidden layers
                   >
    MLPParams;

class MLPBase :  virtual public SnippetsTestsCommon {
public:
    constexpr static size_t default_thread_count = 0;

protected:
    void SetUp() override;
    void compile_model() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    virtual std::shared_ptr<SnippetsFunctionBase> get_subgraph(size_t num_hidden_layers) const = 0;
    virtual void init_params(std::vector<InputShape>& input_shapes, ov::element::Type& prc, ov::AnyMap& additional_config) = 0;

    size_t m_thread_count;
    std::vector<ov::element::Type> m_input_types;
    size_t m_num_input_shapes, m_num_hidden_layers;
};

class MLP : public testing::WithParamInterface<ov::test::snippets::MLPParams>,
            virtual public MLPBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::MLPParams> obj);

protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph(size_t num_hidden_layers) const override;
    void init_params(std::vector<InputShape>& input_shapes, ov::element::Type& prc, ov::AnyMap& additional_config) override;
};

class MLPQuantized : public MLP {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph(size_t num_hidden_layers) const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov

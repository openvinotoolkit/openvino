// Copyright (C) 2022 Intel Corporation
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
                   bool,                            // With Multiply
                   size_t,                          // Thread count
                   size_t,                          // Expected num nodes
                   size_t,                          // Expected num subgraphs
                   std::string,                     // Target Device
                   ov::AnyMap                       // Config
                   >
    MHAParams;

typedef std::tuple<std::vector<InputShape>,         // Input shapes
                   std::vector<ov::element::Type>,  // Input Element types
                   ov::element::Type,               // Inference precision
                   size_t,                          // Thread count
                   size_t,                          // Expected num nodes
                   size_t,                          // Expected num subgraphs
                   std::string,                     // Target Device
                   ov::AnyMap                       // Config
                   >
MHAWithDynamicMulParams;

class MHABase :  virtual public SnippetsTestsCommon {
public:
    constexpr static size_t default_thread_count = 0;

protected:
    void SetUp() override;
    void compile_model() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    virtual std::shared_ptr<SnippetsFunctionBase> get_subgraph() const = 0;
    virtual void init_params(std::vector<InputShape>& input_shapes, ov::element::Type& prc, ov::AnyMap& additional_config) = 0;
    virtual void init_thresholds();

    size_t m_thread_count;
    std::vector<ov::element::Type> m_input_types;
};

class MHA : public testing::WithParamInterface<ov::test::snippets::MHAParams>,
            virtual public MHABase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAParams> obj);

protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph() const override;
    void init_params(std::vector<InputShape>& input_shapes, ov::element::Type& prc, ov::AnyMap& additional_config) override;

    bool m_with_mul = false;
};

class MHASelect : public MHA {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    std::shared_ptr<SnippetsFunctionBase> get_subgraph() const override;
};

class MHAWOTransposeOnInputs : public MHA {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph() const override;
};

class MHAWOTranspose : public MHA {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph() const override;
};

class MHAMulAdd : public MHA {
    std::shared_ptr<SnippetsFunctionBase> get_subgraph() const override;
};

class MHATransposedB : public MHA {
    std::shared_ptr<SnippetsFunctionBase> get_subgraph() const override;
};

class MHAINT8MatMul : public MHA {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph() const override;
    void init_thresholds() override;
};

class MHAQuantMatMul0 : public MHA {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph() const override;
};

class MHAFQAfterMatMul : public MHA {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph() const override;
};

class MHAFQ : public MHA {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph() const override;
    void init_thresholds() override;
};

class MHAWithExtractedReshape : public MHA {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph() const override;
};

class MHAWithDynamicMul : public testing::WithParamInterface<ov::test::snippets::MHAWithDynamicMulParams>,
                          virtual public MHABase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAWithDynamicMulParams> obj);

protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph() const override;
    void init_params(std::vector<InputShape>& input_shapes, ov::element::Type& prc, ov::AnyMap& additional_config) override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov

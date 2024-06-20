// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <common_test_utils/ov_test_utils.hpp>
#include "snippets/op/subgraph.hpp"
#include "snippets_helpers.hpp"
#include "snippets/pass/manager.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/runtime_configurator.hpp"

namespace ov {
namespace test {
namespace snippets {

using BlockedShapeVector = ov::snippets::op::Subgraph::BlockedShapeVector;

class DummyEmitter : public ov::snippets::Emitter {
public:
    DummyEmitter(const std::vector<ov::Node::type_info_t>& custom_opset = {}) : ov::snippets::Emitter() {}
    void emit_code(const std::vector<size_t>&,
                   const std::vector<size_t>&,
                   const std::vector<size_t>&,
                   const std::vector<size_t>&) const override {}
    void emit_data() const override {}
};

struct DummyCompiledSnippet : public ov::snippets::CompiledSnippet {
    const uint8_t* get_code() const override { return nullptr; }
    size_t get_code_size() const override { return 0; }
    bool empty() const override { return true; }
};

class DummyRuntimeConfig : public ov::snippets::RuntimeConfig {
public:
    DummyRuntimeConfig() : ov::snippets::RuntimeConfig() {}
};

class DummyRuntimeConfigurator : public ov::snippets::RuntimeConfigurator {
public:
    DummyRuntimeConfigurator() : RuntimeConfigurator(std::make_shared<DummyRuntimeConfig>()) {}
};

class DummyTargetMachine : public ov::snippets::TargetMachine {
public:
    DummyTargetMachine(const std::vector<ov::Node::type_info_t>& custom_opset = {});
    bool is_supported() const override { return true; }
    ov::snippets::CompiledSnippetPtr get_snippet() override { return std::make_shared<DummyCompiledSnippet>(); }
    size_t get_lanes() const override { return 10; }
    std::shared_ptr<TargetMachine> clone() const override { return std::make_shared<DummyTargetMachine>(); }
    size_t get_reg_count() const override { return 16; }
};

class DummyGenerator : public ov::snippets::Generator {
public:
    DummyGenerator() : ov::snippets::Generator(std::make_shared<DummyTargetMachine>()) {}
    DummyGenerator(const std::shared_ptr<ov::snippets::TargetMachine>& t) : ov::snippets::Generator(t) {}
    std::shared_ptr<Generator> clone() const override { return std::make_shared<DummyGenerator>(target); }

protected:
    ov::snippets::RegType get_op_out_reg_type(const ov::Output<ov::Node>& out) const override { return ov::snippets::RegType::vec; };
};

class LoweringTests : public TransformationTestsF {
public:
    LoweringTests();

    void SetUp() override;
    void TearDown() override;

    static std::shared_ptr<ov::snippets::op::Subgraph> getSubgraph(const std::shared_ptr<Model>& f);
    using IShapeInferSnippetsFactory = ov::snippets::IShapeInferSnippetsFactory;
    static std::shared_ptr<ov::snippets::op::Subgraph>
            getLoweredSubgraph(const std::shared_ptr<Model>& f,
                               const std::vector<ov::snippets::pass::Manager::PositionedPassBase>& backend_passes = {},
                               const std::shared_ptr<ov::snippets::lowered::pass::PassConfig>& lowered_pass_config =
                                    std::make_shared<ov::snippets::lowered::pass::PassConfig>(),
                               const std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered>& lowered_backend_passes = {},
                               const std::shared_ptr<ov::snippets::Generator>& generator = nullptr,
                               size_t min_parallel_work_amount = 8, size_t min_kernel_work_amount = 256,
                               const std::shared_ptr<IShapeInferSnippetsFactory>& factory = std::make_shared<IShapeInferSnippetsFactory>());
    static std::shared_ptr<ov::snippets::op::Subgraph> getTokenizedSubgraph(const std::shared_ptr<Model>& f);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
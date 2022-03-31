// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <common_test_utils/ngraph_test_utils.hpp>
#include "snippets/op/subgraph.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

using BlockedShapeVector = ngraph::snippets::op::Subgraph::BlockedShapeVector;

class DummyEmitter : public ngraph::snippets::Emitter {
public:
    // Here I pass Add to Emitter, but could be any other op, since it's ignored anyway.
    DummyEmitter() : ngraph::snippets::Emitter(std::make_shared<ov::op::v1::Add>()) {}
    void emit_code(const std::vector<size_t>&,
                   const std::vector<size_t>&,
                   const std::vector<size_t>&,
                   const std::vector<size_t>&) const override {}
    void emit_data() const override {}
};

class DummyTargetMachine : public ngraph::snippets::TargetMachine {
public:
    DummyTargetMachine();
    bool is_supported() const override { return true; }
    ngraph::snippets::code get_snippet() const override { return nullptr; }
    size_t get_lanes() const override { return 1; }
};

class DummyGenerator : public ngraph::snippets::Generator {
public:
    DummyGenerator() : ngraph::snippets::Generator(std::make_shared<DummyTargetMachine>()) {}
};

class LoweringTests : public TransformationTestsF {
protected:
    static std::shared_ptr<ngraph::snippets::op::Subgraph> getSubgraph(const std::shared_ptr<Model>& f);
    static std::shared_ptr<ngraph::snippets::op::Subgraph> getLoweredSubgraph(const std::shared_ptr<Model>& f);
    static std::shared_ptr<ngraph::snippets::op::Subgraph> getTokenizedSubgraph(const std::shared_ptr<Model>& f);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ngraph_test_utils.hpp>
#include "lowering_utils.hpp"
#include "snippets/pass/collapse_subgraph.hpp"


namespace ov {
namespace test {
namespace snippets {

DummyTargetMachine::DummyTargetMachine() {
    auto dummy_functor = [this](const std::shared_ptr<ngraph::Node>& n) {
        return std::make_shared<DummyEmitter>();
    };
    jitters[op::v0::Parameter::get_type_info_static()] = dummy_functor;
    jitters[op::v0::Constant::get_type_info_static()] = dummy_functor;
    jitters[op::v0::Result::get_type_info_static()] = dummy_functor;
    jitters[op::v1::Add::get_type_info_static()] = dummy_functor;
    jitters[op::v1::Subtract::get_type_info_static()] = dummy_functor;
    jitters[op::v1::Multiply::get_type_info_static()] = dummy_functor;
    jitters[op::v1::Multiply::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::Load::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::VectorLoad::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::ScalarLoad::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::BroadcastLoad::get_type_info_static()] = dummy_functor;

    jitters[ngraph::snippets::op::Store::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::VectorStore::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::ScalarStore::get_type_info_static()] = dummy_functor;

    jitters[ngraph::snippets::op::Scalar::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::BroadcastMove::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::Kernel::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::Tile::get_type_info_static()] = dummy_functor;
}

std::shared_ptr<ngraph::snippets::op::Subgraph> LoweringTests::getSubgraph(const std::shared_ptr<Model>& f) {
    std::shared_ptr<ngraph::snippets::op::Subgraph> subgraph;
    for (const auto &op : f->get_ops()) {
        bool is_subgraph = is_type<ngraph::snippets::op::Subgraph>(op);
        if (is_subgraph) {
            NGRAPH_CHECK(subgraph.use_count() == 0,
                         "Functions provided for lowering tests contains more than one subgraph.");
            subgraph = as_type_ptr<ngraph::snippets::op::Subgraph>(op);
        }
        NGRAPH_CHECK(is_subgraph ||
                     is_type<ov::op::v0::Parameter>(op) ||
                     is_type<ov::op::v0::Constant>(op) ||
                     is_type<ov::op::v0::Result>(op),
                     "Functions provided for lowering tests is not fully tokenizable");
    }
    return subgraph;
}

std::shared_ptr<ngraph::snippets::op::Subgraph> LoweringTests::getLoweredSubgraph(const std::shared_ptr<Model> &f) {
    auto subgraph = getTokenizedSubgraph(f);
    subgraph->set_generator(std::make_shared<DummyGenerator>());
    subgraph->generate();
    return subgraph;
}

std::shared_ptr<ngraph::snippets::op::Subgraph> LoweringTests::getTokenizedSubgraph(const std::shared_ptr<Model> &f) {
    // Perform tokenization
    ngraph::pass::Manager m;
    m.register_pass<ngraph::snippets::pass::EnumerateNodes>();
    m.register_pass<ngraph::snippets::pass::TokenizeSnippets>();
    m.run_passes(f);
    // Perform lowering
    return getSubgraph(f);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
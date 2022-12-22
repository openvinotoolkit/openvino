// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ngraph_test_utils.hpp>
#include "lowering_utils.hpp"
#include "snippets/pass/tokenization.hpp"


namespace ov {
namespace test {
namespace snippets {

DummyTargetMachine::DummyTargetMachine() {
    auto dummy_functor = ngraph::snippets::jitters_value {
        [](const std::shared_ptr<ngraph::Node>& n) { return std::make_shared<DummyEmitter>(); },
        {}
    };

    jitters[op::v0::Parameter::get_type_info_static()] = dummy_functor;
    jitters[op::v0::Constant::get_type_info_static()] = dummy_functor;
    jitters[op::v0::Result::get_type_info_static()] = dummy_functor;
    jitters[op::v1::Add::get_type_info_static()] = dummy_functor;
    jitters[op::v1::Subtract::get_type_info_static()] = dummy_functor;
    jitters[op::v1::Multiply::get_type_info_static()] = dummy_functor;
    jitters[op::v1::Divide::get_type_info_static()] = dummy_functor;
    jitters[op::v1::Maximum::get_type_info_static()] = dummy_functor;
    jitters[op::v0::Exp::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::PowerStatic::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::HorizonMax::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::HorizonSum::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::Load::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::BroadcastLoad::get_type_info_static()] = dummy_functor;

    jitters[ngraph::snippets::op::Store::get_type_info_static()] = dummy_functor;

    jitters[ngraph::snippets::op::Scalar::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::BroadcastMove::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::Kernel::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::LoopBegin::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::LoopEnd::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::Brgemm::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::Buffer::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::VectorBuffer::get_type_info_static()] = dummy_functor;
    jitters[ngraph::snippets::op::Fill::get_type_info_static()] = dummy_functor;
}

LoweringTests::LoweringTests() : TransformationTestsF() {
    // external subgraph input shape and internal parameters shapes
    // might differ due to the blocked layout
    // so input & output descriptors shouldn't be checked
    comparator.disable(FunctionsComparator::CmpValues::SUBGRAPH_DESCRIPTORS);
}

void LoweringTests::SetUp() {
    manager.register_pass<ngraph::pass::InitNodeInfo>();
}

void LoweringTests::TearDown() {
    auto cloned_function = ngraph::clone_function(*function);
    if (!function_ref) {
        function_ref = cloned_function;
    }
    manager.run_passes(function);
        ASSERT_NO_THROW(check_rt_info(function));

    if (comparator.should_compare(FunctionsComparator::ACCURACY)) {
        auto acc_comparator = FunctionsComparator::no_default();
        acc_comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
        auto res = acc_comparator.compare(function, cloned_function);
        ASSERT_TRUE(res.valid) << res.message;
        comparator.disable(FunctionsComparator::CmpValues::ACCURACY);
    }
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
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

std::shared_ptr<ngraph::snippets::op::Subgraph> LoweringTests::getLoweredSubgraph(const std::shared_ptr<Model> &f,
                                                                                  const ov::PartialShape& master_shape) {
    auto subgraph = getTokenizedSubgraph(f);
    subgraph->set_generator(std::make_shared<DummyGenerator>());
    subgraph->set_master_shape(master_shape);
    const auto& body = subgraph->body_ptr();
    auto& body_rt_info = body->get_rt_info();
    // todo: insertLoops pass requires body_rt_info["PluginShapesOverride"] and subgraph->set_tile_rank to work normally
    //  consider revising snippets-plugin shape and scheduling communication
    std::vector<std::vector<size_t>> new_shapes;
    for (const auto& p : body->get_parameters()) {
        const auto pshape = p->get_output_partial_shape(0);
        if (pshape.is_dynamic())
            IE_THROW() << "getLoweredSubgraph supports only static shapes";
        new_shapes.push_back(pshape.get_shape());
    }
    for (const auto& r : body->get_results()) {
        const auto pshape = r->get_input_partial_shape(0);
        if (pshape.is_dynamic())
            IE_THROW() << "getLoweredSubgraph supports only static shapes";
        new_shapes.push_back(pshape.get_shape());
    }
    body_rt_info["PluginShapesOverride"] = new_shapes;
    subgraph->set_tile_rank(2);
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
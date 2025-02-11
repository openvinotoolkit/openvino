// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_test_utils.hpp>
#include "lowering_utils.hpp"
#include "utils.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/gn_tokenization.hpp"
#include "snippets/lowered/expression.hpp"


namespace ov {
namespace test {
namespace snippets {

DummyTargetMachine::DummyTargetMachine(const std::vector<ov::Node::type_info_t>&custom_opset)
  : TargetMachine(std::make_shared<DummyRuntimeConfigurator>()) {
    auto dummy_functor = ov::snippets::jitters_value {
        [](const ov::snippets::lowered::ExpressionPtr& n) { return std::make_shared<DummyEmitter>(); },
        [](const std::shared_ptr<ov::Node>& n) { return std::set<std::vector<element::Type>>{};}
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
    jitters[op::v0::Sqrt::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::PowerStatic::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::HorizonMax::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::HorizonSum::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::Load::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::BroadcastLoad::get_type_info_static()] = dummy_functor;

    jitters[ov::snippets::op::Store::get_type_info_static()] = dummy_functor;

    jitters[ov::snippets::op::Scalar::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::BroadcastMove::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::KernelDynamic::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::KernelStatic::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::LoopBegin::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::LoopEnd::get_type_info_static()] = dummy_functor;
#ifdef SNIPPETS_DEBUG_CAPS
    jitters[ov::snippets::op::PerfCountBegin::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::PerfCountEnd::get_type_info_static()] = dummy_functor;
#endif
    jitters[ov::snippets::op::Brgemm::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::Buffer::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::VectorBuffer::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::Fill::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::ReduceMax::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::ReduceSum::get_type_info_static()] = dummy_functor;
    jitters[ov::snippets::op::Reshape::get_type_info_static()] = dummy_functor;

    for (const auto& elem : custom_opset) {
        jitters[elem] = dummy_functor;
    }
}

std::vector<ov::snippets::Reg> DummyTargetMachine::get_abi_arg_regs() const {
    const auto num_abi_regs = 4;
    std::vector<ov::snippets::Reg> reg_pool;
    reg_pool.reserve(num_abi_regs);
    for (size_t i = 0; i < num_abi_regs; i++)
        reg_pool.emplace_back(ov::snippets::RegType::gpr, i);
    return reg_pool;
}

std::vector<ov::snippets::Reg> DummyTargetMachine::get_gp_reg_pool() const {
    const auto num_gp_regs = 16;
    std::vector<ov::snippets::Reg> reg_pool;
    reg_pool.reserve(num_gp_regs);
    for (size_t i = 0; i < num_gp_regs; i++)
        reg_pool.emplace_back(ov::snippets::RegType::gpr, i);
    return reg_pool;
}

std::vector<ov::snippets::Reg> DummyTargetMachine::get_vec_reg_pool() const {
    const auto num_vec_regs = 16;
    std::vector<ov::snippets::Reg> reg_pool;
    reg_pool.reserve(num_vec_regs);
    for (size_t i = 0; i < num_vec_regs; i++)
        reg_pool.emplace_back(ov::snippets::RegType::vec, i);
    return reg_pool;
}

LoweringTests::LoweringTests() : TransformationTestsF() {
    // external subgraph input shape and internal parameters shapes
    // might differ due to the blocked layout
    // so input & output descriptors shouldn't be checked
    comparator.disable(FunctionsComparator::CmpValues::SUBGRAPH_DESCRIPTORS);
}

void LoweringTests::SetUp() {
    manager.register_pass<ov::pass::InitNodeInfo>();
}

void LoweringTests::TearDown() {
    ASSERT_TRUE(model);
    auto cloned_model = model->clone();
    if (!model_ref) {
        model_ref = cloned_model;
    }
    manager.run_passes(model);
    OV_ASSERT_NO_THROW(check_rt_info(model));

    if (comparator.should_compare(FunctionsComparator::ACCURACY)) {
        auto acc_comparator = FunctionsComparator::no_default();
        acc_comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
        auto res = acc_comparator.compare(model, cloned_model);
        ASSERT_TRUE(res.valid) << res.message;
        comparator.disable(FunctionsComparator::CmpValues::ACCURACY);
    }
    auto res = comparator.compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

std::shared_ptr<ov::snippets::op::Subgraph> LoweringTests::getSubgraph(const std::shared_ptr<Model>& f) {
    std::shared_ptr<ov::snippets::op::Subgraph> subgraph;
    for (const auto& op : f->get_ops()) {
        bool is_subgraph = is_type<ov::snippets::op::Subgraph>(op);
        if (is_subgraph) {
            OPENVINO_ASSERT(subgraph.use_count() == 0,
                            "Functions provided for lowering tests contains more than one subgraph.");
            subgraph = as_type_ptr<ov::snippets::op::Subgraph>(op);
        }
        OPENVINO_ASSERT(is_subgraph ||
                        is_type<ov::op::v0::Parameter>(op) ||
                        is_type<ov::op::v0::Constant>(op) ||
                        is_type<ov::op::v0::Result>(op),
                     "Models provided for lowering tests is not fully tokenizable");
    }
    return subgraph;
}

std::shared_ptr<ov::snippets::op::Subgraph>
        LoweringTests::getLoweredSubgraph(const std::shared_ptr<Model> &f,
                                          const std::vector<ov::snippets::pass::Manager::PositionedPassBase>& backend_passes,
                                          const std::shared_ptr<ov::snippets::lowered::pass::PassConfig>& lowered_pass_config,
                                          const std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered>& lowered_backend_passes,
                                          const std::shared_ptr<ov::snippets::Generator>& generator,
                                          size_t min_parallel_work_amount, size_t min_kernel_work_amount,
                                          const std::shared_ptr<IShapeInferSnippetsFactory>& factory) {
    auto subgraph = getTokenizedSubgraph(f);
    subgraph->set_generator(generator == nullptr ? std::make_shared<DummyGenerator>() : generator);
    subgraph->set_tile_rank(2);
    // Note: lowered_pipeline would have no effect on subgraph body, since it's applied on linear IR
    subgraph->generate({}, {}, {}, backend_passes, lowered_pass_config, lowered_backend_passes, min_parallel_work_amount, min_kernel_work_amount, factory);
    return subgraph;
}

std::shared_ptr<ov::snippets::op::Subgraph> LoweringTests::getTokenizedSubgraph(const std::shared_ptr<Model> &f) {
    // Perform tokenization
    ov::pass::Manager m;
    ov::snippets::pass::SnippetsTokenization::Config config = get_default_tokenization_config();
    m.register_pass<ov::snippets::pass::EnumerateNodes>();
    m.register_pass<ov::snippets::pass::TokenizeGNSnippets>();
    m.register_pass<ov::snippets::pass::TokenizeSnippets>(config);
    m.run_passes(f);
    // Perform lowering
    return getSubgraph(f);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/loop.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/moe_router_fused.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/les_context.hpp"
#include "program_node.h"
#include "runtime/ocl/ocl_engine.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "runtime/ocl/ocl_event.hpp"

#include <chrono>
#include <future>

namespace ov::test {

using ::tests::get_test_default_config;
using ::tests::get_test_engine;

TEST(moe_offload_property_test, execution_config_roundtrip) {
    auto config = get_test_default_config(get_test_engine());

    ASSERT_EQ(config.get_offload_ratio(), 0U);

    config.set_property(ov::intel_gpu::offload_ratio(37));

    ASSERT_EQ(config.get_offload_ratio(), 37U);
}

TEST(moe_offload_property_test, default_value_is_zero) {
    auto config = get_test_default_config(get_test_engine());
    ASSERT_EQ(config.get_offload_ratio(), 0U);
}

TEST(moe_offload_property_test, set_and_get_various_values) {
    auto config = get_test_default_config(get_test_engine());

    config.set_property(ov::intel_gpu::offload_ratio(1));
    ASSERT_EQ(config.get_offload_ratio(), 1U);

    config.set_property(ov::intel_gpu::offload_ratio(50));
    ASSERT_EQ(config.get_offload_ratio(), 50U);

    config.set_property(ov::intel_gpu::offload_ratio(100));
    ASSERT_EQ(config.get_offload_ratio(), 100U);
    EXPECT_ANY_THROW(config.set_property(ov::intel_gpu::offload_ratio(101)));
}

TEST(moe_offload_property_test, set_back_to_zero_disables) {
    auto config = get_test_default_config(get_test_engine());

    config.set_property(ov::intel_gpu::offload_ratio(37));
    ASSERT_EQ(config.get_offload_ratio(), 37U);

    config.set_property(ov::intel_gpu::offload_ratio(0));
    ASSERT_EQ(config.get_offload_ratio(), 0U);
}

TEST(moe_les_config, disabled_by_default) {
    cldnn::ExecutionConfig config;
    EXPECT_FALSE(config.get_enable_les());
    EXPECT_EQ(config.get_les_queue_capacity(), 32U);
}

TEST(moe_les_config, roundtrip_and_clone) {
    cldnn::ExecutionConfig config;
    config.set_property({ov::intel_gpu::enable_les(true), ov::intel_gpu::les_queue_capacity(7)});
    auto cloned = config.clone();
    EXPECT_TRUE(cloned.get_enable_les());
    EXPECT_EQ(cloned.get_les_queue_capacity(), 7U);
    config.set_property(ov::intel_gpu::enable_les(false));
    EXPECT_FALSE(config.get_enable_les());
    EXPECT_TRUE(cloned.get_enable_les());
}

TEST(moe_les_config, rejects_zero_capacity) {
    cldnn::ExecutionConfig config;
    EXPECT_ANY_THROW(config.set_property(ov::intel_gpu::les_queue_capacity(0)));
}

TEST(moe_les_config, rejects_excessive_capacity) {
    cldnn::ExecutionConfig config;
    EXPECT_ANY_THROW(config.set_property(ov::intel_gpu::les_queue_capacity(1025)));
}

TEST(moe_les_config, accepts_capacity_boundaries) {
    cldnn::ExecutionConfig config;
    config.set_property(ov::intel_gpu::les_queue_capacity(1));
    EXPECT_EQ(config.get_les_queue_capacity(), 1U);
    config.set_property(ov::intel_gpu::les_queue_capacity(1024));
    EXPECT_EQ(config.get_les_queue_capacity(), 1024U);
}

TEST(moe_les_network, non_moe_graph_creates_no_context_and_preserves_output) {
    auto& engine = get_test_engine();
    cldnn::layout input_layout{cldnn::data_types::f32, cldnn::format::bfyx, {1, 1, 1, 2}};
    auto input = engine.allocate_memory(input_layout);
    tests::set_values(input, std::vector<float>{-2.0f, 3.0f});
    cldnn::topology topology;
    topology.add(cldnn::input_layout("input", input_layout));
    topology.add(cldnn::activation("output", cldnn::input_info("input"), cldnn::activation_func::relu));
    for (size_t ratio : {0U, 50U}) {
        auto config = get_test_default_config(engine);
        config.set_property({ov::intel_gpu::enable_les(true), ov::intel_gpu::offload_ratio(ratio)});
        cldnn::network network(engine, topology, config);
        EXPECT_EQ(network.get_les_context(), nullptr);
        network.set_input_data("input", input);
        auto output = network.execute().at("output").get_memory();
        cldnn::mem_lock<float, cldnn::mem_lock_type::read> values(output, network.get_stream());
        EXPECT_FLOAT_EQ(values[0], 0.0f);
        EXPECT_FLOAT_EQ(values[1], 3.0f);
        EXPECT_EQ(network.get_les_epoch(), 0U);
    }
}

class moe_les_chain : public testing::Test {
protected:
    std::unique_ptr<cldnn::program> make_program(bool enabled = true) {
        auto config = get_test_default_config(get_test_engine());
        config.set_property(ov::intel_gpu::enable_les(enabled));
        return std::make_unique<cldnn::program>(get_test_engine(), config);
    }

    void add(cldnn::program& program, const std::string& id,
             const std::vector<std::string>& dependencies = {}, bool moe = false) {
        std::vector<cldnn::input_info> inputs;
        for (const auto& dependency : dependencies) {
            inputs.emplace_back(dependency);
        }
        std::shared_ptr<cldnn::primitive> primitive;
        if (moe) {
            primitive = std::make_shared<cldnn::moe_3gemm_fused_compressed>(id, inputs, cldnn::MOECompressed::Config{});
        } else if (inputs.empty()) {
            cldnn::layout output_layout{ov::PartialShape{1, 1, 2}, cldnn::data_types::f32, cldnn::format::bfyx};
            primitive = std::make_shared<cldnn::input_layout>(id, output_layout);
            auto& node = program.get_or_create(primitive);
            node.set_output_layout(output_layout, false);
            return;
        } else {
            primitive = std::make_shared<cldnn::eltwise>(id, inputs, cldnn::eltwise_mode::sum);
        }
        auto& node = program.get_or_create(primitive);
        for (const auto& dependency : dependencies) {
            program.add_connection(program.get_node(dependency), node);
        }
        if (moe) {
            cldnn::layout output_layout{ov::PartialShape{1, 1, 2}, cldnn::data_types::f32, cldnn::format::bfyx};
            node.set_output_layout(output_layout, false);
        }
    }

    cldnn::program::les_moe_link find(const cldnn::program& program, const std::string& target) {
        for (const auto& link : program.get_les_moe_links()) {
            if (link.target_moe == target) {
                return link;
            }
        }
        ADD_FAILURE() << "Missing LES descriptor for " << target;
        return {};
    }

    std::unique_ptr<cldnn::program> router_program(bool sigmoid,
                                                   bool variable_weights = false,
                                                   bool projection_bias = false,
                                                   bool wrong_router_ports = false,
                                                   bool multiple_tokens = false,
                                                   bool dynamic_tokens = false) {
        auto program = make_program();
        cldnn::layout hidden_layout{ov::PartialShape{1, 1, 2}, cldnn::data_types::f16, cldnn::format::bfyx};
        if (multiple_tokens) {
            hidden_layout = cldnn::layout{ov::PartialShape{1, 2, 2}, cldnn::data_types::f16, cldnn::format::bfyx};
        }
        if (dynamic_tokens) {
            hidden_layout = cldnn::layout{ov::PartialShape{1, -1, 2}, cldnn::data_types::f16, cldnn::format::bfyx};
        }
        auto& hidden = program->get_or_create(std::make_shared<cldnn::input_layout>("hidden", hidden_layout));
        auto hidden_output_layout = hidden_layout;
        hidden.set_output_layout(hidden_output_layout, false);
        add(*program, "source", {"hidden"}, true);
        auto source_output_layout = hidden_layout;
        program->get_node("source").set_output_layout(source_output_layout, false);
        const cldnn::layout weights_layout{ov::PartialShape{4, 2}, cldnn::data_types::f16, cldnn::format::bfyx};
        if (variable_weights) {
            program->get_or_create(std::make_shared<cldnn::input_layout>("router_weights", weights_layout));
        } else {
            program->get_or_create(std::make_shared<cldnn::data>("router_weights", get_test_engine().allocate_memory(weights_layout)));
        }
        std::string projection_bias_id;
        if (projection_bias) {
            projection_bias_id = "projection_bias";
            auto bias_memory = get_test_engine().allocate_memory(cldnn::layout{
                ov::PartialShape{4}, cldnn::data_types::f16, cldnn::format::bfyx});
            program->get_or_create(std::make_shared<cldnn::data>(projection_bias_id, bias_memory));
        }
        auto projection = std::make_shared<cldnn::fully_connected>("projection", cldnn::input_info("source"),
                                                                  "router_weights", projection_bias_id, 3);
        auto& projection_node = program->get_or_create(projection);
        program->add_connection(program->get_node("source"), projection_node);
        program->add_connection(program->get_node("router_weights"), projection_node);
        if (projection_bias) {
            program->add_connection(program->get_node(projection_bias_id), projection_node);
        }
        cldnn::MoERouterFused::Config config{};
        config.num_expert = 4;
        config.top_k = 2;
        config.routing_type = sigmoid ? cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS
                                      : cldnn::MoERouterFused::RoutingType::SOFTMAX;
        std::vector<cldnn::input_info> inputs{cldnn::input_info("projection")};
        if (sigmoid) {
            for (const auto& parameter : std::vector<std::pair<std::string, size_t>>{{"bias", 4}, {"eps", 1}}) {
                auto memory = get_test_engine().allocate_memory(cldnn::layout{
                    ov::PartialShape{static_cast<int64_t>(parameter.second)}, cldnn::data_types::f16, cldnn::format::bfyx});
                program->get_or_create(std::make_shared<cldnn::data>(parameter.first, memory));
                inputs.emplace_back(parameter.first);
            }
        }
        auto& router = program->get_or_create(std::make_shared<cldnn::moe_router_fused>("router", inputs, config));
        for (const auto& input : inputs) {
            program->add_connection(program->get_node(input.pid), router);
        }
        auto& target = program->get_or_create(std::make_shared<cldnn::moe_3gemm_fused_compressed>("target",
            std::vector<cldnn::input_info>{cldnn::input_info("source"), cldnn::input_info("router", 0),
                                          cldnn::input_info("router", wrong_router_ports ? 0 : 1)},
            cldnn::MOECompressed::Config{}));
        program->add_connection(program->get_node("source"), target);
        program->add_connection(router, target, 0);
        program->add_connection(router, target, wrong_router_ports ? 0 : 1);
        return program;
    }
};

TEST_F(moe_les_chain, constant_projection_and_router_parameters_are_qualified) {
    for (bool sigmoid : {false, true}) {
        auto program = router_program(sigmoid);
        auto links = program->get_les_router_links();
        ASSERT_EQ(links.size(), 2U);
        const auto& link = links.back();
        EXPECT_EQ(link.status, cldnn::program::les_router_status::supported);
        EXPECT_EQ(link.router, "router");
        EXPECT_EQ(link.projection, "projection");
        EXPECT_EQ(link.weights, "router_weights");
        EXPECT_EQ(link.experts, 4U);
        EXPECT_EQ(link.top_k, 2U);
        EXPECT_EQ(link.sigmoid_bias, sigmoid);
        EXPECT_EQ(link.routing_bias, sigmoid ? "bias" : "");
        EXPECT_EQ(link.routing_eps, sigmoid ? "eps" : "");
    }
}

TEST_F(moe_les_chain, constant_projection_bias_is_described) {
    auto program = router_program(false, false, true);
    const auto& link = program->get_les_router_links().back();
    EXPECT_EQ(link.status, cldnn::program::les_router_status::supported);
    EXPECT_EQ(link.projection_bias, "projection_bias");
}

TEST_F(moe_les_chain, router_hidden_reshape_preserves_source_identity) {
    auto program = router_program(false);
    auto& source = program->get_node("source");
    auto& projection = program->get_node("projection");
    auto& view = program->get_or_create(std::make_shared<cldnn::reshape>(
        "router_hidden_view", cldnn::input_info("source"), cldnn::tensor{1, 1, 2, 1}));
    program->add_connection(source, view);
    projection.replace_dependency(0, view);
    EXPECT_EQ(program->get_les_router_links().back().status, cldnn::program::les_router_status::supported);
    view.replace_dependency(0, program->get_node("hidden"));
    EXPECT_EQ(program->get_les_router_links().back().status, cldnn::program::les_router_status::unsupported_projection);
}

TEST_F(moe_les_chain, multi_token_proxy_is_rejected) {
    auto program = router_program(false, false, false, false, true);
    EXPECT_EQ(program->get_les_router_links().back().status, cldnn::program::les_router_status::unsupported_layout);
}

TEST_F(moe_les_chain, dynamic_tokens_with_static_hidden_width_are_qualified) {
    auto program = router_program(false, false, false, false, false, true);
    EXPECT_EQ(program->get_les_router_links().back().status, cldnn::program::les_router_status::supported);
}

TEST_F(moe_les_chain, dynamic_hidden_width_is_rejected) {
    auto program = router_program(false, false, false, false, false, true);
    cldnn::layout hidden_layout{ov::PartialShape{1, -1, -1}, cldnn::data_types::f16, cldnn::format::bfyx};
    program->get_node("hidden").set_output_layout(hidden_layout, false);
    program->get_node("source").set_output_layout(hidden_layout, false);
    EXPECT_EQ(program->get_les_router_links().back().status, cldnn::program::les_router_status::unsupported_layout);
}

TEST_F(moe_les_chain, incorrect_router_output_port_is_rejected) {
    auto program = router_program(false, false, false, true);
    EXPECT_EQ(program->get_les_router_links().back().status, cldnn::program::les_router_status::unsupported_router);
}

TEST_F(moe_les_chain, variable_router_weights_are_rejected) {
    auto program = router_program(false, true);
    EXPECT_EQ(program->get_les_router_links().back().status, cldnn::program::les_router_status::non_constant_parameter);
}

TEST_F(moe_les_chain, different_topk_producers_are_not_a_router_descriptor) {
    auto program = make_program();
    add(*program, "hidden");
    add(*program, "source", {"hidden"}, true);
    add(*program, "weights");
    add(*program, "indices");
    add(*program, "target", {"source", "weights", "indices"}, true);
    auto links = program->get_les_router_links();
    ASSERT_EQ(links.size(), 2U);
    EXPECT_EQ(links.back().status, cldnn::program::les_router_status::unsupported_router);
    EXPECT_TRUE(links.back().router.empty());
}

TEST_F(moe_les_chain, dense_gap_uses_nearest_moe_and_stable_ids) {
    auto program = make_program();
    add(*program, "hidden");
    add(*program, "first", {"hidden"}, true);
    add(*program, "dense", {"first", "hidden"});
    add(*program, "second", {"dense"}, true);
    add(*program, "residual", {"second", "dense"});
    add(*program, "third", {"residual"}, true);
    EXPECT_EQ(find(*program, "first").status, cldnn::program::les_link_status::no_predecessor);
    auto second = find(*program, "second");
    EXPECT_EQ(second.status, cldnn::program::les_link_status::candidate);
    EXPECT_EQ(second.source_moe, "first");
    EXPECT_EQ(second.proxy_hidden, "hidden");
    EXPECT_EQ(second.target_hidden, "dense");
    EXPECT_EQ(find(*program, "third").status, cldnn::program::les_link_status::candidate);
    EXPECT_EQ(find(*program, "third").source_moe, "second");
    EXPECT_EQ(program->get_les_moe_links().size(), 3U);
}

TEST_F(moe_les_chain, independent_chains_are_not_connected_by_order) {
    auto program = make_program();
    add(*program, "hidden");
    add(*program, "a", {"hidden"}, true);
    add(*program, "b", {"hidden"}, true);
    add(*program, "c", {"b"}, true);
    add(*program, "d", {"a"}, true);
    EXPECT_EQ(find(*program, "c").source_moe, "b");
    EXPECT_EQ(find(*program, "d").source_moe, "a");
}

TEST_F(moe_les_chain, merge_rejects_multiple_nearest_predecessors) {
    auto program = make_program();
    add(*program, "hidden");
    add(*program, "left", {"hidden"}, true);
    add(*program, "right", {"hidden"}, true);
    add(*program, "merge", {"left", "right"});
    add(*program, "target", {"merge"}, true);
    auto link = find(*program, "target");
    EXPECT_EQ(link.status, cldnn::program::les_link_status::ambiguous_predecessor);
    EXPECT_TRUE(link.source_moe.empty());
}

TEST_F(moe_les_chain, fork_rejects_both_successors) {
    auto program = make_program();
    add(*program, "hidden");
    add(*program, "source", {"hidden"}, true);
    add(*program, "left", {"source"}, true);
    add(*program, "right", {"source"}, true);
    EXPECT_EQ(find(*program, "left").status, cldnn::program::les_link_status::branched_predecessor);
    EXPECT_EQ(find(*program, "right").status, cldnn::program::les_link_status::branched_predecessor);
}

TEST_F(moe_les_chain, router_side_input_does_not_create_hidden_chain) {
    auto program = make_program();
    add(*program, "hidden");
    add(*program, "unrelated", {"hidden"}, true);
    add(*program, "target", {"hidden", "unrelated"}, true);
    EXPECT_EQ(find(*program, "target").status, cldnn::program::les_link_status::no_predecessor);
}

TEST_F(moe_les_chain, control_flow_is_not_traversed) {
    auto program = make_program();
    add(*program, "hidden");
    add(*program, "source", {"hidden"}, true);
    auto boundary = std::make_shared<cldnn::loop>();
    boundary->id = "boundary";
    auto& node = program->get_or_create(boundary);
    program->add_connection(program->get_node("source"), node);
    add(*program, "target", {"boundary"}, true);
    auto link = find(*program, "target");
    EXPECT_EQ(link.status, cldnn::program::les_link_status::unsupported_topology);
    EXPECT_TRUE(link.source_moe.empty());
}

TEST_F(moe_les_chain, missing_hidden_is_explicitly_unsupported) {
    auto program = make_program();
    add(*program, "source", {}, true);
    add(*program, "target", {"source"}, true);
    EXPECT_EQ(find(*program, "source").status, cldnn::program::les_link_status::unsupported_topology);
    EXPECT_EQ(find(*program, "target").status, cldnn::program::les_link_status::unsupported_topology);
}

TEST_F(moe_les_chain, disabled_and_non_moe_programs_have_no_descriptors) {
    auto disabled = make_program(false);
    add(*disabled, "hidden");
    add(*disabled, "moe", {"hidden"}, true);
    EXPECT_TRUE(disabled->get_les_moe_links().empty());
    auto plain = make_program();
    add(*plain, "hidden");
    add(*plain, "dense", {"hidden"});
    EXPECT_TRUE(plain->get_les_moe_links().empty());
}

TEST(moe_les_network, transfer_lane_is_retained_until_active_upload_finishes) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::queue_type(cldnn::QueueTypes::in_order));
    auto transfer = engine.create_stream(config);
    std::weak_ptr<cldnn::stream> lifetime = transfer;
    auto context = std::make_unique<cldnn::les_context>(2, std::move(transfer));
    auto memory = engine.allocate_memory(cldnn::layout{cldnn::data_types::f32, cldnn::format::bfyx, {1, 1, 1, 2}});
    cl::UserEvent gate(dynamic_cast<cldnn::ocl::ocl_engine&>(engine).get_cl_context());
    std::promise<void> submitted;
    auto ready = submitted.get_future();
    auto ticket = context->submit(context->begin_epoch(), [&, lane = context->get_transfer_stream(), memory] {
        std::vector<float> source{2.0f, 3.0f};
        dynamic_cast<cldnn::ocl::ocl_stream&>(*lane).enqueue_external_barrier({std::make_shared<cldnn::ocl::ocl_event>(gate)});
        auto uploaded = memory->copy_from(*lane, source.data(), false);
        lane->flush();
        submitted.set_value();
        uploaded->wait();
    });
    auto status = ready.wait_for(std::chrono::seconds(5));
    if (status != std::future_status::ready) {
        gate.setStatus(CL_COMPLETE);
        ASSERT_EQ(status, std::future_status::ready);
    }
    auto destroyed = std::async(std::launch::async, [owned = std::move(context)]() mutable { owned.reset(); });
    auto blocked = destroyed.wait_for(std::chrono::milliseconds(20));
    gate.setStatus(CL_COMPLETE);
    EXPECT_EQ(blocked, std::future_status::timeout);
    ASSERT_EQ(destroyed.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    destroyed.get();
    EXPECT_THROW(ticket.completion.get(), cldnn::les_task_cancelled);
    EXPECT_TRUE(lifetime.expired());
    cldnn::mem_lock<float, cldnn::mem_lock_type::read> values(memory, tests::get_test_stream());
    EXPECT_FLOAT_EQ(values[0], 2.0f);
    EXPECT_FLOAT_EQ(values[1], 3.0f);
}

}  // namespace ov::test
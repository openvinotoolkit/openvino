// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/broadcast.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/slice_scatter.hpp"
#include "primitive_inst.h"

#include "runtime/ocl/ocl_event.hpp"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(network_test, model_with_scalar_input_is_not_dynamic) {
    auto& engine = get_test_engine();
    ov::PartialShape input_shape = {};
    layout in_layout{input_shape, data_types::f32, format::bfyx};

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(broadcast("output", input_info("input"), {1, 2}, ov::AxisSet{}));

    network net(engine, topology);

    ASSERT_FALSE(net.is_dynamic());
}

TEST(network_test, model_with_empty_input_is_not_dynamic) {
    auto& engine = get_test_engine();
    ov::PartialShape input_shape = {1, 0};
    layout in_layout{input_shape, data_types::f32, format::bfyx};
    auto const_mem = engine.allocate_memory({{1, 2}, data_types::f32, format::bfyx});

    topology topology;
    topology.add(input_layout("input0", in_layout));
    topology.add(data("input1", const_mem));
    topology.add(concatenation("output", { input_info("input0"), input_info("input1") }, 1));

    network net(engine, topology, {ov::intel_gpu::allow_new_shape_infer(true)});

    ASSERT_FALSE(net.is_dynamic());
}

TEST(network_test, model_with_dynamic_input_is_dynamic) {
    auto& engine = get_test_engine();
    ov::PartialShape input_shape = {1, -1};
    layout in_layout{input_shape, data_types::f32, format::bfyx};
    auto const_mem = engine.allocate_memory({{1, 2}, data_types::f32, format::bfyx});

    topology topology;
    topology.add(input_layout("input0", in_layout));
    topology.add(data("input1", const_mem));
    topology.add(concatenation("output", { input_info("input0"), input_info("input1") }, 1));

    network net(engine, topology, {ov::intel_gpu::allow_new_shape_infer(true)});

    ASSERT_TRUE(net.is_dynamic());
}

TEST(network_test, has_proper_event_for_in_order_queue) {
    auto& engine = get_test_engine();
    layout in_layout{{1, 2, 2, 4}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory(in_layout);
    auto const_mem = engine.allocate_memory({{1, 2, 2, 4}, data_types::f32, format::bfyx});

    topology topology;
    topology.add(input_layout("input1", in_layout));
    topology.add(data("input2", const_mem));
    topology.add(activation("activation1", input_info("input1"), activation_func::clamp, {-10.f, 10.f}));
    topology.add(concatenation("concat", { input_info("activation1"), input_info("input2") }, 1));
    topology.add(reorder("reorder", input_info("concat"), in_layout));
    topology.add(activation("activation2", input_info("concat"), activation_func::relu));

    auto impl_desc = ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::cpu};
    auto impl_forcing_map = ov::intel_gpu::ImplForcingMap{{"activation2", impl_desc}};

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(impl_forcing_map));

    network net(engine, topology, config);

    net.set_input_data("input1", input_mem);
    net.execute();

    ASSERT_TRUE(net.has_event("activation1"));
    ASSERT_TRUE(net.has_event("concat"));
    ASSERT_TRUE(net.has_event("reorder"));
    ASSERT_TRUE(net.has_event("activation2"));

    auto concat_ev = net.get_primitive_event("concat");
    auto reorder_ev = net.get_primitive_event("reorder");
    auto activation_ev = net.get_primitive_event("activation2");

    OV_ASSERT_NO_THROW(downcast<ocl::ocl_base_event>(concat_ev.get()));
    OV_ASSERT_NO_THROW(downcast<ocl::ocl_base_event>(reorder_ev.get()));
    OV_ASSERT_NO_THROW(downcast<ocl::ocl_base_event>(activation_ev.get()));

    // Check if we have real underlying OpenCL events
    ASSERT_TRUE(downcast<ocl::ocl_base_event>(concat_ev.get())->get().get() != nullptr);
    ASSERT_TRUE(downcast<ocl::ocl_base_event>(reorder_ev.get())->get().get() != nullptr);
    ASSERT_TRUE(downcast<ocl::ocl_base_event>(activation_ev.get())->get().get() != nullptr);
}

TEST(network_test, has_proper_event_for_in_order_queue_optimized_out) {
    auto& engine = get_test_engine();
    layout in_layout{{1, 2, 2, 4}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory(in_layout);
    auto const_mem = engine.allocate_memory({{1, 2, 2, 4}, data_types::f32, format::bfyx});

    topology topology;
    topology.add(input_layout("input1", in_layout));
    topology.add(data("input2", const_mem));
    topology.add(concatenation("concat", { input_info("input1"), input_info("input2") }, 1));
    topology.add(reshape("reshape", input_info("concat"), false, {1, 2, 4, 4}, {1, 2, 4, 4}));
    topology.add(reorder("reorder", input_info("reshape"), in_layout));
    topology.add(activation("activation", input_info("reshape"), activation_func::relu));

    auto impl_desc = ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::cpu};
    auto impl_forcing_map = ov::intel_gpu::ImplForcingMap{{"activation", impl_desc}};

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(impl_forcing_map));

    network net(engine, topology, config);

    net.set_input_data("input1", input_mem);
    net.execute();

    // reshape is optimized out with need_completion_event=true. So it doesn't have event.
    ASSERT_TRUE(net.has_event("concat"));
    ASSERT_TRUE(net.has_event("reorder"));
    ASSERT_TRUE(net.has_event("activation"));

    auto concat_ev = net.get_primitive_event("concat");
    auto reorder_ev = net.get_primitive_event("reorder");
    auto activation_ev = net.get_primitive_event("activation");

    OV_ASSERT_NO_THROW(downcast<ocl::ocl_base_event>(concat_ev.get()));
    OV_ASSERT_NO_THROW(downcast<ocl::ocl_base_event>(reorder_ev.get()));
    OV_ASSERT_NO_THROW(downcast<ocl::ocl_base_event>(activation_ev.get()));

    // Check if we have real underlying OpenCL events
    ASSERT_TRUE(downcast<ocl::ocl_base_event>(concat_ev.get())->get().get() != nullptr);
    ASSERT_TRUE(downcast<ocl::ocl_base_event>(reorder_ev.get())->get().get() != nullptr);
    ASSERT_TRUE(downcast<ocl::ocl_base_event>(activation_ev.get())->get().get() != nullptr);
}

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(network_test, has_proper_event_for_in_order_queue_onednn) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    layout in_layout{{1, 16, 2, 4}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory(in_layout);
    auto weights = engine.allocate_memory({{16, 16, 1, 1}, data_types::f32, format::bfyx});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weights", weights));
    topology.add(convolution("conv", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(activation("activation", input_info("conv"), activation_func::relu));
    topology.add(reorder("reorder", input_info("conv"), in_layout));

    auto impl_desc_cpu = ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::cpu};
    auto impl_desc_onednn = ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::onednn};
    auto impl_forcing_map = ov::intel_gpu::ImplForcingMap{{"conv", impl_desc_onednn}, {"activation", impl_desc_cpu}};

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(impl_forcing_map));

    network net(engine, topology, config);
    net.set_input_data("input", input_mem);
    net.execute();

    ASSERT_TRUE(net.has_event("conv"));
    ASSERT_TRUE(net.has_event("reorder"));
    ASSERT_TRUE(net.has_event("activation"));

    auto conv_ev = net.get_primitive_event("conv");
    auto reorder_ev = net.get_primitive_event("reorder");
    auto activation_ev = net.get_primitive_event("activation");

    OV_ASSERT_NO_THROW(downcast<ocl::ocl_base_event>(conv_ev.get()));
    OV_ASSERT_NO_THROW(downcast<ocl::ocl_base_event>(reorder_ev.get()));
    OV_ASSERT_NO_THROW(downcast<ocl::ocl_base_event>(activation_ev.get()));

    // Check if we have real underlying OpenCL events
    ASSERT_TRUE(downcast<ocl::ocl_base_event>(conv_ev.get())->get().get() != nullptr);
    ASSERT_TRUE(downcast<ocl::ocl_base_event>(reorder_ev.get())->get().get() != nullptr);
    ASSERT_TRUE(downcast<ocl::ocl_base_event>(activation_ev.get())->get().get() != nullptr);
}

TEST(network_test, scratchpad_test) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    // benchdnn parameters:
    // --ip --engine=gpu:0 --dir=FWD_B --dt=f16:f16:f16 --stag=abcd --wtag=any --dtag=ab --attr-scratchpad=user mb16384ic768ih1iw1oc3072
    layout in_layout{{16384, 768}, data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(in_layout);
    auto weights = engine.allocate_memory({{3072, 768}, data_types::f16, format::oiyx});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weights", weights));
    topology.add(fully_connected("fc_prim", input_info("input"), "weights", ""));

    auto impl_desc_onednn = ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::onednn};
    auto impl_forcing_map = ov::intel_gpu::ImplForcingMap{{"fc_prim", impl_desc_onednn}};

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(impl_forcing_map));

    network net1(engine, topology, config);
    net1.set_input_data("input", input_mem);
    net1.execute();

    network net2(net1.get_program(), 1);
    net2.set_input_data("input", input_mem);
    net2.execute();

    auto fc1 = net1.get_primitive("fc_prim");
    auto fc2 = net2.get_primitive("fc_prim");

    if (fc1->get_intermediates_memories().size() > 0 && fc2->get_intermediates_memories().size() > 0) {
        ASSERT_TRUE(fc1->get_intermediates_memories()[0]->buffer_ptr() !=
                    fc2->get_intermediates_memories()[0]->buffer_ptr());
    }
}

// this test verifies the case where an example primitive - in this case it's slice_scatter
// calls update_output_memory before buid_deps is called, in this case we expect
// that build_deps will be called from update_output_memory before accessing calling things like input_memory_ptr
TEST(network_test, update_output_memory_calls_build_deps) {
    auto& engine = get_test_engine();

    auto data_mem = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 2, 3}});
    auto updates_mem = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 2, 3}});
    auto start_mem = engine.allocate_memory(layout{ov::PartialShape{4}, data_types::i64, format::bfyx});
    auto stop_mem = engine.allocate_memory(layout{ov::PartialShape{4}, data_types::i64, format::bfyx});
    auto step_mem = engine.allocate_memory(layout{ov::PartialShape{4}, data_types::i64, format::bfyx});

    set_values<float>(data_mem, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
    set_values<float>(updates_mem, {10.f, 20.f, 30.f, 40.f, 50.f, 60.f});
    set_values<int64_t>(start_mem, {0, 0, 0, 0});
    set_values<int64_t>(stop_mem, {1, 1, 2, 3});
    set_values<int64_t>(step_mem, {1, 1, 1, 1});

    topology topo;
    topo.add(input_layout("data", data_mem->get_layout()));
    topo.add(input_layout("updates", updates_mem->get_layout()));
    topo.add(data("start", start_mem));
    topo.add(data("stop", stop_mem));
    topo.add(data("step", step_mem));
    // slice scatter is used here, but there are other primitives that have similar logic: gather, scatter_update, etc...
    topo.add(slice_scatter("slice_scatter", {input_info("data"), input_info("updates"), input_info("start"), input_info("stop"), input_info("step")}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topo, config);
    auto& slice_scatter_node = prog->get_node("slice_scatter");
    // force the flag to be true, so that it can pass the early return at the top of slice_scatter_inst::update_output_memory
    slice_scatter_node.can_be_optimized(true);
    ASSERT_TRUE(slice_scatter_node.can_be_optimized());

    cldnn::network::ptr network;
    ASSERT_NO_THROW(network = network::allocate_network(engine, prog));
    ASSERT_NE(network, nullptr);
}

#endif

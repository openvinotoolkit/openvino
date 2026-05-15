// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/impl_forcing_map_read.hpp"
#include "gemm_inst.h"

using namespace cldnn;
using namespace ::tests;

using ImplForcingMap = ov::intel_gpu::ImplForcingMap;

TEST(force_implementations, parse_ImplForcingMap_simple) {
    const std::string force_impl_string = "{conv1:ocl:kernelname:any}";
    ov::Any any(force_impl_string);

    ImplForcingMap map;
    OV_ASSERT_NO_THROW(map = any.as<ImplForcingMap>());
    ASSERT_EQ(map.size(), 1);

    ASSERT_NE(map.find("conv1"), map.end());
    const auto& conv1_desc = map.at("conv1");
    ASSERT_EQ(conv1_desc.impl_type, impl_types::ocl);
    ASSERT_EQ(conv1_desc.kernel_name, "kernelname");
    ASSERT_EQ(conv1_desc.output_format, format::any);
}

TEST(force_implementations, parse_ImplForcingMap_multiple_entries) {
    const std::string force_impl_string = "{conv1:cpu::any,conv2:ocl:kernel:any,pool1:onednn::bfyx}";
    ov::Any any(force_impl_string);

    ImplForcingMap map;
    OV_ASSERT_NO_THROW(map = any.as<ImplForcingMap>());
    ASSERT_EQ(map.size(), 3);

    ASSERT_NE(map.find("conv1"), map.end());
    const auto& conv1_desc = map.at("conv1");
    ASSERT_EQ(conv1_desc.impl_type, impl_types::cpu);
    ASSERT_EQ(conv1_desc.kernel_name, "");
    ASSERT_EQ(conv1_desc.output_format, format::any);

    ASSERT_NE(map.find("conv2"), map.end());
    const auto& conv2_desc = map.at("conv2");
    ASSERT_EQ(conv2_desc.impl_type, impl_types::ocl);
    ASSERT_EQ(conv2_desc.kernel_name, "kernel");
    ASSERT_EQ(conv2_desc.output_format, format::any);

    ASSERT_NE(map.find("pool1"), map.end());
    const auto& pool1_desc = map.at("pool1");
    ASSERT_EQ(pool1_desc.impl_type, impl_types::onednn);
    ASSERT_EQ(pool1_desc.kernel_name, "");
    ASSERT_EQ(pool1_desc.output_format, format::bfyx);
}

TEST(force_implementations, parse_ImplForcingMap_colon_in_name) {
    const std::string force_impl_string = "{\"layer:name\":ocl::any}";
    ov::Any any(force_impl_string);

    ImplForcingMap map;
    OV_ASSERT_NO_THROW(map = any.as<ImplForcingMap>());
    ASSERT_EQ(map.size(), 1);

    ASSERT_NE(map.find("layer:name"), map.end());
    const auto& layer_desc = map.at("layer:name");
    ASSERT_EQ(layer_desc.impl_type, impl_types::ocl);
    ASSERT_EQ(layer_desc.kernel_name, "");
    ASSERT_EQ(layer_desc.output_format, format::any);
}


TEST(force_implementations, force_ocl_for_gemm) {
    auto& engine = get_test_engine();

    auto input1_layout = layout{ov::PartialShape{1, 1, 3, 4}, data_types::f32, format::bfyx};
    auto input2_layout = layout{ov::PartialShape{1, 1, 4, 2}, data_types::f32, format::bfyx};

    auto input1 = engine.allocate_memory(input1_layout);
    auto input2 = engine.allocate_memory(input2_layout);

    set_values(input1, std::vector<float>{
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 1.f, 2.f, 3.f
    });
    set_values(input2, std::vector<float>{
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f,
        7.f, 8.f
    });

    topology topology(
        input_layout("input1", input1_layout),
        input_layout("input2", input2_layout),
        gemm("gemm", { input_info("input1"), input_info("input2") }, data_types::f32, false, false, 1.f, 0.f, 4, 4)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::force_implementations(
        ov::intel_gpu::ImplForcingMap{{"gemm", {format::any, "", impl_types::ocl}}}
    ));

    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto gemm_prim = network.get_primitive("gemm");
    ASSERT_TRUE(gemm_prim != nullptr);
    auto impl = gemm_prim->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->m_manager != nullptr);
    EXPECT_EQ(impl->m_manager->get_impl_type(), impl_types::ocl);

    ASSERT_NO_THROW(network.execute());
}

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(force_implementations, force_onednn_for_gemm) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    auto input1_layout = layout{ov::PartialShape{1, 1, 3, 4}, data_types::f16, format::bfyx};
    auto input2_layout = layout{ov::PartialShape{1, 1, 4, 2}, data_types::f16, format::bfyx};

    auto input1 = engine.allocate_memory(input1_layout);
    auto input2 = engine.allocate_memory(input2_layout);

    set_values(input1, std::vector<ov::float16>{
        ov::float16(1.f), ov::float16(2.f), ov::float16(3.f), ov::float16(4.f),
        ov::float16(5.f), ov::float16(6.f), ov::float16(7.f), ov::float16(8.f),
        ov::float16(9.f), ov::float16(1.f), ov::float16(2.f), ov::float16(3.f)
    });
    set_values(input2, std::vector<ov::float16>{
        ov::float16(1.f), ov::float16(2.f),
        ov::float16(3.f), ov::float16(4.f),
        ov::float16(5.f), ov::float16(6.f),
        ov::float16(7.f), ov::float16(8.f)
    });

    topology topology(
        input_layout("input1", input1_layout),
        input_layout("input2", input2_layout),
        gemm("gemm", { input_info("input1"), input_info("input2") }, data_types::f16, false, false, 1.f, 0.f, 4, 4)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::force_implementations(
        ov::intel_gpu::ImplForcingMap{{"gemm", {format::any, "", impl_types::onednn}}}
    ));

    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto gemm_prim = network.get_primitive("gemm");
    ASSERT_TRUE(gemm_prim != nullptr);
    auto impl = gemm_prim->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->m_manager != nullptr);
    EXPECT_EQ(impl->m_manager->get_impl_type(), impl_types::onednn);

    ASSERT_NO_THROW(network.execute());
}
#endif

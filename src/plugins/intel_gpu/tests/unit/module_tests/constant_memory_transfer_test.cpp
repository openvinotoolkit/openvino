// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"

namespace {

constexpr const char* data_id = "const_data";

std::string get_unsupported_reason(const cldnn::engine& engine) {
    const auto& device_info = engine.get_device_info();
    if (device_info.dev_type != cldnn::device_type::integrated_gpu) {
        return "Requires integrated GPU";
    }
    if (device_info.arch <= cldnn::gpu_arch::xe2) {
        return "Requires arch > Xe2 GPU";
    }
    if (!device_info.has_separate_cache) {
        return "Requires separate host/device cache";
    }
    if (!engine.supports_allocation(cldnn::allocation_type::usm_host)) {
        return "Requires usm_host allocation support";
    }
    if (!engine.supports_allocation(cldnn::allocation_type::usm_device)) {
        return "Requires usm_device allocation support";
    }
    return {};
}

cldnn::topology make_topology_with_host_constant(cldnn::engine& engine) {
    const cldnn::layout data_layout{{1, 8}, cldnn::data_types::f32, cldnn::format::bfyx};
    auto data_mem = engine.allocate_memory(data_layout, cldnn::allocation_type::usm_host, false);

    cldnn::topology topology;
    topology.add(cldnn::input_layout("input", data_layout));
    topology.add(cldnn::data(data_id, data_mem));
    topology.add(cldnn::concatenation("concat", {cldnn::input_info("input"), cldnn::input_info(data_id)}, 1));
    return topology;
}

cldnn::program::ptr build_loaded_program(cldnn::engine& engine,
                                         cldnn::topology& topology,
                                         const ov::intel_gpu::ExecutionConfig& config) {
    cldnn::membuf mem_buf;
    {
        std::ostream out_mem(&mem_buf);
        cldnn::BinaryOutputBuffer output_buffer(out_mem);
        output_buffer.set_stream(tests::get_test_stream_ptr().get());
        cldnn::program::build_program(engine, topology, config)->save(output_buffer);
    }

    std::istream in_mem(&mem_buf);
    cldnn::BinaryInputBuffer input_buffer(in_mem, engine);
    auto loaded_program = std::make_shared<cldnn::program>(engine, config);
    loaded_program->load(input_buffer);
    return loaded_program;
}

}  // namespace

TEST(TransferMemoryToDeviceTest, ProgramTransfersLoadedHostConstantToDeviceOnArchAfterXe2) {
    auto& engine = tests::get_test_engine();
    if (const auto reason = get_unsupported_reason(engine); !reason.empty()) {
        GTEST_SKIP() << reason;
    }

    auto config = tests::get_test_default_config(engine);
    auto topology = make_topology_with_host_constant(engine);
    auto loaded_program = build_loaded_program(engine, topology, config);

    auto& data_node_before = loaded_program->get_node(data_id).as<cldnn::data>();
    ASSERT_EQ(data_node_before.get_attached_memory().get_allocation_type(), cldnn::allocation_type::usm_host);

    cldnn::program_wrapper::transfer_memory_to_device(*loaded_program);

    auto& data_node_after = loaded_program->get_node(data_id).as<cldnn::data>();
    EXPECT_EQ(data_node_after.get_attached_memory().get_allocation_type(), cldnn::allocation_type::usm_device);
}

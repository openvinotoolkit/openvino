// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>

#include "test_utils.h"
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/eltwise.hpp>

using namespace cldnn;
using namespace tests;

class test_device_mem_usage_estimation: public ::testing::Test {
public:
    void test_basic(bool is_caching_test) {
        ExecutionConfig cfg = get_test_default_config(get_test_engine());
        cfg.set_property(ov::intel_gpu::queue_type(QueueTypes::out_of_order));

        std::shared_ptr<cldnn::engine> engine1 = create_test_engine();
        if (engine1->get_device_info().supports_immad) {
            // Enable this test for out_of_order queue-type if Onednn supports out_of_order
            return;
        }

        auto input1 = engine1->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });
        auto input2 = engine1->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });
        topology topology(
            input_layout("input1", input1->get_layout()),
            input_layout("input2", input2->get_layout()),
            permute("permute1", input_info("input1"), { 0, 3, 1, 2 }),
            permute("permute2", input_info("input2"), { 0, 2, 1, 3 }),
            eltwise("eltw", { input_info("permute1"), input_info("permute2") }, eltwise_mode::sum, data_types::f16),
            reorder("output", input_info("eltw"), format::bfyx, data_types::f32)
        );

        auto prog = program::build_program(*engine1, topology, cfg);
        std::pair<int64_t, int64_t> estimated_mem_usage = prog->get_estimated_device_mem_usage();

        std::shared_ptr<cldnn::engine> engine2 = create_test_engine();
        auto input3 = engine2->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });
        auto input4 = engine2->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });

        cldnn::network::ptr network = get_network(*engine2, topology, cfg, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input1", input3);
        network->set_input_data("input2", input4);
        ASSERT_EQ(estimated_mem_usage.first + estimated_mem_usage.second, engine2->get_used_device_memory(allocation_type::usm_device));
    }
};

TEST_F(test_device_mem_usage_estimation, basic) {
    this->test_basic(false);
}

TEST_F(test_device_mem_usage_estimation, basic_cached) {
    this->test_basic(true);
}

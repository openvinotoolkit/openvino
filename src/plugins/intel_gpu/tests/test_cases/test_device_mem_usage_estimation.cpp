// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cstddef>

#include "test_utils.h"
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/eltwise.hpp>

using namespace cldnn;
using namespace tests;

TEST(test_device_mem_usage_estimation, basic) {
    std::shared_ptr<cldnn::engine> engine1 = create_test_engine(cldnn::queue_types::out_of_order);

    auto input1 = engine1->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });
    auto input2 = engine1->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });
    topology topology(
        input_layout("input1", input1->get_layout()),
        input_layout("input2", input2->get_layout()),
        permute("permute1", "input1", { 0, 3, 1, 2 }),
        permute("permute2", "input2", { 0, 2, 1, 3 }),
        eltwise("eltw", {"permute1", "permute2"}, eltwise_mode::sum, data_types::f16),
        reorder("output", "eltw", format::bfyx, data_types::f32)
    );

    auto prog = program::build_program(*engine1, topology, build_options());
    std::pair<int64_t, int64_t> estimated_mem_usage = prog->get_estimated_device_mem_usage();

    std::shared_ptr<cldnn::engine> engine2 = create_test_engine(cldnn::queue_types::out_of_order);
    auto input3 = engine2->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });
    auto input4 = engine2->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });

    network network(*engine2, topology);
    network.set_input_data("input1", input3);
    network.set_input_data("input2", input4);
    ASSERT_EQ(estimated_mem_usage.first + estimated_mem_usage.second, engine2->get_used_device_memory(allocation_type::usm_device));
}

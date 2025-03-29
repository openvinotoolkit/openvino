// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/reorder.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(DISABLED_oooq_test, simple) {
    auto eng = engine::create(engine_types::ocl, runtime_types::ocl);

    auto in_layout = layout{ data_types::f32, format::bfyx, { 1, 1, 1, 1 } };
    auto concat_layout = layout{ data_types::f32, format::bfyx, { 1, 1, 1, 2 } };
    auto input_mem = eng->allocate_memory(in_layout);
    set_values(input_mem, { 50 });

    /*                 ---- r1 ---- r3 ----            -- r7 --
                     /                      \        /          \
        in --- r0 --                          - c6 -             --- c9
                     \                      /        \          /
                       -- r2 -- r4 -- r5 --            -- r8 --
    */

    topology tpl;
    tpl.add(input_layout("in", input_mem->get_layout()));
    tpl.add(reorder("r0", input_info("in"), input_mem->get_layout(), std::vector<float>{ 0 }));
    tpl.add(reorder("r1", input_info("r0"), input_mem->get_layout(), std::vector<float>{ 1 }));
    tpl.add(reorder("r2", input_info("r0"), input_mem->get_layout(), std::vector<float>{ 2 }));
    tpl.add(reorder("r3", input_info("r1"), input_mem->get_layout(), std::vector<float>{ 3 }));
    tpl.add(reorder("r4", input_info("r2"), input_mem->get_layout(), std::vector<float>{ 4 }));
    tpl.add(reorder("r5", input_info("r4"), input_mem->get_layout(), std::vector<float>{ 5 }));

    tpl.add(concatenation("c6", { input_info("r3"), input_info("r5") }, 3));

    tpl.add(reorder("r7", input_info("c6"), concat_layout, std::vector<float>{ 7 }));
    tpl.add(reorder("r8", input_info("c6"), concat_layout, std::vector<float>{ 8 }));
    tpl.add(concatenation("c9", { input_info("r7"), input_info("r8") }, 2));

    ExecutionConfig cfg = get_test_default_config(*eng);
    cfg.set_property(ov::intel_gpu::queue_type(QueueTypes::out_of_order));
    if (eng->get_device_info().supports_immad) {
        // Onednn currently does NOT support out_of_order queue-type
        return;
    }

    network net{ *eng, tpl, cfg };

    net.set_input_data("in", input_mem);
    auto output = net.execute().at("c9").get_memory();

    ASSERT_TRUE(output->get_layout().spatial(0) == 2);
    ASSERT_TRUE(output->get_layout().spatial(1) == 2);
    ASSERT_TRUE(output->get_layout().feature() == 1);
    ASSERT_TRUE(output->get_layout().batch() == 1);
}

// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/reorder.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(DISABLED_oooq_test, simple)
{
    engine_configuration cfg{ false, queue_types::out_of_order };
    auto eng = engine::create(engine_types::ocl, runtime_types::ocl, cfg);

    auto input_mem = eng->allocate_memory(layout{ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    set_values(input_mem, { 50 });

    /*                 ---- r1 ---- r3 ----            -- r7 --
                     /                      \        /          \
        in --- r0 --                          - c6 -             --- c9
                     \                      /        \          /
                       -- r2 -- r4 -- r5 --            -- r8 --
    */

    topology tpl;
    tpl.add(input_layout("in", input_mem->get_layout()));
    tpl.add(reorder("r0", "in", input_mem->get_layout(), std::vector<float>{ 0 }));
    tpl.add(reorder("r1", "r0", input_mem->get_layout(), std::vector<float>{ 1 }));
    tpl.add(reorder("r2", "r0", input_mem->get_layout(), std::vector<float>{ 2 }));
    tpl.add(reorder("r3", "r1", input_mem->get_layout(), std::vector<float>{ 3 }));
    tpl.add(reorder("r4", "r2", input_mem->get_layout(), std::vector<float>{ 4 }));
    tpl.add(reorder("r5", "r4", input_mem->get_layout(), std::vector<float>{ 5 }));

    tpl.add(concatenation("c6", { "r3", "r5" }, 3));
    layout concat_lay = input_mem->get_layout();
    concat_lay.size.spatial[0] *= 2;

    tpl.add(reorder("r7", "c6", concat_lay, std::vector<float>{ 7 }));
    tpl.add(reorder("r8", "c6", concat_lay, std::vector<float>{ 8 }));
    tpl.add(concatenation("c9", { "r7", "r8" }, 2));
    concat_lay.size.spatial[1] *= 2;

    build_options options;
    network net{ *eng, tpl, options };

    net.set_input_data("in", input_mem);
    auto output = net.execute().at("c9").get_memory();

    EXPECT_TRUE(output->get_layout().spatial(0) == 2);
    EXPECT_TRUE(output->get_layout().spatial(1) == 2);
    EXPECT_TRUE(output->get_layout().feature() == 1);
    EXPECT_TRUE(output->get_layout().batch() == 1);
}

/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>
#include <api/engine.hpp>
#include <api/memory.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/input_layout.hpp>
#include <api/concatenation.hpp>
#include <api/reorder.hpp>

#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(DISABLED_oooq_test, simple)
{
    engine_configuration cfg{ false, false, false, std::string(), std::string(), true };
    engine eng{ cfg };

    memory input_mem = memory::allocate(eng, layout{ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    set_values(input_mem, { 50 });

    /*                 ---- r1 ---- r3 ----            -- r7 --
                     /                      \        /          \
        in --- r0 --                          - c6 -             --- c9
                     \                      /        \          /
                       -- r2 -- r4 -- r5 --            -- r8 --
    */

    topology tpl;
    tpl.add(input_layout("in", input_mem.get_layout()));
    tpl.add(reorder("r0", "in", input_mem.get_layout(), std::vector<float>{ 0 }));
    tpl.add(reorder("r1", "r0", input_mem.get_layout(), std::vector<float>{ 1 }));
    tpl.add(reorder("r2", "r0", input_mem.get_layout(), std::vector<float>{ 2 }));
    tpl.add(reorder("r3", "r1", input_mem.get_layout(), std::vector<float>{ 3 }));
    tpl.add(reorder("r4", "r2", input_mem.get_layout(), std::vector<float>{ 4 }));
    tpl.add(reorder("r5", "r4", input_mem.get_layout(), std::vector<float>{ 5 }));

    tpl.add(concatenation("c6", { "r3", "r5" }, concatenation::along_x));
    layout concat_lay = input_mem.get_layout();
    concat_lay.size.spatial[0] *= 2;

    tpl.add(reorder("r7", "c6", concat_lay, std::vector<float>{ 7 }));
    tpl.add(reorder("r8", "c6", concat_lay, std::vector<float>{ 8 }));
    tpl.add(concatenation("c9", { "r7", "r8" }, concatenation::along_y));
    concat_lay.size.spatial[1] *= 2;

    build_options options;
    network net{ eng, tpl, options };

    net.set_input_data("in", input_mem);
    auto output = net.execute().at("c9").get_memory();

    EXPECT_TRUE(output.get_layout().size.spatial[0] == 2);
    EXPECT_TRUE(output.get_layout().size.spatial[1] == 2);
    EXPECT_TRUE(output.get_layout().size.feature[0] == 1);
    EXPECT_TRUE(output.get_layout().size.batch[0] == 1);
}
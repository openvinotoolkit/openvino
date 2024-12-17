// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"
#include "convolution_inst.h"
#include "permute_inst.h"
#include "reshape_inst.h"

using namespace cldnn;
using namespace ::tests;

TEST(merge_reorder_reshape_permute, optimize_yolo) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ov::PartialShape({1, 2, 4, 6}), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape({1, 2, 4, 6}), data_types::f16, format::bfyx});
    auto weight = engine.allocate_memory(layout{ov::PartialShape({3, 2, 1, 1}), data_types::f16, format::bfyx});
    set_values<ov::float16>(input, { 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f,
                        1.f, 2.f, 3.f, 1.f, 2.f, 4.f,
                        5.f, 1.f, 1.f, 2.f, 1.f, 2.f,
                        2.0f, 3.0f, 1.0f, 4.0f, 1.0f, 4.0f,
                        3.0f, 2.0f, 0.0f, 1.0f, 0.0f, 2.0f,
                        2.f, 4.f, 1.f, 1.f, 2.f, 1.f,
                        1.f, 2.f, 0.f, 2.f, 5.f, 2.f,
                        4.0f, 3.0f, 1.0f, 0.0f, 3.0f, 2.0f});

    set_values<ov::float16>(weight, { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));
    topology.add(convolution("convolution", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(reorder("reorder_inter", input_info("convolution"), format::bfyx, data_types::f16));
    topology.add(reshape("reshape", input_info("reorder_inter"), false, {1, 3, 24, 1}, ov::PartialShape{1, 3, 24, 1}));
    topology.add(permute("permute_inter", input_info("reshape"), {0, 2, 1, 3}));
    topology.add(softmax("softmax", input_info("permute_inter"), 1));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config);

    network net(prog);

    net.set_input_data("input", input);
    auto output = net.execute();

    auto out_mem = output.at("softmax").get_memory();
    mem_lock<ov::float16> lock(out_mem, get_test_stream());

    for (size_t i = 0; i < out_mem->count(); i++) {
        float actual = lock[i];
        std::cout << actual << ", ";
    }
    std::cout << std::endl;
}

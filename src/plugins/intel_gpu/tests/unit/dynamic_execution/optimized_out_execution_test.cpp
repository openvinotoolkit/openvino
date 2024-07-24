// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace optimized_out_execution_test {
TEST(optimized_out_execution_test, concat_blocked_format) {
    auto& engine = get_test_engine();

    auto input1_layout_dyn = layout{ov::PartialShape::dynamic(4), data_types::f16, format::b_fs_yx_fsv16};
    auto input2_layout_dyn = layout{ov::PartialShape::dynamic(4), data_types::f16, format::b_fs_yx_fsv16};
    auto input3_layout_dyn = layout{ov::PartialShape::dynamic(4), data_types::f16, format::b_fs_yx_fsv16};
    auto input4_layout_dyn = layout{ov::PartialShape::dynamic(4), data_types::f16, format::b_fs_yx_fsv16};

    auto input1_layout = layout{ov::PartialShape{1, 16, 2, 1}, data_types::f16, format::b_fs_yx_fsv16};
    auto input2_layout = layout{ov::PartialShape{1, 16, 2, 1}, data_types::f16, format::b_fs_yx_fsv16};
    auto input3_layout = layout{ov::PartialShape{1, 16, 2, 1}, data_types::f16, format::b_fs_yx_fsv16};
    auto input4_layout = layout{ov::PartialShape{1, 16, 2, 1}, data_types::f16, format::b_fs_yx_fsv16};

    auto input1 = engine.allocate_memory(input1_layout);
    auto input2 = engine.allocate_memory(input2_layout);
    auto input3 = engine.allocate_memory(input3_layout);
    auto input4 = engine.allocate_memory(input4_layout);

    set_values<ov::float16>(input1, {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f
    });
    set_values<ov::float16>(input2, {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f
    });
    set_values<ov::float16>(input3, {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f
    });
    set_values<ov::float16>(input4, {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f
    });

    topology topology(input_layout("input1", input1_layout_dyn),
                      input_layout("input2", input2_layout_dyn),
                      input_layout("input3", input3_layout_dyn),
                      input_layout("input4", input4_layout_dyn),
                      eltwise("eltwise1", input_info("input1"), input_info("input2"), eltwise_mode::sum),
                      eltwise("eltwise2", input_info("input3"), input_info("input4"), eltwise_mode::sum),
                      concatenation("concat", { input_info("eltwise1"), input_info("eltwise2") }, 1),
                      permute("permute", input_info("concat"), {0, 2, 3, 1}),
                      reorder("output", input_info("permute"), format::bfyx, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    network.set_input_data("input4", input4);

    std::vector<ov::float16> ref = {
            2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f,
            20.0f, 40.0f, 60.0f, 80.0f, 100.0f, 120.0f, 140.0f, 160.0f,
            2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f,
            20.0f, 40.0f, 60.0f, 80.0f, 100.0f, 120.0f, 140.0f, 160.0f,
            2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f,
            20.0f, 40.0f, 60.0f, 80.0f, 100.0f, 120.0f, 140.0f, 160.0f,
            2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f,
            20.0f, 40.0f, 60.0f, 80.0f, 100.0f, 120.0f, 140.0f, 160.0f,
            2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f,
            20.0f, 40.0f, 60.0f, 80.0f, 100.0f, 120.0f, 140.0f, 160.0f,
            2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f,
            20.0f, 40.0f, 60.0f, 80.0f, 100.0f, 120.0f, 140.0f, 160.0f,
            2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f,
            20.0f, 40.0f, 60.0f, 80.0f, 100.0f, 120.0f, 140.0f, 160.0f,
            2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f,
            20.0f, 40.0f, 60.0f, 80.0f, 100.0f, 120.0f, 140.0f, 160.0f
    };

    auto outputs = network.execute();
    auto output_mem = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> output_mem_ptr(output_mem, get_test_stream());

    for (size_t i = 0; i < output_mem->get_layout().get_linear_size(); ++i) {
        ASSERT_EQ(output_mem_ptr[i], ref[i]);
    }
}
}  // is_valid_fusion_tests

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "fully_connected_inst.h"
#include "gather_inst.h"
#include "pass_manager.h"
#include "permute_inst.h"
#include "reshape_inst.h"
#include "shape_of_inst.h"
#include "convolution_inst.h"
#include "dft_inst.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(add_required_all_reduce, base_func) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{1, 32, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f16, format::bfyx};
    auto weight_mem = engine.allocate_memory({{2, 32}, data_types::f32, format::bfyx});
    std::vector<float> weight_data(weight_mem->get_layout().count());
    std::iota(weight_data.begin(), weight_data.end(), 1.0f);
    set_values(weight_mem, weight_data);

    auto input_l = layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx};
    topology topology(input_layout("input", input_l),
                data("weight", weight_mem),
                fully_connected("fc", input_info("input"), {"weight"}, "", data_types::f32),
                reorder("reorder", input_info("fc"), format::bfyx, data_types::f32)); /*output padding*/

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    auto input_mem = engine.allocate_memory({{10, 32}, data_types::f32, format::bfyx});
    std::vector<float> input_data(input_mem->get_layout().count());
    std::iota(input_data.begin(), input_data.end(), 0.5f);
    set_values(input_mem, input_data);

    network.set_input_data("input", input_mem);
}

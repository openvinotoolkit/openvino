// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/shape_of.hpp>
#include <intel_gpu/primitives/broadcast.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/non_zero.hpp>

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace update_shape_tests {
TEST(update_shape_test, ocl_impl_in_shapeof_subgraph) {
    auto& engine = get_test_engine();

    layout const1_gather_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto const1_gather = engine.allocate_memory(const1_gather_layout);
    set_values<int32_t>(const1_gather, {1});

    layout const_broadcast_layout = layout{ov::PartialShape{}, data_types::i32, format::bfyx};
    auto const_broadcast = engine.allocate_memory(const_broadcast_layout);
    set_values<int32_t>(const_broadcast, {1});

    layout input_l= layout{ov::PartialShape{1, 128}, data_types::i32, format::bfyx};
    auto input_mem = engine.allocate_memory(input_l);
    set_values<int32_t>(input_mem, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8,});

    auto input_l_dynamic = layout{ov::PartialShape::dynamic(2), data_types::i32, format::bfyx};
    topology topology(input_layout("input", input_l_dynamic),
                      data("const1_gather", const1_gather),
                      data("const_broadcast", const_broadcast),
                      shape_of("shape_of", input_info("input"), data_types::i32),
                      gather("gather", input_info("shape_of"), input_info("const1_gather"), 0, 1, ov::Shape({1})),
                      broadcast("broadcast1", input_info("const_broadcast"), input_info("gather"), {}, ov::op::BroadcastType::NUMPY),
                      count_nonzero("count_nonzero", input_info("broadcast1")),
                      gather_nonzero("gather_nonzero", input_info("broadcast1"), input_info("count_nonzero")),
                      broadcast("broadcast2", input_info("gather_nonzero"), input_info("shape_of"), {}, ov::op::BroadcastType::BIDIRECTIONAL));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    std::map<primitive_id, network_output> outputs;
    OV_ASSERT_NO_THROW(outputs = network.execute());
}
}  // update_shape_test

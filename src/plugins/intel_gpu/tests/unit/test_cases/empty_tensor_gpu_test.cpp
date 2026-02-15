// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/non_zero.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include "openvino/reference/non_zero.hpp"

#include <cstddef>

using namespace cldnn;
using namespace ::tests;
namespace {
struct empty_tensor_test_params {
    layout nonzero_input_layout;
    layout concat_input_layout;
    int64_t concat_axis;
};

class test_empty_tensor : public testing::TestWithParam<empty_tensor_test_params> {};

TEST_P(test_empty_tensor, concat_two_inputs) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto p = GetParam();
    auto& engine = get_test_engine();

    auto nonzero_input_mem = engine.allocate_memory(p.nonzero_input_layout);
    auto concat_data_mem = engine.allocate_memory(p.concat_input_layout);

    std::vector<int32_t> concat_another_input_data = rg.generate_random_1d<int32_t>(p.concat_input_layout.count(), 0, 100);

    set_values(concat_data_mem, concat_another_input_data);

    topology topology;
    topology.add(input_layout("nonzero_input", p.nonzero_input_layout));
    topology.add(data("concat_data", concat_data_mem));
    topology.add(count_nonzero("count_nonzero", input_info("nonzero_input")));
    topology.add(gather_nonzero("gather_nonzero", input_info("nonzero_input"), input_info("count_nonzero")));
    topology.add(concatenation("concat", { input_info("gather_nonzero"), input_info("concat_data") }, p.concat_axis));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    std::vector<int32_t> nonzero_input_with_all_zero(p.nonzero_input_layout.count());
    std::fill(nonzero_input_with_all_zero.begin(), nonzero_input_with_all_zero.end(), 0);
    set_values(nonzero_input_mem, nonzero_input_with_all_zero); // nonzero output shape will be (2, 0)

    network.set_input_data("nonzero_input", nonzero_input_mem);
    auto outputs = network.execute();
    auto output = outputs.at("concat").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < output->get_layout().count(); ++i) {
        ASSERT_EQ(concat_another_input_data[i], output_ptr[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_empty, test_empty_tensor,
    testing::ValuesIn(std::vector<empty_tensor_test_params>{
        {
            layout{ov::PartialShape{1, 2}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{2, 3}, data_types::i32, format::bfyx},
            1
        },
        {
            layout{ov::PartialShape{2, 3, 4}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{3, 4}, data_types::i32, format::bfyx},
            1
        },
        {
            layout{ov::PartialShape{3, 1, 2, 5, 1}, data_types::i32, format::bfzyx},
            layout{ov::PartialShape{5, 3}, data_types::i32, format::bfyx},
            1
        }
    }));
} // namespace

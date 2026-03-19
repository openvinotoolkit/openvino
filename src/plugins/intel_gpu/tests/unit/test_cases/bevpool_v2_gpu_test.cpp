// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/bevpool_v2.hpp>
#include <intel_gpu/primitives/input_layout.hpp>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

TEST(bevpool_v2_gpu_test, ref_comp_f32) {
    auto& engine = get_test_engine();
    auto stream = get_test_stream_ptr(get_test_default_config(engine));

    const auto cf = engine.allocate_memory({ov::PartialShape{1, 1, 2, 2}, data_types::f32, format::bfyx});
    const auto dw = engine.allocate_memory({ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx});
    const auto idx = engine.allocate_memory({ov::PartialShape{5}, data_types::i32, format::bfyx});
    const auto itv = engine.allocate_memory({ov::PartialShape{6}, data_types::i32, format::bfyx});

    set_values<float>(cf, {10.f, 20.f, 30.f, 40.f});
    set_values<float>(dw, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    set_values<int32_t>(idx, {0, 1, 2, 3, 4});
    set_values<int32_t>(itv, {0, 2, 0, 2, 5, 1});

    const bound3d x_bound{-10.f, 10.f, 0.5f};
    const bound3d y_bound{-10.f, 10.f, 0.5f};
    const bound3d z_bound{-5.f, 3.f, 0.5f};
    const bound3d d_bound{0.f, 2.f, 1.f};

    topology topology;
    topology.add(input_layout("cf", cf->get_layout()));
    topology.add(input_layout("dw", dw->get_layout()));
    topology.add(input_layout("idx", idx->get_layout()));
    topology.add(input_layout("itv", itv->get_layout()));
    topology.add(bevpool_v2("bevpool_v2",
                            {input_info("cf"), input_info("dw"), input_info("idx"), input_info("itv")},
                            1,
                            1,
                            2,
                            2,
                            1,
                            2,
                            x_bound,
                            y_bound,
                            z_bound,
                            d_bound));

    auto network = get_network(engine, topology, get_test_default_config(engine), stream, false);
    network->set_input_data("cf", cf);
    network->set_input_data("dw", dw);
    network->set_input_data("idx", idx);
    network->set_input_data("itv", itv);

    const auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t{1});

    const auto output = outputs.at("bevpool_v2").get_memory();
    const auto expected = engine.allocate_memory({ov::PartialShape{1, 1, 2, 1}, data_types::f32, format::bfyx});
    set_values<float>(expected, {50.f, 300.f});

    mem_lock<float> output_ptr(output, get_test_stream());
    mem_lock<float> expected_ptr(expected, get_test_stream());
    ASSERT_EQ(output_ptr.size(), expected_ptr.size());
    for (size_t i = 0; i < output_ptr.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_ptr[i], output_ptr[i], 2e-3f)) << "at index " << i;
    }
}

}  // namespace

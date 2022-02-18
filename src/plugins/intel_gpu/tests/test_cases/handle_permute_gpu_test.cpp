// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convert_color.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/resample.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(handle_permute, convert_permute_to_reorder) {
    auto& engine = get_test_engine();

    int32_t width = 224;
    int32_t height = 448;
    int32_t input_height = height + height / 2;

    auto input = engine.allocate_memory({ data_types::f32, format::byxf, { 1, 1, width, input_height } });

    std::vector<float> input_data = generate_random_1d<float>(width * input_height, 0, 255);
    set_values(input, input_data);

    layout output_layout(data_types::f32, cldnn::format::byxf, { 1, 3, width, height });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(convert_color("convert_color", { "input" }, cldnn::convert_color::color_format::NV12, cldnn::convert_color::color_format::RGB,
                               cldnn::convert_color::memory_type::buffer, output_layout));
    topology.add(permute("permute", "convert_color", { 0, 2, 3, 1 }));
    topology.add(resample("resample", "permute", { 1, 3, width, height }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    std::vector<int32_t> expected_shape = { 1, 3, width, height };
    std::vector<int32_t> output_shape = outputs.at("resample").get_memory()->get_layout().size.sizes();

    for (size_t i = 0; i < expected_shape.size(); ++i) {
        EXPECT_EQ(output_shape[i], expected_shape[i]);
    }
}

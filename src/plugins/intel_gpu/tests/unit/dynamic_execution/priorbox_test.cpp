// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "openvino/reference/prior_box.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/prior_box.hpp>
#include <intel_gpu/primitives/shape_of.hpp>
#include <intel_gpu/primitives/strided_slice.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace priorbox_constant_propagation_test {
TEST(DISABLED_priorbox_constant_propagation_test, basic) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    layout input_layout_1 = layout{ov::PartialShape{1, 24, 30, 30}, data_types::f32, format::bfyx};
    layout input_layout_2= layout{ov::PartialShape{1, 3, 224, 224}, data_types::u8, format::bfyx};
    layout bias_layout = layout{ov::PartialShape{1, 24, 1, 1}, data_types::f32, format::bfyx};
    layout scale_layout = layout{ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx};

    auto input1_mem = engine.allocate_memory(input_layout_1);
    auto input2_mem = engine.allocate_memory(input_layout_2);
    auto bias_mem = engine.allocate_memory(bias_layout);
    auto scale_mem = engine.allocate_memory(scale_layout);

    auto input1_data = rg.generate_random_1d<float>(input_layout_1.count(), -10, 10, 8);
    auto input2_data = rg.generate_random_1d<uint8_t>(input_layout_2.count(), 0, 255);
    auto bias_data = rg.generate_random_1d<float>(bias_layout.count(), -10, 10, 8);
    auto scale_data = rg.generate_random_1d<float>(scale_layout.count(), -10, 10, 8);

    set_values(input1_mem, input1_data);
    set_values(input2_mem, input2_data);
    set_values(bias_mem, bias_data);
    set_values(scale_mem, scale_data);

    ov::op::v8::PriorBox::Attributes attrs;
    attrs.min_size = {64};
    attrs.max_size = {300};
    attrs.aspect_ratio = {2};
    attrs.variance = {0.1, 0.1, 0.2, 0.2};
    attrs.step = 1.0f;
    attrs.offset = 0.5f;
    attrs.clip = false;
    attrs.flip = true;
    attrs.scale_all_sizes = true;
    attrs.fixed_ratio = {};
    attrs.fixed_size = {};
    attrs.density = {};
    attrs.min_max_aspect_ratios_order = true;

    cldnn::tensor out_size{};
    cldnn::tensor img_size{};

    auto in_layout_1 = layout{ov::PartialShape{1, 24, -1, -1}, data_types::f32, format::bfyx};
    auto in_layout_2 = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    std::vector<input_info> inputs1{ input_info("strideSlice1"), input_info("strideSlice2")};
    std::vector<input_info> inputs2{ input_info("strideSlice3"), input_info("strideSlice4")};

    auto const_shape = engine.allocate_memory({ov::PartialShape{3}, data_types::i32, format::bfyx});
    set_values<int32_t>(const_shape, {1, 2, -1});

    topology topology(input_layout("input1", in_layout_1),
                      input_layout("input2", in_layout_2),
                      data("bias1", bias_mem),
                      data("bias2", bias_mem),
                      data("scale1", bias_mem),
                      data("pattern_id1", const_shape),
                      eltwise("eltwise_b1_mul", input_info("bias1"), input_info("scale1"), eltwise_mode::prod),
                      eltwise("eltwise_b1_add", input_info("eltwise_b1_mul"), input_info("bias2"), eltwise_mode::sum),
                      eltwise("eltwise1", input_info("input1"), input_info("eltwise_b1_add"), eltwise_mode::sum),
                      shape_of("shapeOf1", input_info("eltwise1"), data_types::i32),
                      shape_of("shapeOf2", input_info("input2"), data_types::i32),
                      strided_slice("strideSlice1", input_info("shapeOf1"), {2}, {4}, {1}, {0}, {1}, {0}, {0}, {0}, {2}),
                      strided_slice("strideSlice2", input_info("shapeOf2"), {2}, {4}, {1}, {0}, {1}, {0}, {0}, {0}, {2}),
                      prior_box("priorBox1",
                                inputs1,
                                out_size,
                                img_size,
                                attrs.min_size,
                                attrs.max_size,
                                attrs.aspect_ratio,
                                attrs.flip,
                                attrs.clip,
                                attrs.variance,
                                attrs.step,
                                attrs.offset,
                                attrs.scale_all_sizes,
                                attrs.fixed_ratio,
                                attrs.fixed_size,
                                attrs.density,
                                false),
                      reshape("reshape1", input_info("priorBox1"), input_info("pattern_id1"), false, ov::PartialShape::dynamic(3)),
                      input_layout("input3", in_layout_1),
                      input_layout("input4", in_layout_2),
                      data("bias3", bias_mem),
                      data("bias4", bias_mem),
                      data("scale2", bias_mem),
                      data("pattern_id2", const_shape),
                      eltwise("eltwise_b2_mul", input_info("bias3"), input_info("scale2"), eltwise_mode::prod),
                      eltwise("eltwise_b2_add", input_info("eltwise_b2_mul"), input_info("bias4"), eltwise_mode::sum),
                      eltwise("eltwise2", input_info("input3"), input_info("eltwise_b2_add"), eltwise_mode::sum),
                      shape_of("shapeOf3", input_info("eltwise2"), data_types::i32),
                      shape_of("shapeOf4", input_info("input4"), data_types::i32),
                      strided_slice("strideSlice3", input_info("shapeOf3"), {2}, {4}, {1}, {0}, {1}, {0}, {0}, {0}, {2}),
                      strided_slice("strideSlice4", input_info("shapeOf4"), {2}, {4}, {1}, {0}, {1}, {0}, {0}, {0}, {2}),
                      prior_box("priorBox2",
                                inputs2,
                                out_size,
                                img_size,
                                attrs.min_size,
                                attrs.max_size,
                                attrs.aspect_ratio,
                                attrs.flip,
                                attrs.clip,
                                attrs.variance,
                                attrs.step,
                                attrs.offset,
                                attrs.scale_all_sizes,
                                attrs.fixed_ratio,
                                attrs.fixed_size,
                                attrs.density,
                                false),
                      reshape("reshape2", input_info("priorBox2"), input_info("pattern_id2"), false, ov::PartialShape::dynamic(3)),
                      concatenation("output", {input_info("reshape1"), input_info("reshape2")}, 0));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input1", input1_mem);
    network.set_input_data("input2", input2_mem);
    network.set_input_data("input3", input1_mem);
    network.set_input_data("input4", input2_mem);
    auto outputs = network.execute();
    auto output_mem = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_mem_ptr(output_mem, get_test_stream());

    std::vector<float> ref(28800);
    std::vector<int32_t> output_size = {30, 30};
    std::vector<int32_t> image_size = {224, 224};
    ov::reference::prior_box(output_size.data(), image_size.data(), ref.data(), ov::Shape{2, 14400}, attrs);
    std::vector<float> ref_concat = ref;
    ref_concat.insert(ref_concat.end(), ref.begin(), ref.end());

    for (size_t i = 0; i < output_mem->get_layout().get_linear_size(); ++i) {
        ASSERT_EQ(output_mem_ptr[i], ref_concat[i]);
    }
}
}  // is_valid_fusion_tests

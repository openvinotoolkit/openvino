// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/segment_max.hpp>

#include "test_utils.h"
#include "segment_max_inst.h"

using namespace cldnn;
using namespace ::tests;

namespace {

constexpr float REL_EPS = 2e-3f;
constexpr float ABS_EPS = 1e-5f;

namespace helpers {

template <typename T>
memory::ptr AllocateTensor(ov::PartialShape shape, const std::vector<T>& data) {
    const layout lo = {shape, ov::element::from<T>(), cldnn::format::bfyx};
    EXPECT_EQ(lo.get_linear_size(), data.size());
    memory::ptr tensor = get_test_engine().allocate_memory(lo);
    set_values<T>(tensor, data);
    return tensor;
}

template <typename T>
void CompareTypedBuffers(const memory::ptr& output, const std::vector<T>& expected, cldnn::stream& stream) {
    mem_lock<T> output_ptr(output, stream);
    ASSERT_EQ(output_ptr.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_TRUE(are_equal(expected[i], output_ptr[i], REL_EPS, ABS_EPS))
            << "at index " << i << " expected=" << expected[i] << " actual=" << output_ptr[i];
    }
}

}  // namespace helpers

// ----------- Test parameters -----------

struct SegmentMaxTestParams {
    ov::PartialShape data_shape;           // Static data shape
    std::vector<float> data_values;
    std::vector<int32_t> segment_ids;
    int fill_mode;                          // 0 = ZERO, 1 = LOWEST
    std::vector<float> expected_output;
    ov::PartialShape expected_output_shape; // Static expected output shape
    std::string test_name;
};

class segment_max_gpu_test : public ::testing::TestWithParam<SegmentMaxTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SegmentMaxTestParams>& obj) {
        return obj.param.test_name;
    }

    void Execute(const SegmentMaxTestParams& p) {
        auto stream = get_test_stream_ptr(get_test_default_config(engine_));

        // Allocate input memories
        auto data_mem = helpers::AllocateTensor<float>(p.data_shape, p.data_values);
        auto seg_mem = helpers::AllocateTensor<int32_t>(
            ov::PartialShape{static_cast<int64_t>(p.segment_ids.size())}, p.segment_ids);

        // Build topology with dynamic input layouts
        topology topology;

        auto data_dynamic_layout = layout{ov::PartialShape::dynamic(p.data_shape.rank().get_length()),
                                          data_types::f32, format::bfyx};
        auto seg_dynamic_layout = layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx};

        topology.add(input_layout("data", data_dynamic_layout));
        topology.add(input_layout("segment_ids", seg_dynamic_layout));

        // Create primitive with stored segment_ids_data for compile-time shape inference
        auto prim = segment_max("segment_max", input_info("data"), input_info("segment_ids"), p.fill_mode);
        // Store segment_ids values in the primitive so calc_output_layouts can compute static output shape
        prim.segment_ids_data.reserve(p.segment_ids.size());
        for (auto v : p.segment_ids) {
            prim.segment_ids_data.push_back(static_cast<int64_t>(v));
        }
        topology.add(prim);

        // Add reorder for output (required for dynamic shape pipeline)
        topology.add(reorder("output", input_info("segment_max"), format::bfyx, data_types::f32));

        // Configure for dynamic shapes
        ExecutionConfig config = get_test_default_config(engine_);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        cldnn::network::ptr network = get_network(engine_, topology, config, stream, false);
        network->set_input_data("data", data_mem);
        network->set_input_data("segment_ids", seg_mem);

        auto inst = network->get_primitive("segment_max");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto outputs = network->execute();
        auto output_mem = outputs.at("output").get_memory();

        // Validate output shape
        auto out_layout = output_mem->get_layout();
        ASSERT_EQ(out_layout.get_partial_shape(), p.expected_output_shape);

        // Validate output values
        helpers::CompareTypedBuffers<float>(output_mem, p.expected_output, get_test_stream());
    }

private:
    engine& engine_ = get_test_engine();
};

// ----------- Test cases -----------

std::vector<SegmentMaxTestParams> generateSegmentMaxTestParams() {
    std::vector<SegmentMaxTestParams> params;

    // 1D basic: 3 segments, no empty segments
    params.push_back({
        ov::PartialShape{6},                                           // data_shape
        {1.0f, 5.0f, 3.0f, 2.0f, 8.0f, 4.0f},                        // data_values
        {0, 0, 1, 2, 2, 2},                                           // segment_ids
        0,                                                              // fill_mode = ZERO
        {5.0f, 3.0f, 8.0f},                                           // expected_output
        ov::PartialShape{3},                                           // expected_output_shape
        "basic_1d_3seg"
    });

    // 1D with empty segment (fill_mode = ZERO)
    params.push_back({
        ov::PartialShape{4},
        {10.0f, 20.0f, 30.0f, 40.0f},
        {0, 0, 2, 2},                                                 // segment 1 is empty
        0,
        {20.0f, 0.0f, 40.0f},
        ov::PartialShape{3},
        "empty_seg_zero_fill"
    });

    // 1D with empty segment (fill_mode = LOWEST)
    params.push_back({
        ov::PartialShape{4},
        {10.0f, 20.0f, 30.0f, 40.0f},
        {0, 0, 2, 2},                                                 // segment 1 is empty
        1,
        {20.0f, std::numeric_limits<float>::lowest(), 40.0f},
        ov::PartialShape{3},
        "empty_seg_lowest_fill"
    });

    // 1D with negative values
    params.push_back({
        ov::PartialShape{5},
        {-3.0f, -1.0f, -5.0f, -2.0f, -4.0f},
        {0, 0, 1, 1, 1},
        0,
        {-1.0f, -2.0f},
        ov::PartialShape{2},
        "negative_values"
    });

    // 2D basic: data=[4,3], segment_ids=[4], 2 segments
    params.push_back({
        ov::PartialShape{4, 3},
        {1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f,
         7.0f, 8.0f, 9.0f,
         10.0f, 11.0f, 12.0f},
        {0, 0, 1, 1},
        0,
        {4.0f, 5.0f, 6.0f,           // max of rows 0,1
         10.0f, 11.0f, 12.0f},       // max of rows 2,3
        ov::PartialShape{2, 3},
        "basic_2d"
    });

    // 2D with empty segment (fill_mode = ZERO)
    params.push_back({
        ov::PartialShape{3, 2},
        {1.0f, 2.0f,
         3.0f, 4.0f,
         5.0f, 6.0f},
        {0, 2, 2},                                                    // segment 1 is empty
        0,
        {1.0f, 2.0f,                 // segment 0
         0.0f, 0.0f,                 // segment 1 (empty, ZERO fill)
         5.0f, 6.0f},               // segment 2
        ov::PartialShape{3, 2},
        "2d_empty_seg_zero"
    });

    // Single element per segment
    params.push_back({
        ov::PartialShape{3},
        {100.0f, 200.0f, 300.0f},
        {0, 1, 2},
        0,
        {100.0f, 200.0f, 300.0f},
        ov::PartialShape{3},
        "single_elem_per_seg"
    });

    // All elements in one segment
    params.push_back({
        ov::PartialShape{5},
        {3.0f, 1.0f, 4.0f, 1.0f, 5.0f},
        {0, 0, 0, 0, 0},
        0,
        {5.0f},
        ov::PartialShape{1},
        "all_one_segment"
    });

    return params;
}

TEST_P(segment_max_gpu_test, ref_comp_f32) {
    const auto& testParams = GetParam();
    Execute(testParams);
}

INSTANTIATE_TEST_SUITE_P(
    segment_max_gpu_test_suite,
    segment_max_gpu_test,
    testing::ValuesIn(generateSegmentMaxTestParams()),
    segment_max_gpu_test::getTestCaseName
);

}  // namespace

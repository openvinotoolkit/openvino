// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the SegmentMax cldnn primitive.
// Test cases are based on the SegmentMax-16 specification:
// https://docs.openvino.ai/nightly/documentation/openvino-ir-format/operation-sets/operation-specs/arithmetic/segment-max-16.html

#include <limits>

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
        if constexpr (std::is_floating_point_v<T>) {
            ASSERT_TRUE(are_equal(expected[i], output_ptr[i], REL_EPS, ABS_EPS))
                << "at index " << i << " expected=" << expected[i] << " actual=" << output_ptr[i];
        } else {
            ASSERT_EQ(expected[i], output_ptr[i])
                << "at index " << i << " expected=" << expected[i] << " actual=" << output_ptr[i];
        }
    }
}

}  // namespace helpers

// ============================================================================
// Test fixture for f32 data with i32 segment_ids (parameterized)
// ============================================================================

struct SegmentMaxTestParams {
    ov::PartialShape data_shape;
    std::vector<float> data_values;
    std::vector<int32_t> segment_ids;
    int fill_mode;                          // 0 = ZERO, 1 = LOWEST
    int64_t num_segments;                   // -1 = not set (infer from segment_ids)
    std::vector<float> expected_output;
    ov::PartialShape expected_output_shape;
    std::string test_name;
};

class SegmentMaxGpuTest : public ::testing::TestWithParam<SegmentMaxTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SegmentMaxTestParams>& obj) {
        return obj.param.test_name;
    }

    void Execute(const SegmentMaxTestParams& p) {
        auto stream = get_test_stream_ptr(get_test_default_config(engine_));

        auto data_mem = helpers::AllocateTensor<float>(p.data_shape, p.data_values);
        auto seg_mem = helpers::AllocateTensor<int32_t>(
            ov::PartialShape{static_cast<int64_t>(p.segment_ids.size())}, p.segment_ids);

        topology topology;

        auto data_dynamic_layout = layout{ov::PartialShape::dynamic(p.data_shape.rank().get_length()),
                                          data_types::f32, format::bfyx};
        auto seg_dynamic_layout = layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx};

        topology.add(input_layout("data", data_dynamic_layout));
        topology.add(input_layout("segment_ids", seg_dynamic_layout));

        auto prim = segment_max("segment_max", input_info("data"), input_info("segment_ids"), p.fill_mode);
        prim.segment_ids_data.reserve(p.segment_ids.size());
        for (auto v : p.segment_ids) {
            prim.segment_ids_data.push_back(static_cast<int64_t>(v));
        }
        if (p.num_segments >= 0) {
            prim.num_segments_val = p.num_segments;
        }
        topology.add(prim);

        topology.add(reorder("output", input_info("segment_max"), format::bfyx, data_types::f32));

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

        auto out_layout = output_mem->get_layout();
        ASSERT_EQ(out_layout.get_partial_shape(), p.expected_output_shape);

        helpers::CompareTypedBuffers<float>(output_mem, p.expected_output, *stream);
    }

private:
    engine& engine_ = get_test_engine();
};

// ----------- Test cases (f32 data, i32 segment_ids) -----------

std::vector<SegmentMaxTestParams> generateSegmentMaxTestParams() {
    std::vector<SegmentMaxTestParams> params;

    // --- Basic functionality ---

    // 1D basic: 3 segments, no empty segments
    params.push_back({
        ov::PartialShape{6},
        {1.0f, 5.0f, 3.0f, 2.0f, 8.0f, 4.0f},
        {0, 0, 1, 2, 2, 2},
        0, -1,                                                         // fill_mode=ZERO, num_segments=infer
        {5.0f, 3.0f, 8.0f},
        ov::PartialShape{3},
        "basic_1d_3seg"
    });

    // Single element per segment
    params.push_back({
        ov::PartialShape{3},
        {100.0f, 200.0f, 300.0f},
        {0, 1, 2},
        0, -1,
        {100.0f, 200.0f, 300.0f},
        ov::PartialShape{3},
        "single_elem_per_seg"
    });

    // All elements in one segment
    params.push_back({
        ov::PartialShape{5},
        {3.0f, 1.0f, 4.0f, 1.0f, 5.0f},
        {0, 0, 0, 0, 0},
        0, -1,
        {5.0f},
        ov::PartialShape{1},
        "all_one_segment"
    });

    // 1D with negative values
    params.push_back({
        ov::PartialShape{5},
        {-3.0f, -1.0f, -5.0f, -2.0f, -4.0f},
        {0, 0, 1, 1, 1},
        0, -1,
        {-1.0f, -2.0f},
        ov::PartialShape{2},
        "negative_values"
    });

    // --- Empty segment handling ---

    // 1D with empty segment (fill_mode = ZERO)
    params.push_back({
        ov::PartialShape{4},
        {10.0f, 20.0f, 30.0f, 40.0f},
        {0, 0, 2, 2},                                                 // segment 1 is empty
        0, -1,
        {20.0f, 0.0f, 40.0f},
        ov::PartialShape{3},
        "empty_seg_zero_fill"
    });

    // 1D with empty segment (fill_mode = LOWEST)
    params.push_back({
        ov::PartialShape{4},
        {10.0f, 20.0f, 30.0f, 40.0f},
        {0, 0, 2, 2},                                                 // segment 1 is empty
        1, -1,
        {20.0f, std::numeric_limits<float>::lowest(), 40.0f},
        ov::PartialShape{3},
        "empty_seg_lowest_fill"
    });

    // --- 2D data ---

    // 2D basic: data=[4,3], segment_ids=[4], 2 segments
    params.push_back({
        ov::PartialShape{4, 3},
        {1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f,
         7.0f, 8.0f, 9.0f,
         10.0f, 11.0f, 12.0f},
        {0, 0, 1, 1},
        0, -1,
        {4.0f, 5.0f, 6.0f,
         10.0f, 11.0f, 12.0f},
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
        0, -1,
        {1.0f, 2.0f,
         0.0f, 0.0f,
         5.0f, 6.0f},
        ov::PartialShape{3, 2},
        "2d_empty_seg_zero"
    });

    // 2D with empty segment (fill_mode = LOWEST) — Spec Example 3
    params.push_back({
        ov::PartialShape{3, 4},
        {1.0f, 2.0f, 3.0f, 4.0f,
         5.0f, 6.0f, 7.0f, 8.0f,
         9.0f, 10.0f, 11.0f, 12.0f},
        {0, 1, 1},                                                    // 2 segments, no empty
        1, -1,
        {1.0f, 2.0f, 3.0f, 4.0f,                                     // segment 0: row 0
         9.0f, 10.0f, 11.0f, 12.0f},                                  // segment 1: max(row 1, row 2)
        ov::PartialShape{2, 4},
        "2d_lowest_fill"
    });

    // --- num_segments tests (Spec Examples 1 & 2) ---

    // num_segments < max(segment_ids) + 1  (truncation) — Spec Example 1
    // segment_ids = [0, 0, 2, 3, 3] defines segments 0..3, but num_segments=2 truncates to [0, 1]
    params.push_back({
        ov::PartialShape{5},
        {1.0f, 5.0f, 3.0f, 2.0f, 8.0f},
        {0, 0, 2, 3, 3},
        0, 2,                                                          // num_segments=2 < 4
        {5.0f, 0.0f},                                                 // seg 0: max(1,5)=5, seg 1: empty=0
        ov::PartialShape{2},
        "num_segments_truncation"
    });

    // num_segments > max(segment_ids) + 1  (padding) — Spec Example 2
    // segment_ids = [0, 0, 2, 3, 3] defines segments 0..3, num_segments=8 pads with empty segments
    params.push_back({
        ov::PartialShape{5},
        {1.0f, 5.0f, 3.0f, 2.0f, 8.0f},
        {0, 0, 2, 3, 3},
        0, 8,                                                          // num_segments=8 > 4
        {5.0f, 0.0f, 3.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f},           // seg 4-7 are empty (ZERO fill)
        ov::PartialShape{8},
        "num_segments_padding_zero"
    });

    // num_segments > max(segment_ids) + 1 with fill_mode = LOWEST
    params.push_back({
        ov::PartialShape{3},
        {10.0f, 20.0f, 30.0f},
        {0, 1, 1},
        1, 5,                                                          // num_segments=5, LOWEST fill
        {10.0f, 30.0f,
         std::numeric_limits<float>::lowest(),
         std::numeric_limits<float>::lowest(),
         std::numeric_limits<float>::lowest()},
        ov::PartialShape{5},
        "num_segments_padding_lowest"
    });

    // num_segments == max(segment_ids) + 1  (exact match, same as default)
    params.push_back({
        ov::PartialShape{4},
        {1.0f, 2.0f, 3.0f, 4.0f},
        {0, 1, 1, 2},
        0, 3,                                                          // num_segments=3 == max(2)+1
        {1.0f, 3.0f, 4.0f},
        ov::PartialShape{3},
        "num_segments_exact"
    });

    // 2D + num_segments padding
    params.push_back({
        ov::PartialShape{3, 2},
        {1.0f, 2.0f,
         3.0f, 4.0f,
         5.0f, 6.0f},
        {0, 0, 1},
        0, 4,                                                          // num_segments=4, with padding
        {3.0f, 4.0f,                                                  // seg 0: max(row 0, row 1)
         5.0f, 6.0f,                                                  // seg 1: row 2
         0.0f, 0.0f,                                                  // seg 2: empty (ZERO)
         0.0f, 0.0f},                                                 // seg 3: empty (ZERO)
        ov::PartialShape{4, 2},
        "2d_num_segments_padding"
    });

    return params;
}

TEST_P(SegmentMaxGpuTest, ref_comp_f32) {
    const auto& testParams = GetParam();
    Execute(testParams);
}

INSTANTIATE_TEST_SUITE_P(
    SegmentMaxGpuTestSuite,
    SegmentMaxGpuTest,
    testing::ValuesIn(generateSegmentMaxTestParams()),
    SegmentMaxGpuTest::getTestCaseName
);

// ============================================================================
// Non-parameterized tests for i64 segment_ids and i32 data type
// ============================================================================

class SegmentMaxGpuTypeTest : public ::testing::Test {
protected:
    engine& engine_ = get_test_engine();
};

// Test with i64 segment_ids (spec allows i32 or i64 for T_IDX1)
TEST_F(SegmentMaxGpuTypeTest, i64_segment_ids) {
    auto stream = get_test_stream_ptr(get_test_default_config(engine_));

    std::vector<float> data_values = {1.0f, 5.0f, 3.0f, 2.0f, 8.0f, 4.0f};
    std::vector<int64_t> seg_ids = {0, 0, 1, 2, 2, 2};

    auto data_mem = helpers::AllocateTensor<float>(ov::PartialShape{6}, data_values);
    auto seg_mem = helpers::AllocateTensor<int64_t>(ov::PartialShape{6}, seg_ids);

    topology topology;
    topology.add(input_layout("data",
                              layout{ov::PartialShape::dynamic(1), data_types::f32, format::bfyx}));
    topology.add(input_layout("segment_ids",
                              layout{ov::PartialShape::dynamic(1), data_types::i64, format::bfyx}));

    auto prim = segment_max("segment_max", input_info("data"), input_info("segment_ids"), 0);
    prim.segment_ids_data = {0, 0, 1, 2, 2, 2};
    topology.add(prim);
    topology.add(reorder("output", input_info("segment_max"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine_);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto network = get_network(engine_, topology, config, stream, false);
    network->set_input_data("data", data_mem);
    network->set_input_data("segment_ids", seg_mem);

    auto outputs = network->execute();
    auto output_mem = outputs.at("output").get_memory();

    ASSERT_EQ(output_mem->get_layout().get_partial_shape(), ov::PartialShape{3});

    std::vector<float> expected = {5.0f, 3.0f, 8.0f};
    helpers::CompareTypedBuffers<float>(output_mem, expected, *stream);
}

// Test with i32 data type (spec: "any supported numerical data type")
TEST_F(SegmentMaxGpuTypeTest, i32_data) {
    auto stream = get_test_stream_ptr(get_test_default_config(engine_));

    std::vector<int32_t> data_values = {10, 50, 30, 20, 80, 40};
    std::vector<int32_t> seg_ids = {0, 0, 1, 2, 2, 2};

    auto data_mem = helpers::AllocateTensor<int32_t>(ov::PartialShape{6}, data_values);
    auto seg_mem = helpers::AllocateTensor<int32_t>(ov::PartialShape{6}, seg_ids);

    topology topology;
    topology.add(input_layout("data",
                              layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx}));
    topology.add(input_layout("segment_ids",
                              layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx}));

    auto prim = segment_max("segment_max", input_info("data"), input_info("segment_ids"), 0);
    prim.segment_ids_data = {0, 0, 1, 2, 2, 2};
    topology.add(prim);
    topology.add(reorder("output", input_info("segment_max"), format::bfyx, data_types::i32));

    ExecutionConfig config = get_test_default_config(engine_);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto network = get_network(engine_, topology, config, stream, false);
    network->set_input_data("data", data_mem);
    network->set_input_data("segment_ids", seg_mem);

    auto outputs = network->execute();
    auto output_mem = outputs.at("output").get_memory();

    ASSERT_EQ(output_mem->get_layout().get_partial_shape(), ov::PartialShape{3});

    std::vector<int32_t> expected = {50, 30, 80};
    helpers::CompareTypedBuffers<int32_t>(output_mem, expected, *stream);
}

// Test with i32 data type and fill_mode = LOWEST (empty segment)
TEST_F(SegmentMaxGpuTypeTest, i32_data_lowest_fill) {
    auto stream = get_test_stream_ptr(get_test_default_config(engine_));

    std::vector<int32_t> data_values = {10, 20, 30, 40};
    std::vector<int32_t> seg_ids = {0, 0, 2, 2};  // segment 1 is empty

    auto data_mem = helpers::AllocateTensor<int32_t>(ov::PartialShape{4}, data_values);
    auto seg_mem = helpers::AllocateTensor<int32_t>(ov::PartialShape{4}, seg_ids);

    topology topology;
    topology.add(input_layout("data",
                              layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx}));
    topology.add(input_layout("segment_ids",
                              layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx}));

    auto prim = segment_max("segment_max", input_info("data"), input_info("segment_ids"), 1);
    prim.segment_ids_data = {0, 0, 2, 2};
    topology.add(prim);
    topology.add(reorder("output", input_info("segment_max"), format::bfyx, data_types::i32));

    ExecutionConfig config = get_test_default_config(engine_);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto network = get_network(engine_, topology, config, stream, false);
    network->set_input_data("data", data_mem);
    network->set_input_data("segment_ids", seg_mem);

    auto outputs = network->execute();
    auto output_mem = outputs.at("output").get_memory();

    ASSERT_EQ(output_mem->get_layout().get_partial_shape(), ov::PartialShape{3});

    std::vector<int32_t> expected = {20, std::numeric_limits<int32_t>::lowest(), 40};
    helpers::CompareTypedBuffers<int32_t>(output_mem, expected, *stream);
}

}  // namespace

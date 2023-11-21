// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "intel_gpu/primitives/bucketize.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace tests;

namespace {

template <class I, class B, class O>
struct bucketize_test_inputs {
    std::vector<I> input_values;
    std::vector<B> buckets_values;
    std::vector<O> output_values_right_bound;
    std::vector<O> output_values_left_bound;
};

template <class I, class B, class O>
using bucketize_test_params = std::tuple<bucketize_test_inputs<I, B, O>, format::type, bool>;

template <class I, class B, class O>
struct bucketize_test : testing::TestWithParam<bucketize_test_params<I, B, O>> {
    void test() {
        format fmt = format::bfyx;
        bucketize_test_inputs<I, B, O> p;
        bool is_caching_test;
        std::tie(p, fmt, is_caching_test) = testing::TestWithParam<bucketize_test_params<I, B, O>>::GetParam();
        auto& engine = get_test_engine();

        const layout in_layout(ov::element::from<I>(),
                               format::bfyx,
                               tensor(format::bfyx, {1, 1, 1, static_cast<int>(p.input_values.size())}));
        auto input = engine.allocate_memory(in_layout);
        set_values(input, p.input_values);

        const layout buckets_layout(ov::element::from<B>(),
                                    format::bfyx,
                                    tensor(format::bfyx, {static_cast<int>(p.buckets_values.size()), 1, 1, 1}));
        auto buckets = engine.allocate_memory(buckets_layout);
        set_values(buckets, p.buckets_values);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(input_layout("buckets", buckets->get_layout()));
        topology.add(reorder("reordered_input", input_info("input"), fmt, ov::element::from<I>()));
        topology.add(reorder("reordered_buckets", input_info("buckets"), fmt, ov::element::from<B>()));

        topology.add(
            bucketize("bucketize_right_bound", { input_info("reordered_input"), input_info("buckets") }, ov::element::from<O>(), true));
        topology.add(
            bucketize("bucketize_left_bound", { input_info("reordered_input"), input_info("buckets") }, ov::element::from<O>(), false));
        topology.add(
            reorder("plane_bucketize_right_bound", input_info("bucketize_right_bound"), format::bfyx, ov::element::from<O>()));
        topology.add(
            reorder("plane_bucketize_left_bound", input_info("bucketize_left_bound"), format::bfyx, ov::element::from<O>()));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);
        network->set_input_data("buckets", buckets);
        const auto outputs = network->execute();

        {
            auto output = outputs.at("plane_bucketize_right_bound").get_memory();
            cldnn::mem_lock<O> output_ptr(output, get_test_stream());
            ASSERT_EQ(output_ptr.size(), p.output_values_right_bound.size());
            for (size_t i = 0; i < output_ptr.size(); ++i) {
                ASSERT_EQ(p.output_values_right_bound[i], output_ptr[i]);
            }
        }

        {
            auto output = outputs.at("plane_bucketize_left_bound").get_memory();
            cldnn::mem_lock<O> output_ptr(output, get_test_stream());
            ASSERT_EQ(output_ptr.size(), p.output_values_left_bound.size());
            for (size_t i = 0; i < output_ptr.size(); ++i) {
                ASSERT_EQ(p.output_values_left_bound[i], output_ptr[i]);
            }
        }
    }

    static std::string PrintToStringParamName(const testing::TestParamInfo<bucketize_test_params<I, B, O>>& info) {
        std::ostringstream result;
        result << "inType=" << ov::element::Type(ov::element::from<I>()) << "_";
        result << "bucketsType=" << ov::element::Type(ov::element::from<B>()) << "_";
        result << "outType=" << ov::element::Type(ov::element::from<O>()) << "_";
        result << "format=" << std::get<1>(info.param);
        result << "is_caching_test=" << std::get<2>(info.param);
        return result.str();
    }
};

template <class I, class B, class O>
std::vector<bucketize_test_inputs<I, B, O>> getBucketizeParams() {
    return {
        {{8, 1, 2, 1, 8, 5, 1, 5, 0, 20},  // Input values
         {1, 4, 10, 20},                   // Bucket values
         {2, 0, 1, 0, 2, 2, 0, 2, 0, 3},   // Output values with right bound
         {2, 1, 1, 1, 2, 2, 1, 2, 0, 4}},  // Output values with left bound
    };
}

template <class I, class B, class O>
std::vector<bucketize_test_inputs<I, B, O>> getBucketizeFloatingPointParams() {
    return {
        {{8.f, 1.f, 2.f, 1.1f, 8.f, 10.f, 1.f, 10.2f, 0.f, 20.f},  // Input values
         {1.f, 4.f, 10.f, 20.f},                                   // Bucket values
         {2, 0, 1, 1, 2, 2, 0, 3, 0, 3},                           // Output values with right bound
         {2, 1, 1, 1, 2, 3, 1, 3, 0, 4}},                          // Output values with left bound
    };
}

const std::vector<format::type> layout_formats = {format::bfyx,
                                                  format::bs_fs_yx_bsv32_fsv32,
                                                  format::bs_fs_yx_bsv32_fsv16,
                                                  format::b_fs_yx_fsv32,
                                                  format::b_fs_yx_fsv16,
                                                  format::bs_fs_yx_bsv16_fsv16};

#define INSTANTIATE_BUCKETIZE_TEST_SUITE(inputType, bucketsType, outType, func)                               \
    using bucketize_test_##inputType##bucketsType##outType = bucketize_test<inputType, bucketsType, outType>; \
    TEST_P(bucketize_test_##inputType##bucketsType##outType, test) {                                          \
        test();                                                                                               \
    }                                                                                                         \
    INSTANTIATE_TEST_SUITE_P(bucketize_smoke_##inputType##bucketsType##outType,                               \
                             bucketize_test_##inputType##bucketsType##outType,                                \
                             testing::Combine(testing::ValuesIn(func<inputType, bucketsType, outType>()),     \
                                              testing::ValuesIn(layout_formats),                              \
                                              testing::Values(false)),                                        \
                             bucketize_test_##inputType##bucketsType##outType::PrintToStringParamName);

INSTANTIATE_BUCKETIZE_TEST_SUITE(int8_t, int32_t, int32_t, getBucketizeParams)
INSTANTIATE_BUCKETIZE_TEST_SUITE(uint8_t, int64_t, int64_t, getBucketizeParams)
INSTANTIATE_BUCKETIZE_TEST_SUITE(int32_t, uint8_t, int32_t, getBucketizeParams)
INSTANTIATE_BUCKETIZE_TEST_SUITE(int64_t, int8_t, int64_t, getBucketizeParams)
INSTANTIATE_BUCKETIZE_TEST_SUITE(int64_t, int32_t, int32_t, getBucketizeParams)

using ov::float16;
INSTANTIATE_BUCKETIZE_TEST_SUITE(float, float16, int64_t, getBucketizeFloatingPointParams)
INSTANTIATE_BUCKETIZE_TEST_SUITE(float16, float, int32_t, getBucketizeFloatingPointParams)
INSTANTIATE_BUCKETIZE_TEST_SUITE(float, float, int64_t, getBucketizeFloatingPointParams)
INSTANTIATE_BUCKETIZE_TEST_SUITE(float16, float16, int32_t, getBucketizeFloatingPointParams)
INSTANTIATE_TEST_SUITE_P(export_import,
                         bucketize_test_float16float16int32_t,
                         testing::Combine(testing::ValuesIn(getBucketizeFloatingPointParams<ov::float16, ov::float16, int32_t>()),
                                          testing::Values(layout_formats[0]),
                                          testing::Values(true)),
                         bucketize_test_float16float16int32_t::PrintToStringParamName);

#undef INSTANTIATE_BUCKETIZE_TEST_SUITE

}  // namespace

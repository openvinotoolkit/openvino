// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <test_utils/test_utils.h>

#include <intel_gpu/primitives/unique.hpp>
#include <vector>

using namespace cldnn;
using namespace tests;

namespace {

template <typename vecElementType>
std::string vec2str(const std::vector<vecElementType>& vec) {
    if (!vec.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<vecElementType>(result, "."));
        result << vec.back() << ")";
        return result.str();
    }
    return "()";
}

template <class ElemT, class IndexT, class CountT>
struct unique_test_inputs {
    ov::Shape data_shape;
    std::vector<ElemT> input_data;
    std::vector<ElemT> expected_unique_values;
    std::vector<IndexT> expected_indices;
    std::vector<IndexT> expected_rev_indices;
    std::vector<CountT> expected_counts;
    bool flattened;
    int64_t axis;
    bool sorted;
};

template <class ElemT, class IndexT, class CountT>
using unique_test_params = std::tuple<unique_test_inputs<ElemT, IndexT, CountT>, format::type>;

template <class ElemT, class IndexT, class CountT>
struct unique_gpu_test : public testing::TestWithParam<unique_test_params<ElemT, IndexT, CountT>> {
public:
    void test() {
        const auto& [p, fmt] = testing::TestWithParam<unique_test_params<ElemT, IndexT, CountT>>::GetParam();

        auto& engine = get_test_engine();
        const auto elem_data_type = ov::element::from<ElemT>();
        const auto index_data_type = ov::element::from<IndexT>();
        const auto count_data_type = ov::element::from<CountT>();
        const auto plain_format = format::bfyx;

        const layout in_layout(p.data_shape, elem_data_type, plain_format);
        auto input = engine.allocate_memory(in_layout);
        set_values(input, p.input_data);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(reorder("reordered_input", input_info("input"), fmt, elem_data_type));
        topology.add(unique_count("unique_count", {input_info("reordered_input")}, p.flattened, p.axis));
        topology.add(unique_gather("unique_gather",
                                   {input_info("reordered_input"), input_info("unique_count")},
                                   p.flattened,
                                   p.axis,
                                   p.sorted,
                                   elem_data_type,
                                   index_data_type,
                                   count_data_type));
        topology.add(reorder("expected_unique_values", input_info("unique_gather", 0), plain_format, elem_data_type));
        topology.add(reorder("expected_indices", input_info("unique_gather", 1), plain_format, index_data_type));
        topology.add(reorder("expected_rev_indices", input_info("unique_gather", 2), plain_format, index_data_type));
        topology.add(reorder("expected_counts", input_info("unique_gather", 3), plain_format, count_data_type));

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network network(engine, topology, config);
        network.set_input_data("input", input);

        const auto outputs = network.execute();

        const auto expected_unique_values = outputs.at("expected_unique_values").get_memory();
        cldnn::mem_lock<ElemT> expected_unique_values_ptr(expected_unique_values, get_test_stream());
        ASSERT_EQ(expected_unique_values_ptr.size(), p.expected_unique_values.size());
        for (auto i = 0U; i < expected_unique_values_ptr.size(); ++i) {
            ASSERT_EQ(expected_unique_values_ptr[i], p.expected_unique_values[i]);
        }

        const auto expected_indices = outputs.at("expected_indices").get_memory();
        cldnn::mem_lock<IndexT> expected_indices_ptr(expected_indices, get_test_stream());
        ASSERT_EQ(expected_indices_ptr.size(), p.expected_indices.size());
        for (auto i = 0U; i < expected_indices_ptr.size(); ++i) {
            ASSERT_EQ(expected_indices_ptr[i], p.expected_indices[i]);
        }

        const auto expected_rev_indices = outputs.at("expected_rev_indices").get_memory();
        cldnn::mem_lock<IndexT> expected_rev_indices_ptr(expected_rev_indices, get_test_stream());
        ASSERT_EQ(expected_rev_indices_ptr.size(), p.expected_rev_indices.size());
        for (auto i = 0U; i < expected_rev_indices_ptr.size(); ++i) {
            ASSERT_EQ(expected_rev_indices_ptr[i], p.expected_rev_indices[i]);
        }

        const auto expected_counts = outputs.at("expected_counts").get_memory();
        cldnn::mem_lock<CountT> expected_counts_ptr(expected_counts, get_test_stream());
        ASSERT_EQ(expected_counts_ptr.size(), p.expected_counts.size());
        for (auto i = 0U; i < expected_counts_ptr.size(); ++i) {
            ASSERT_EQ(expected_counts_ptr[i], p.expected_counts[i]);
        }
    }

    static std::string PrintToStringParamName(
        const testing::TestParamInfo<unique_test_params<ElemT, IndexT, CountT>>& info) {
        const auto& [p, fmt] = info.param;

        std::ostringstream result;
        result << "data_shape=" << vec2str(p.data_shape) << "; ";
        result << "input_data=" << vec2str(p.input_data) << "; ";
        result << "data_type=" << ov::element::from<ElemT>() << "; ";
        result << "index_type=" << ov::element::from<IndexT>() << "; ";
        result << "counts_type=" << ov::element::from<CountT>() << "; ";
        result << "sorted=" << p.sorted << "; ";
        if (!p.flattened) {
            result << "axis=" << p.axis << "; ";
        }
        result << "fmt=" << fmt_to_str(fmt) << "; ";
        return result.str();
    }
};

template <class ElemT, class IndexT, class CountT>
std::vector<unique_test_inputs<ElemT, IndexT, CountT>> getUniqueParams() {
    return {
        {
            ov::Shape{5},
            std::vector<ElemT>{5, 4, 3, 2, 1},
            std::vector<ElemT>{5, 4, 3, 2, 1},
            std::vector<IndexT>{0, 1, 2, 3, 4},
            std::vector<IndexT>{0, 1, 2, 3, 4},
            std::vector<CountT>{1, 1, 1, 1, 1},
            true,
            0,
            false,
        },
        {
            ov::Shape{5},
            std::vector<ElemT>{5, 4, 3, 2, 1},
            std::vector<ElemT>{1, 2, 3, 4, 5},
            std::vector<IndexT>{4, 3, 2, 1, 0},
            std::vector<IndexT>{4, 3, 2, 1, 0},
            std::vector<CountT>{1, 1, 1, 1, 1},
            true,
            0,
            true,
        },
        {
            ov::Shape{7},
            std::vector<ElemT>{1, 3, 5, 3, 2, 4, 2},
            std::vector<ElemT>{1, 3, 5, 2, 4},
            std::vector<IndexT>{0, 1, 2, 4, 5},
            std::vector<IndexT>{0, 1, 2, 1, 3, 4, 3},
            std::vector<CountT>{1, 2, 1, 2, 1},
            true,
            0,
            false,
        },
        {
            ov::Shape{7},
            std::vector<ElemT>{1, 3, 5, 3, 2, 4, 2},
            std::vector<ElemT>{1, 2, 3, 4, 5},
            std::vector<IndexT>{0, 4, 1, 5, 2},
            std::vector<IndexT>{0, 2, 4, 2, 1, 3, 1},
            std::vector<CountT>{1, 2, 2, 1, 1},
            true,
            0,
            true,
        },
        {
            ov::Shape{7},
            std::vector<ElemT>{3, 1, 5, 3, 2, 4, 2},
            std::vector<ElemT>{1, 2, 3, 4, 5},
            std::vector<IndexT>{1, 4, 0, 5, 2},
            std::vector<IndexT>{2, 0, 4, 2, 1, 3, 1},
            std::vector<CountT>{1, 2, 2, 1, 1},
            true,
            0,
            true,
        },
        {
            ov::Shape{7},
            std::vector<ElemT>{3, 3, 5, 3, 2, 4, 2},
            std::vector<ElemT>{2, 3, 4, 5},
            std::vector<IndexT>{4, 0, 5, 2},
            std::vector<IndexT>{1, 1, 3, 1, 0, 2, 0},
            std::vector<CountT>{2, 3, 1, 1},
            true,
            0,
            true,
        },
        {
            ov::Shape{7},
            std::vector<ElemT>{1, 3, 5, 3, 2, 4, 2},
            std::vector<ElemT>{1, 2, 3, 4, 5},
            std::vector<IndexT>{0, 4, 1, 5, 2},
            std::vector<IndexT>{0, 2, 4, 2, 1, 3, 1},
            std::vector<CountT>{1, 2, 2, 1, 1},
            false,
            0,
            true,
        },
        {
            ov::Shape{2, 6},
            std::vector<ElemT>{3, 5, 3, 2, 4, 2, 1, 2, 3, 4, 5, 6},
            std::vector<ElemT>{3, 5, 2, 4, 1, 6},
            std::vector<IndexT>{0, 1, 3, 4, 6, 11},
            std::vector<IndexT>{0, 1, 0, 2, 3, 2, 4, 2, 0, 3, 1, 5},
            std::vector<CountT>{3, 2, 3, 2, 1, 1},
            true,
            0,
            false,
        },
        {
            ov::Shape{2, 4},
            std::vector<ElemT>{1, 2, 3, 4, 1, 2, 3, 5},
            std::vector<ElemT>{1, 2, 3, 4, 1, 2, 3, 5},
            std::vector<IndexT>{0, 1},
            std::vector<IndexT>{0, 1},
            std::vector<CountT>{1, 1},
            false,
            0,
            false,
        },
        {
            ov::Shape{2, 4},
            std::vector<ElemT>{1, 2, 3, 4, 1, 2, 3, 5},
            std::vector<ElemT>{1, 2, 3, 4, 1, 2, 3, 5},
            std::vector<IndexT>{0, 1, 2, 3},
            std::vector<IndexT>{0, 1, 2, 3},
            std::vector<CountT>{1, 1, 1, 1},
            false,
            1,
            false,
        },
        {
            ov::Shape{2, 4},
            std::vector<ElemT>{1, 2, 2, 4, 1, 2, 2, 5},
            std::vector<ElemT>{1, 2, 4, 1, 2, 5},
            std::vector<IndexT>{0, 1, 3},
            std::vector<IndexT>{0, 1, 1, 2},
            std::vector<CountT>{1, 2, 1},
            false,
            1,
            false,
        },
        {
            ov::Shape{2, 2, 3},
            std::vector<ElemT>{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6},
            std::vector<ElemT>{1, 2, 3, 4, 5, 6},
            std::vector<IndexT>{0},
            std::vector<IndexT>{0, 0},
            std::vector<CountT>{2},
            false,
            0,
            false,
        },
        {
            ov::Shape{2, 3, 2},
            std::vector<ElemT>{-3, -2, -5, 4, -3, 2, 3, -4, 1, 2, -1, 4},
            std::vector<ElemT>{-3, -2, -5, 4, -3, 2, 3, -4, 1, 2, -1, 4},
            std::vector<IndexT>{0, 1},
            std::vector<IndexT>{0, 1},
            std::vector<CountT>{1, 1},
            false,
            0,
            true,
        },
        {
            ov::Shape{2, 3, 2},
            std::vector<ElemT>{-3, -2, -5, 4, -3, 2, 3, -4, 1, 2, -1, 4},
            std::vector<ElemT>{-3, -2, -5, 4, -3, 2, 3, -4, 1, 2, -1, 4},
            std::vector<IndexT>{0, 1},
            std::vector<IndexT>{0, 1},
            std::vector<CountT>{1, 1},
            false,
            0,
            false,
        },
        {
            ov::Shape{2, 2, 3},
            std::vector<ElemT>{6, 5, 4, 6, 5, 4, 3, 2, 1, 3, 2, 1},
            std::vector<ElemT>{6, 5, 4, 3, 2, 1},
            std::vector<IndexT>{0},
            std::vector<IndexT>{0, 0},
            std::vector<CountT>{2},
            false,
            1,
            false,
        },
        {
            ov::Shape{2, 2, 3},
            std::vector<ElemT>{-1, 2, -1, 5, -3, 5, 7, -8, 7, 4, 4, 4},
            std::vector<ElemT>{-1, 2, 5, -3, 7, -8, 4, 4},
            std::vector<IndexT>{0, 1},
            std::vector<IndexT>{0, 1, 0},
            std::vector<CountT>{2, 1},
            false,
            2,
            false,
        },
        {
            ov::Shape{2, 2, 3},
            std::vector<ElemT>{-1, -1, 2, 5, 5, -3, 7, 7, -8, 4, 4, 4},
            std::vector<ElemT>{-1, 2, 5, -3, 7, -8, 4, 4},
            std::vector<IndexT>{0, 2},
            std::vector<IndexT>{0, 0, 1},
            std::vector<CountT>{2, 1},
            false,
            2,
            false,
        },
        {
            ov::Shape{2, 2, 3},
            std::vector<ElemT>{2, -1, -1, -3, 5, 5, -8, 7, 7, 4, 4, 4},
            std::vector<ElemT>{2, -1, -3, 5, -8, 7, 4, 4},
            std::vector<IndexT>{0, 1},
            std::vector<IndexT>{0, 1, 1},
            std::vector<CountT>{1, 2},
            false,
            2,
            false,
        },
        {
            ov::Shape{2, 2, 3},
            std::vector<ElemT>{2, -1, -1, -3, 5, 5, -8, 7, 7, 4, 4, 4},
            std::vector<ElemT>{-1, 2, 5, -3, 7, -8, 4, 4},
            std::vector<IndexT>{1, 0},
            std::vector<IndexT>{1, 0, 0},
            std::vector<CountT>{2, 1},
            false,
            2,
            true,
        },
        {
            ov::Shape{2, 2, 3},
            std::vector<ElemT>{-1, -1, -1, 3, 2, 2, 6, 7, 7, 4, 4, 4},
            std::vector<ElemT>{-1, -1, 2, 3, 7, 6, 4, 4},
            std::vector<IndexT>{1, 0},
            std::vector<IndexT>{1, 0, 0},
            std::vector<CountT>{2, 1},
            false,
            2,
            true,
        },
        {
            ov::Shape{1, 3, 16},
            std::vector<ElemT>{15,  -20, -11, 10, -21, 8,  -15, -10, 7,  20, -19, -14, -13, -16, -7,  -2,
                               -17, -4,  21,  -6, 11,  8,  17,  6,   7,  20, -3,  2,   -13, -16, -23, 14,
                               -1,  12,  5,   -6, 11,  -8, 1,   -10, 23, 20, -19, 18,  3,   -16, -7,  14},
            std::vector<ElemT>{-23, -21, -20, -19, -17, -16, -15, -14, -13, -11, -10, -8, -7, -6, -4, -3, -2, -1,
                               1,   2,   3,   5,   6,   7,   8,   10,  11,  12,  14,  15, 17, 18, 20, 21, 23},
            std::vector<IndexT>{30, 4,  1,  10, 16, 13, 6, 11, 12, 2,  7,  37, 14, 19, 17, 26, 15, 32,
                                38, 27, 44, 34, 23, 8,  5, 3,  20, 33, 31, 0,  22, 43, 9,  18, 40},
            std::vector<IndexT>{29, 2,  9,  25, 1,  24, 6,  10, 23, 32, 3,  7,  8,  5, 12, 16,
                                4,  14, 33, 13, 26, 24, 30, 22, 23, 32, 15, 19, 8,  5, 0,  28,
                                17, 27, 21, 13, 26, 11, 18, 10, 34, 32, 3,  31, 20, 5, 12, 28},
            std::vector<CountT>{1, 1, 1, 2, 1, 3, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 3, 1, 1},
            true,
            0,
            true,
        },
    };
}

const std::vector<format::type> layout_formats = {format::bfyx, format::b_fs_yx_fsv16};

#define INSTANTIATE_UNIQUE_TEST_SUITE(elem_type, index_type, count_type)                                               \
    using unique_gpu_test_##elem_type##index_type##count_type = unique_gpu_test<elem_type, index_type, count_type>;    \
    TEST_P(unique_gpu_test_##elem_type##index_type##count_type, test) {                                                \
        ASSERT_NO_FATAL_FAILURE(test());                                                                               \
    }                                                                                                                  \
    INSTANTIATE_TEST_SUITE_P(smoke_unique_##elem_type##index_type##count_type,                                         \
                             unique_gpu_test_##elem_type##index_type##count_type,                                      \
                             testing::Combine(testing::ValuesIn(getUniqueParams<elem_type, index_type, count_type>()), \
                                              testing::ValuesIn(layout_formats)),                                      \
                             unique_gpu_test_##elem_type##index_type##count_type::PrintToStringParamName);

INSTANTIATE_UNIQUE_TEST_SUITE(float, int64_t, int32_t);

}  // namespace

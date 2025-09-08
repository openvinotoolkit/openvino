// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/cum_sum.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "cum_sum_inst.h"

#include <algorithm>
#include <fstream>

using namespace cldnn;
using namespace ::tests;

template<typename T = float>
static std::vector<T> cumsum(const std::vector<T>& input,
                             const cldnn::format& format,
                             const std::vector<int>& shape,
                             const int axis = 0,
                             bool exclusive = false,
                             bool reverse = false) {
    std::vector<T> output(input.size());
    int dimNum = 0;
    std::vector<int> reordered_shape = shape;

    if (format == format::bfwzyx) {
        dimNum = 6;
    } else if (format == format::bfzyx) {
        dimNum = 5;
        for (int i = 2; i < dimNum; ++i) {
            reordered_shape[i] = shape[i + 1];
        }
    } else {
        dimNum = 4;
        for (int i = 2; i < dimNum; ++i) {
            reordered_shape[i] = shape[i + 2];
        }
    }

    std::vector<int> sizeDim(dimNum);
    sizeDim[dimNum - 1] = 1;
    for (size_t i = dimNum - 1, mult = 1; i > 0; --i) {
        mult *= reordered_shape[i];
        sizeDim[i - 1] = static_cast<int>(mult);
    }

    auto getFullIndex = [&sizeDim](int ind) {
        std::vector<int> fullInd(sizeDim.size());
        fullInd[0] = ind / sizeDim[0];
        for (int i = 1, numItems = 0; i < static_cast<int>(fullInd.size()); ++i) {
            numItems += fullInd[i - 1] * sizeDim[i - 1];
            fullInd[i] = (ind - numItems)/sizeDim[i];
        }
        return fullInd;
    };

    auto getIndex = [&sizeDim](std::vector<int> fullInd) {
        size_t index = 0;
        for (size_t i = 0; i < fullInd.size(); ++i) {
            index += fullInd[i] * sizeDim[i];
        }
        return index;
    };

    for (int i = 0; i < static_cast<int>(output.size()); ++i) {
        auto fullInd = getFullIndex(i);

        int stopInd = fullInd[axis] + 1;
        if (reverse) {
            stopInd = reordered_shape[axis];
            if (exclusive) {
                ++fullInd[axis];
            }
        } else {
            fullInd[axis] = 0;
            if (exclusive) {
                --stopInd;
            }
        }

        T res = (T)0;
        for (; fullInd[axis] < stopInd; ++fullInd[axis]) {
            auto ind = getIndex(fullInd);
            res += input[ind];
        }

        output[i] = (T)res;
    }
    return output;
}

template<typename T1, typename T2>
static std::vector<T1> vectorCast(const std::vector<T2>& vec) {
    std::vector<T1> ret(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        ret[i] = T1(vec[i]);
    }
    return ret;
}

#define CASE_CUM_SUM_AXIS_0 ::testing::Values(5), ::testing::Values(1), ::testing::Values(1), \
                            ::testing::Values(1), ::testing::Values(1), ::testing::Values(1), \
                            ::testing::Values(format::bfyx), ::testing::ValuesIn(axes[0]),    \
                            ::testing::ValuesIn(variants), ::testing::ValuesIn(variants), ::testing::Values(false)
#define CASE_CUM_SUM_AXIS_1 ::testing::Values(2), ::testing::Values(5), ::testing::Values(1), \
                            ::testing::Values(1), ::testing::Values(1), ::testing::Values(1), \
                            ::testing::Values(format::bfyx), ::testing::ValuesIn(axes[1]),    \
                            ::testing::ValuesIn(variants), ::testing::ValuesIn(variants), ::testing::Values(false)
#define CASE_CUM_SUM_AXIS_2 ::testing::Values(5), ::testing::Values(5), ::testing::Values(1), \
                            ::testing::Values(1), ::testing::Values(5), ::testing::Values(1), \
                            ::testing::Values(format::bfyx), ::testing::ValuesIn(axes[2]),    \
                            ::testing::ValuesIn(variants), ::testing::ValuesIn(variants), ::testing::Values(false)
#define CASE_CUM_SUM_AXIS_3 ::testing::Values(5), ::testing::Values(5), ::testing::Values(1), \
                            ::testing::Values(1), ::testing::Values(5), ::testing::Values(5), \
                            ::testing::Values(format::bfyx), ::testing::ValuesIn(axes[3]),    \
                            ::testing::ValuesIn(variants), ::testing::ValuesIn(variants), ::testing::Values(false)
#define CASE_CUM_SUM_AXIS_4 ::testing::Values(5), ::testing::Values(5), ::testing::Values(1), \
                            ::testing::Values(5), ::testing::Values(5), ::testing::Values(5), \
                            ::testing::Values(format::bfzyx), ::testing::ValuesIn(axes[4]),   \
                            ::testing::ValuesIn(variants), ::testing::ValuesIn(variants), ::testing::Values(false)
#define CASE_CUM_SUM_AXIS_5 ::testing::Values(5), ::testing::Values(5), ::testing::Values(5), \
                            ::testing::Values(5), ::testing::Values(5), ::testing::Values(5), \
                            ::testing::Values(format::bfwzyx), ::testing::ValuesIn(axes[5]),  \
                            ::testing::ValuesIn(variants), ::testing::ValuesIn(variants), ::testing::Values(false)

using cum_sum_test_params = std::tuple<int,            // batch
                                       int,            // feature
                                       int,            // w
                                       int,            // z
                                       int,            // y
                                       int,            // x
                                       cldnn::format,  // in_out format
                                       int,            // axis
                                       bool,           // exclusive
                                       bool,           // reverse
                                       bool>;          // is_caching_test

template <typename cum_sum_params, typename input_type = float, typename output_type = float>
class cum_sum_gpu : public ::testing::TestWithParam<cum_sum_params> {
public:
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    data_types get_alloc_data_type(void) {
        if (std::is_same<input_type, float>::value)
            return data_types::f32;
        else if (std::is_same<input_type, ov::float16>::value)
            return data_types::f16;
        else if (std::is_same<input_type, int32_t>::value)
            return data_types::i32;
        else if (std::is_same<input_type, int64_t>::value)
            return data_types::i64;
        else
            throw std::runtime_error("Unsupported cum sum data type in cum_sum_gpu_test.cpp");
    }

    void execute(cum_sum_params& p) {
        auto& engine = get_test_engine();

        auto b = std::get<0>(p);
        auto f = std::get<1>(p);
        auto w = std::get<2>(p);
        auto z = std::get<3>(p);
        auto y = std::get<4>(p);
        auto x = std::get<5>(p);
        tensor shape = tensor{ batch(b), feature(f), spatial(x, y, z, w) };

        auto in_out_format = std::get<6>(p);
        auto axis = std::get<7>(p);
        auto exclusive = std::get<8>(p);
        auto reverse = std::get<9>(p);
        bool is_caching_test = std::get<10>(p);

        auto input = engine.allocate_memory({ get_alloc_data_type(), in_out_format, shape });
        const int inputSize = b * f * w * z * y * x;
        VF<input_type> inputVals = std::is_same<input_type, ov::float16>::value ?
                                   rg.generate_random_1d<input_type>(inputSize, -1, 1, 1) :
                                   rg.generate_random_1d<input_type>(inputSize, -100, 100, 8);

        set_values(input, inputVals);

        topology topology;
        topology.add(input_layout("Input0", input->get_layout()));
        topology.add(cum_sum("cum_sum", input_info("Input0"), axis, exclusive, reverse));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "cum_sum");

        auto output = outputs.at("cum_sum").get_memory();
        cldnn::mem_lock<output_type> output_ptr(output, get_test_stream());

        auto answers = cumsum<output_type>(inputVals, in_out_format, { b, f, w, z, y, x }, axis, exclusive, reverse);
        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i) {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
        }
    }
};

class cum_sum_gpu_fp16 : public ::cum_sum_gpu<cum_sum_test_params, ov::float16, ov::float16> {};
class cum_sum_gpu_fp32 : public ::cum_sum_gpu<cum_sum_test_params, float, float> {};
class cum_sum_gpu_int32 : public ::cum_sum_gpu<cum_sum_test_params, int32_t, int32_t> {};
class cum_sum_gpu_int64 : public ::cum_sum_gpu<cum_sum_test_params, int64_t, int64_t> {};

TEST_P(cum_sum_gpu_fp16, basic) { auto p = GetParam(); execute(p); }
TEST_P(cum_sum_gpu_fp32, basic) { auto p = GetParam(); execute(p); }
TEST_P(cum_sum_gpu_int32, basic) { auto p = GetParam(); execute(p); }
TEST_P(cum_sum_gpu_int64, basic) { auto p = GetParam(); execute(p); }

namespace {
    std::vector<std::vector<int>> axes = {
        { 0 },
        { 0, 1 },
        { 0, 1, 2 },
        { 0, 1, 2, 3 },
        { 0, 1, 2, 3, 4 },
        { 0, 1, 2, 3, 4, 5 },
    };
    std::vector<bool> variants = { false, true };
}

INSTANTIATE_TEST_SUITE_P(axis_0, cum_sum_gpu_fp16, ::testing::Combine(CASE_CUM_SUM_AXIS_0));
INSTANTIATE_TEST_SUITE_P(axis_0, cum_sum_gpu_fp32, ::testing::Combine(CASE_CUM_SUM_AXIS_0));
INSTANTIATE_TEST_SUITE_P(axis_0, cum_sum_gpu_int32, ::testing::Combine(CASE_CUM_SUM_AXIS_0));
INSTANTIATE_TEST_SUITE_P(axis_0, cum_sum_gpu_int64, ::testing::Combine(CASE_CUM_SUM_AXIS_0));

INSTANTIATE_TEST_SUITE_P(axis_1, cum_sum_gpu_fp16, ::testing::Combine(CASE_CUM_SUM_AXIS_1));
INSTANTIATE_TEST_SUITE_P(axis_1, cum_sum_gpu_fp32, ::testing::Combine(CASE_CUM_SUM_AXIS_1));
INSTANTIATE_TEST_SUITE_P(axis_1, cum_sum_gpu_int32, ::testing::Combine(CASE_CUM_SUM_AXIS_1));
INSTANTIATE_TEST_SUITE_P(axis_1, cum_sum_gpu_int64, ::testing::Combine(CASE_CUM_SUM_AXIS_1));

INSTANTIATE_TEST_SUITE_P(axis_2, cum_sum_gpu_fp16, ::testing::Combine(CASE_CUM_SUM_AXIS_2));
INSTANTIATE_TEST_SUITE_P(axis_2, cum_sum_gpu_fp32, ::testing::Combine(CASE_CUM_SUM_AXIS_2));
INSTANTIATE_TEST_SUITE_P(axis_2, cum_sum_gpu_int32, ::testing::Combine(CASE_CUM_SUM_AXIS_2));
INSTANTIATE_TEST_SUITE_P(axis_2, cum_sum_gpu_int64, ::testing::Combine(CASE_CUM_SUM_AXIS_2));

INSTANTIATE_TEST_SUITE_P(axis_3, cum_sum_gpu_fp16, ::testing::Combine(CASE_CUM_SUM_AXIS_3));
INSTANTIATE_TEST_SUITE_P(axis_3, cum_sum_gpu_fp32, ::testing::Combine(CASE_CUM_SUM_AXIS_3));
INSTANTIATE_TEST_SUITE_P(axis_3, cum_sum_gpu_int32, ::testing::Combine(CASE_CUM_SUM_AXIS_3));
INSTANTIATE_TEST_SUITE_P(axis_3, cum_sum_gpu_int64, ::testing::Combine(CASE_CUM_SUM_AXIS_3));

INSTANTIATE_TEST_SUITE_P(axis_4, cum_sum_gpu_fp16, ::testing::Combine(CASE_CUM_SUM_AXIS_4));
INSTANTIATE_TEST_SUITE_P(axis_4, cum_sum_gpu_fp32, ::testing::Combine(CASE_CUM_SUM_AXIS_4));
INSTANTIATE_TEST_SUITE_P(axis_4, cum_sum_gpu_int32, ::testing::Combine(CASE_CUM_SUM_AXIS_4));
INSTANTIATE_TEST_SUITE_P(axis_4, cum_sum_gpu_int64, ::testing::Combine(CASE_CUM_SUM_AXIS_4));

INSTANTIATE_TEST_SUITE_P(axis_5, cum_sum_gpu_fp16, ::testing::Combine(CASE_CUM_SUM_AXIS_5));
INSTANTIATE_TEST_SUITE_P(axis_5, cum_sum_gpu_fp32, ::testing::Combine(CASE_CUM_SUM_AXIS_5));
INSTANTIATE_TEST_SUITE_P(axis_5, cum_sum_gpu_int32, ::testing::Combine(CASE_CUM_SUM_AXIS_5));
INSTANTIATE_TEST_SUITE_P(axis_5, cum_sum_gpu_int64, ::testing::Combine(CASE_CUM_SUM_AXIS_5));

INSTANTIATE_TEST_SUITE_P(export_import, cum_sum_gpu_int64,
    ::testing::Combine(::testing::Values(5), ::testing::Values(5), ::testing::Values(5),
                       ::testing::Values(5), ::testing::Values(5), ::testing::Values(5),
                       ::testing::Values(format::bfwzyx), ::testing::Values(axes[5][0]),
                       ::testing::Values(variants[0]), ::testing::Values(variants[0]),
                       ::testing::Values(true)));

// FIXME: This test fails on some driver versions. Looks like UB in impl or driver issue
TEST(cum_sum_gpu_f16, DISABLED_basic_1d) {
    // Input : 5x1x1x1
    // Output : 5x1x1x1

    auto& engine = get_test_engine();
    tensor shape = { 5, 1, 1, 1 };
    std::vector<float> inputVals = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f
    };
    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, shape });

    set_values(input, vectorCast<ov::float16>(inputVals));

    topology topology;
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(cum_sum("cum_sum", input_info("Input0")));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "cum_sum");

    auto output = outputs.at("cum_sum").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    auto answers = cumsum(inputVals, format::bfyx, { 5, 1, 1, 1, 1, 1 });

    ASSERT_EQ(output->count(), answers.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], half_to_float(output_ptr[i]))) << i;
    }
}

TEST(cum_sum_gpu_fp32, dynamic) {
    auto& engine = get_test_engine();
    ov::Shape shape = { 5, 1, 1, 1 };
    auto in_layout = layout{ov::PartialShape::dynamic(shape.size()), data_types::f32, format::bfyx};
    std::vector<float> input_data = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f
    };
    auto input = engine.allocate_memory(layout{ov::PartialShape(shape), data_types::f32, format::bfyx});

    set_values(input, input_data);

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(cum_sum("cum_sum", input_info("input")));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("cum_sum");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "cum_sum");

    auto output = outputs.at("cum_sum").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    auto answers = cumsum(input_data, format::bfyx, { 5, 1, 1, 1, 1, 1 });

    ASSERT_EQ(output->count(), answers.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(cum_sum_partial, big_shapes) {
    auto& engine = get_test_engine();

    const std::vector<size_t> input_sizes = {16, 17, 34, 65, 256, 300};

    for (const auto num_items : input_sizes) {
        std::vector<float> input_data;
        input_data.resize(num_items);
        std::iota(input_data.begin(), input_data.end(), 1);

        const auto shape_num_items = static_cast<int>(num_items);
        const tensor shape{shape_num_items, 1, 1, 1};
        const layout in_layout{data_types::f32, format::bfyx, shape};
        const auto input = engine.allocate_memory(in_layout);

        set_values(input, input_data);

        topology topology;
        topology.add(input_layout("input", in_layout));
        topology.add(cum_sum("cum_sum", input_info("input")));

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::force_implementations(
                ov::intel_gpu::ImplForcingMap{ {"cum_sum", {format::bfyx, "cum_sum_partial_sum"}} }));
        network network(engine, topology, config);
        network.set_input_data("input", input);

        const auto inst = network.get_primitive("cum_sum");
        const auto outputs = network.execute();
        ASSERT_EQ(outputs.begin()->first, "cum_sum");

        const auto output = outputs.at("cum_sum").get_memory();
        const cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        const auto expected = cumsum(input_data, format::bfyx, {shape_num_items, 1, 1, 1, 1, 1 });

        ASSERT_EQ(expected.size(), num_items);
        ASSERT_EQ(output->count(), num_items);
        for (size_t i = 0; i < num_items; ++i) {
            ASSERT_TRUE(are_equal(expected[i], output_ptr[i])) << "num_items=" << num_items << ", i=" << i;
        }
    }
}

TEST(cum_sum_partial, DISABLED_perf_test) {
    auto& engine = get_test_engine();

    const std::vector<size_t> input_sizes = {1, 2, 4, 8, 16, 17, 34, 48, 65, 256, 300, 515, 1025};

    for (const auto num_items : input_sizes) {
        std::vector<float> input_data;
        input_data.resize(num_items);
        std::iota(input_data.begin(), input_data.end(), 1);

        const auto shape_num_items = static_cast<int>(num_items);
        const tensor shape{shape_num_items, 1, 1, 1};
        const layout in_layout{data_types::f32, format::bfyx, shape};


        const auto input_ref = engine.allocate_memory(in_layout);
        set_values(input_ref, input_data);
        topology topology_ref;
        topology_ref.add(input_layout("input", in_layout));
        topology_ref.add(cum_sum("cum_sum", input_info("input")));
        ExecutionConfig config_ref(ov::enable_profiling(true));
        config_ref.set_property(ov::intel_gpu::force_implementations(
                ov::intel_gpu::ImplForcingMap{ {"cum_sum", {format::bfyx, "cum_sum_ref"}} }));
        network network_ref(engine, topology_ref, config_ref);
        network_ref.set_input_data("input", input_ref);


        const auto input_partial = engine.allocate_memory(in_layout);
        set_values(input_partial, input_data);
        topology topology_partial;
        topology_partial.add(input_layout("input", in_layout));
        topology_partial.add(cum_sum("cum_sum", input_info("input")));
        ExecutionConfig config_partial(ov::enable_profiling(true));
        config_partial.set_property(ov::intel_gpu::force_implementations(
                ov::intel_gpu::ImplForcingMap{ {"cum_sum", {format::bfyx, "cum_sum_partial_sum"}} }));
        network network_partial(engine, topology_partial, config_partial);
        network_partial.set_input_data("input", input_partial);


        std::map<primitive_id, network_output> output_ref;
        std::map<primitive_id, network_output> output_partial;

        constexpr int WARMUP_ROUNDS = 10;
        for (int i = 0; i < WARMUP_ROUNDS; ++i) {
            output_ref = network_ref.execute();
            output_partial = network_partial.execute();
        }

        constexpr int PERFTEST_ROUNDS = 100;
        double exectime_ref = 0.f;
        double exectime_partial = 0.f;
        for (int i = 0; i < PERFTEST_ROUNDS; ++i) {
            output_ref = network_ref.execute();
            const auto t_ref = get_profiling_exectime(output_ref, "cum_sum");
            exectime_ref += t_ref;

            output_partial = network_partial.execute();
            const auto t_partial = get_profiling_exectime(output_partial, "cum_sum");
            exectime_partial += t_partial;
        }
        exectime_ref /= PERFTEST_ROUNDS;
        exectime_partial /= PERFTEST_ROUNDS;

        std::cout << std::endl;
        std::cout << "Execution time for num_items=" << num_items << " "
                  << "cum_sum_ref" << " " << exectime_ref << std::endl;
        std::cout << "Execution time for num_items=" << num_items << " "
                  << "cum_sum_partial_sum" << " " << exectime_partial << std::endl;
    }
}

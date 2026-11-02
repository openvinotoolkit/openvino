// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/slice.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "primitive_inst.h"
#include "slice_inst.h"

#include <random>
#include <algorithm>
#include <vector>

using namespace cldnn;
using namespace ::tests;

namespace {

namespace helpers {

// data_types type traits:
template<typename T>
data_types ToDataType();

template<>
data_types ToDataType<float>() {return data_types::f32;}

template<>
data_types ToDataType<int32_t>() { return data_types::i32; }

template<>
data_types ToDataType<int64_t>()  { return data_types::i64; }

// Generates buffer of the length of shape_size(shape).
template<typename T>
std::vector<T> GenInput(const ov::PartialShape& shape) {
    const size_t size = ov::shape_size(shape.get_shape());
    std::vector<T> result(size);
    std::iota(result.begin(), result.end(), T{0});
    return result;
}
} // namespace helpers

struct SliceTestParams {
    memory::ptr input;
    memory::ptr start;
    memory::ptr stop;
    memory::ptr step;
    memory::ptr axes;
    memory::ptr wanted_output;
    bool is_input_dynamic = false;
    bool is_start_dynamic = false;
    bool is_stop_dynamic = false;
    bool is_step_dynamic = false;
    bool is_axes_dynamic = false;
    bool is_caching_test = false;
};

template<typename T>
class SliceTest : public ::testing::Test {
public:
    // Runs all test cases for given params.
    void RunAllTestCasesForParams(const SliceTestParams& params) {
        RunTestCase(params);
    }

    // Allocates tensoer with given shape and data.
    template<typename TDataType>
    memory::ptr AllocateTensor(ov::PartialShape shape, cldnn::format fmt,
                                const std::vector<TDataType>& data) {
        const layout lo = {shape, helpers::ToDataType<TDataType>(), fmt};
        EXPECT_EQ(lo.get_linear_size(), data.size());
        memory::ptr tensor = this->engine_.allocate_memory(lo);
        set_values<TDataType>(tensor, data);
        return tensor;
    }

    template<typename TypeParam>
    void FillWithBasicBfyxPositiveStepAxesLessThanRankData(SliceTestParams& params) {
        const ov::PartialShape input_shape{ 1, 2, 12, 100 };
        params.input = this->template AllocateTensor<TypeParam>(
            input_shape, format::bfyx, helpers::GenInput<TypeParam>(input_shape));
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{ 3 }, format::bfyx, { 1, 1, 0 });
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{ 3 }, format::bfyx, { 2, 120, 5 });
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{ 3 }, format::bfyx, { 1, 10, 1 });
        params.axes = this->template AllocateTensor<int64_t>(
            ov::PartialShape{ 3 }, format::bfyx, { 1, 3, 2 });
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{ 1, 1, 5, 10 }, format::bfyx, {
                1201, 1211, 1221, 1231, 1241, 1251, 1261, 1271, 1281, 1291,
                1301, 1311, 1321, 1331, 1341, 1351, 1361, 1371, 1381, 1391,
                1401, 1411, 1421, 1431, 1441, 1451, 1461, 1471, 1481, 1491,
                1501, 1511, 1521, 1531, 1541, 1551, 1561, 1571, 1581, 1591,
                1601, 1611, 1621, 1631, 1641, 1651, 1661, 1671, 1681, 1691,
            });
    }

private:
    void SetParameterInput(const std::string& name, topology& topology, const memory::ptr& data_ptr, bool is_dynamic ) {
        if(is_dynamic) {
            auto dynamic_shape = data_ptr->get_layout();
            dynamic_shape.set_partial_shape(ov::PartialShape::dynamic(dynamic_shape.get_rank()));
            topology.add(input_layout(name, dynamic_shape));
        } else {
            topology.add(data(name, data_ptr));
        }
    }

    static std::vector<int64_t> ReadConstant(const memory::ptr& mem) {
        cldnn::mem_lock<int64_t, mem_lock_type::read> lock(mem, get_test_stream());
        return std::vector<int64_t>(lock.data(), lock.data() + lock.size());
    }

    // Runs single tests case for given params.
    void RunTestCase(const SliceTestParams& params) {

        auto dynamic_input = params.input->get_layout();
        dynamic_input.set_partial_shape(ov::PartialShape::dynamic(dynamic_input.get_rank()));
        topology topology;
        topology.add(input_layout("input", params.is_input_dynamic ? dynamic_input : params.input->get_layout()));

        SetParameterInput("start", topology, params.start, params.is_start_dynamic);
        SetParameterInput("stop", topology, params.stop, params.is_stop_dynamic);
        SetParameterInput("step", topology, params.step, params.is_step_dynamic);

        if(params.axes) {
            SetParameterInput("axes", topology, params.axes, params.is_axes_dynamic);
        }

        std::vector<input_info> inputs{input_info("input"),
                                       input_info("start"),
                                       input_info("stop"),
                                       input_info("step")};
        if (params.axes) {
            inputs.push_back(input_info("axes"));
        }

        int32_t non_constant_input_mask = 0;
        std::vector<int64_t> start_constant;
        std::vector<int64_t> step_constant;
        std::vector<int64_t> axes_constant;
        if (params.is_start_dynamic) {
            non_constant_input_mask |= slice::SLICE_NON_CONST_INPUT::START;
        } else {
            start_constant = ReadConstant(params.start);
        }
        if (params.is_step_dynamic) {
            non_constant_input_mask |= slice::SLICE_NON_CONST_INPUT::STEP;
        } else {
            step_constant = ReadConstant(params.step);
        }
        if (params.axes) {
            if (params.is_axes_dynamic) {
                non_constant_input_mask |= slice::SLICE_NON_CONST_INPUT::AXES;
            } else {
                axes_constant = ReadConstant(params.axes);
            }
        }
        topology.add(slice("slice", inputs, non_constant_input_mask, start_constant, step_constant, axes_constant));

        ExecutionConfig config = get_test_default_config(engine_);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        cldnn::network::ptr network =
            get_network(engine_, topology, config, get_test_stream_ptr(), params.is_caching_test);

        network->set_input_data("input", params.input);

        if (params.is_start_dynamic)
            network->set_input_data("start", params.start);
        if (params.is_stop_dynamic)
            network->set_input_data("stop", params.stop);
        if (params.is_step_dynamic)
            network->set_input_data("step", params.step);
        if(params.axes && params.is_axes_dynamic) {
             network->set_input_data("axes", params.axes);
        }

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "slice");

        auto output = outputs.at("slice").get_memory();

        cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());
        cldnn::mem_lock<T> wanted_output_ptr(params.wanted_output, get_test_stream());

        ASSERT_EQ(output->get_layout(), params.wanted_output->get_layout());
        ASSERT_EQ(output_ptr.size(), wanted_output_ptr.size());
        for (size_t i = 0; i < output_ptr.size(); ++i)
            ASSERT_TRUE(are_equal(wanted_output_ptr[i], output_ptr[i], 2e-3));
    }

    engine& engine_ = get_test_engine();
};

using testing::Types;
typedef Types<float, int32_t, int64_t> DataTypes;
TYPED_TEST_SUITE(SliceTest, DataTypes);

TYPED_TEST(SliceTest, bfyx_positive_step) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, bfyx_positive_step_all_static_caching) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    params.is_caching_test = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, bfyx_positive_step_all_dynamic_caching) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    params.is_input_dynamic = true;
    params.is_start_dynamic = true;
    params.is_step_dynamic = true;
    params.is_stop_dynamic = true;
    params.is_axes_dynamic = true;
    params.is_caching_test = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, input_dynamic) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    params.is_input_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, stop_dynamic) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    params.is_stop_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, step_dynamic) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    params.is_step_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, stop_step_dynamic) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    params.is_step_dynamic = true;
    params.is_stop_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, input_stop_step_dynamic) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    params.is_input_dynamic = true;
    params.is_step_dynamic = true;
    params.is_stop_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, start_dynamic) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    params.is_start_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, input_start_stop_step_dynamic) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    params.is_input_dynamic = true;
    params.is_start_dynamic = true;
    params.is_step_dynamic = true;
    params.is_stop_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, axes_dynamic) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    params.is_axes_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, input_axes_dynamic) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    params.is_input_dynamic = true;
    params.is_axes_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, input_step_axes_dynamic) {
    SliceTestParams params;
    this->template FillWithBasicBfyxPositiveStepAxesLessThanRankData<TypeParam>(params);
    params.is_input_dynamic = true;
    params.is_step_dynamic = true;
    params.is_axes_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, bfyx_negative_step) {
    SliceTestParams params;
    const ov::PartialShape input_shape{ 1, 2, 12, 100 };
    params.input = this->template AllocateTensor<TypeParam>(
        input_shape, format::bfyx, helpers::GenInput<TypeParam>(input_shape));
    params.start = this->template AllocateTensor<int64_t>(
        ov::PartialShape{ 4 }, format::bfyx, { 0, 1, 5, 90 });
    params.stop = this->template AllocateTensor<int64_t>(
        ov::PartialShape{ 4 }, format::bfyx, { 1, 0, 0, 10 });
    params.step = this->template AllocateTensor<int64_t>(
        ov::PartialShape{ 4 }, format::bfyx, { 1, -1, -1, -10 });
    params.wanted_output = this->template AllocateTensor<TypeParam>(
        ov::PartialShape{ 1, 1, 5, 8 }, format::bfyx, {
            1789, 1779, 1769, 1759, 1749, 1739, 1729, 1719,
            1689, 1679, 1669, 1659, 1649, 1639, 1629, 1619,
            1589, 1579, 1569, 1559, 1549, 1539, 1529, 1519,
            1489, 1479, 1469, 1459, 1449, 1439, 1429, 1419,
            1389, 1379, 1369, 1359, 1349, 1339, 1329, 1319
        });

    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceTest, bfzyx) {
    SliceTestParams params;
    const ov::PartialShape input_shape{ 2, 3, 10, 12, 5 };
    params.input = this->template AllocateTensor<TypeParam>(
        input_shape, format::bfzyx, helpers::GenInput<TypeParam>(input_shape));
    params.start = this->template AllocateTensor<int64_t>(
        ov::PartialShape{ 5 }, format::bfzyx, { 0, 0, 0, 0, 0 });
    params.stop = this->template AllocateTensor<int64_t>(
        ov::PartialShape{ 5 }, format::bfzyx, { 1, 2, 2, 2, 2 });
    params.step = this->template AllocateTensor<int64_t>(
        ov::PartialShape{ 5 }, format::bfzyx, { 1, 1, 1, 1, 1 });
    params.wanted_output = this->template AllocateTensor<TypeParam>(
        ov::PartialShape{ 1, 2, 2, 2, 2 }, format::bfzyx, {
            0,   1,   5,   6,   60,  61,  65,  66,
            600, 601, 605, 606, 660, 661, 665, 666
        });

    params.is_caching_test = true;

    this->RunAllTestCasesForParams(params);
    params.is_input_dynamic = true;
    params.is_start_dynamic = true;
    params.is_step_dynamic = true;
    params.is_stop_dynamic = true;
    params.is_axes_dynamic = true;

    this->RunAllTestCasesForParams(params);
}

void run_slice_gpu_different_jit_kernels(bool is_caching_test) {
    auto& engine = get_test_engine();
    const ov::PartialShape input_shape{1, 2, 2, 4};
    auto input = engine.allocate_memory({input_shape, data_types::i32, format::bfyx});
    set_values<int32_t>(input, helpers::GenInput<int32_t>(input_shape));

    auto start = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    auto stop = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    auto step = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    auto axes = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    set_values<int64_t>(start, {1, 0, 0});
    set_values<int64_t>(stop, {2, 2, 4});
    set_values<int64_t>(step, {1, 1, 2});
    set_values<int64_t>(axes, {1, 2, 3});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("start", start));
    topology.add(data("stop", stop));
    topology.add(data("step", step));
    topology.add(data("axes", axes));
    topology.add(
        slice("slice_static", {input_info("input"), input_info("start"), input_info("stop"), input_info("step"), input_info("axes")}, 0, {1, 0, 0}, {1, 1, 2}, {1, 2, 3}));

    topology.add(input_layout("dynamic_start", start->get_layout()));
    topology.add(input_layout("dynamic_step", step->get_layout()));
    topology.add(slice("slice_dynamic",
                       {input_info("input"), input_info("dynamic_start"), input_info("stop"), input_info("dynamic_step"), input_info("axes")},
                       slice::SLICE_NON_CONST_INPUT::START | slice::SLICE_NON_CONST_INPUT::STEP,
                       {},
                       {},
                       {1, 2, 3}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    network->set_input_data("dynamic_start", start);
    network->set_input_data("dynamic_step", step);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(2));

    const auto static_slice = network->get_primitive("slice_static");
    const auto dynamic_slice = network->get_primitive("slice_dynamic");
    ASSERT_NE(static_slice, nullptr);
    ASSERT_NE(dynamic_slice, nullptr);
    ASSERT_NE(static_slice->get_impl(), nullptr);
    ASSERT_NE(dynamic_slice->get_impl(), nullptr);
    EXPECT_EQ(static_slice->get_impl()->get_kernel_name(), dynamic_slice->get_impl()->get_kernel_name());
    EXPECT_NE(static_slice->get_impl(), dynamic_slice->get_impl());
    EXPECT_NE(static_slice->desc()->hash(), dynamic_slice->desc()->hash());
    EXPECT_NE(static_slice->get_impl_params()->hash(), dynamic_slice->get_impl_params()->hash());

    const std::vector<int32_t> expected = {8, 10, 12, 14};

    const layout expected_layout{ov::PartialShape{1, 1, 2, 2}, data_types::i32, format::bfyx};
    for (const auto& output_id : {"slice_static", "slice_dynamic"}) {
        auto output = outputs.at(output_id).get_memory();
        cldnn::mem_lock<int32_t, mem_lock_type::read> output_ptr(output, get_test_stream());
        ASSERT_EQ(output->get_layout(), expected_layout);
        for (size_t i = 0; i < expected.size(); ++i)
            ASSERT_EQ(expected[i], output_ptr[i]) << "i=" << i;
    }
}

TEST(slice_gpu_jit_kernel, different_slices_not_cached) {
    run_slice_gpu_different_jit_kernels(false);
}

TEST(slice_gpu_jit_kernel, different_slices_cached) {
    run_slice_gpu_different_jit_kernels(true);
}

// constant value should be baked into hash
void run_slice_gpu_static_slices_different_value(bool is_caching_test, int64_t start1_val, int64_t stop1_val, int64_t start2_val, int64_t stop2_val) {
    const std::vector<int64_t> start1_vals{start1_val};
    const std::vector<int64_t> stop1_vals{stop1_val};
    const std::vector<int64_t> start2_vals{start2_val};
    const std::vector<int64_t> stop2_vals{stop2_val};
    auto& engine = get_test_engine();
    const ov::PartialShape input_shape{1, 2, 2, 4};
    auto input = engine.allocate_memory({input_shape, data_types::i32, format::bfyx});
    set_values<int32_t>(input, helpers::GenInput<int32_t>(input_shape));

    auto start1 = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto stop1 = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto start2 = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto stop2 = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto step = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto axes = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    set_values<int64_t>(start1, start1_vals);
    set_values<int64_t>(stop1, stop1_vals);
    set_values<int64_t>(start2, start2_vals);
    set_values<int64_t>(stop2, stop2_vals);
    set_values<int64_t>(step, {1});
    set_values<int64_t>(axes, {1});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("start1", start1));
    topology.add(data("stop1", stop1));
    topology.add(data("start2", start2));
    topology.add(data("stop2", stop2));
    topology.add(data("step", step));
    topology.add(data("axes", axes));
    topology.add(
        slice("slice_1", {input_info("input"), input_info("start1"), input_info("stop1"), input_info("step"), input_info("axes")}, 0, start1_vals, {1}, {1}));
    topology.add(
        slice("slice_2", {input_info("input"), input_info("start2"), input_info("stop2"), input_info("step"), input_info("axes")}, 0, start2_vals, {1}, {1}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(2));

    const auto slice_1 = network->get_primitive("slice_1");
    const auto slice_2 = network->get_primitive("slice_2");
    ASSERT_NE(slice_1, nullptr);
    ASSERT_NE(slice_2, nullptr);
    // stop not considered in hash, only check start.
    if (start1_val == start2_val) {
        EXPECT_EQ(slice_1->desc()->hash(), slice_2->desc()->hash());
        EXPECT_EQ(*slice_1->desc(), *slice_2->desc());
    } else {
        EXPECT_NE(slice_1->desc()->hash(), slice_2->desc()->hash());
        EXPECT_NE(*slice_1->desc(), *slice_2->desc());
    }
    EXPECT_NE(slice_1->get_impl_params()->hash(), slice_2->get_impl_params()->hash());
    EXPECT_FALSE(*slice_1->get_impl_params() == *slice_2->get_impl_params());

    // axis 2,3 of the {1, 2, 2, 4} input
    const int32_t elems_per_index = 2 * 4;
    std::vector<int32_t> expected1(static_cast<size_t>(elems_per_index * (stop1_val - start1_val)));
    std::vector<int32_t> expected2(static_cast<size_t>(elems_per_index * (stop2_val - start2_val)));
    std::iota(expected1.begin(), expected1.end(), elems_per_index * static_cast<int32_t>(start1_val));
    std::iota(expected2.begin(), expected2.end(), elems_per_index * static_cast<int32_t>(start2_val));

    {
        auto output1 = outputs.at("slice_1").get_memory();
        cldnn::mem_lock<int32_t, mem_lock_type::read> output_ptr(output1, get_test_stream());
        for (size_t i = 0; i < expected1.size(); ++i)
            EXPECT_EQ(expected1[i], output_ptr[i]) << "slice_1 i=" << i;
    }
    {
        auto output2 = outputs.at("slice_2").get_memory();
        cldnn::mem_lock<int32_t, mem_lock_type::read> output_ptr(output2, get_test_stream());
        for (size_t i = 0; i < expected2.size(); ++i)
            EXPECT_EQ(expected2[i], output_ptr[i]) << "slice_2 i=" << i;
    }
}

TEST(slice_gpu_jit_kernel, static_slices_different_value_not_cached) {
    run_slice_gpu_static_slices_different_value(false, 0, 1, 1, 2);
}

TEST(slice_gpu_jit_kernel, static_slices_different_value_cached) {
    run_slice_gpu_static_slices_different_value(true, 0, 1, 1, 2);
}

TEST(slice_gpu_jit_kernel, static_slices_same_start_different_stop_not_cached) {
    run_slice_gpu_static_slices_different_value(false, 0, 1, 0, 2);
}

TEST(slice_gpu_jit_kernel, static_slices_same_start_different_stop_cached) {
    run_slice_gpu_static_slices_different_value(true, 0, 1, 0, 2);
}

// constness is decided by mask, not actual input const-ness
void run_slice_gpu_constant_inputs_with_nonconst_mask(bool is_caching_test) {
    auto& engine = get_test_engine();
    const ov::PartialShape input_shape{1, 2, 2, 4};
    auto input = engine.allocate_memory({input_shape, data_types::i32, format::bfyx});
    set_values<int32_t>(input, helpers::GenInput<int32_t>(input_shape));

    auto start = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    auto stop = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    auto step = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    auto axes = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    set_values<int64_t>(start, {1, 0, 0});
    set_values<int64_t>(stop, {2, 2, 4});
    set_values<int64_t>(step, {1, 1, 2});
    set_values<int64_t>(axes, {1, 2, 3});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("start", start));
    topology.add(data("stop", stop));
    topology.add(data("step", step));
    topology.add(data("axes", axes));
    topology.add(slice("slice_non_const_mask",
                       {input_info("input"), input_info("start"), input_info("stop"), input_info("step"), input_info("axes")},
                       slice::SLICE_NON_CONST_INPUT::START | slice::SLICE_NON_CONST_INPUT::STEP | slice::SLICE_NON_CONST_INPUT::AXES));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

    const auto slice_inst = network->get_primitive("slice_non_const_mask");
    ASSERT_NE(slice_inst, nullptr);
    const auto& slice_node_ref = slice_inst->get_node().as<slice>();
    ASSERT_TRUE(slice_node_ref.get_dependency(SliceKernelRefNeededInputs::kStart).is_constant());
    ASSERT_TRUE(slice_node_ref.get_dependency(SliceKernelRefNeededInputs::kStep).is_constant());
    ASSERT_TRUE(slice_node_ref.get_dependency(SliceKernelRefNeededInputs::kAxes).is_constant());

    const auto needed_inputs = SliceKernelRefNeededInputs::Create(slice_node_ref);
    EXPECT_TRUE(needed_inputs.IsInputNeededInRuntime(SliceKernelRefNeededInputs::kStart));
    EXPECT_TRUE(needed_inputs.IsInputNeededInRuntime(SliceKernelRefNeededInputs::kStep));
    EXPECT_TRUE(needed_inputs.IsInputNeededInRuntime(SliceKernelRefNeededInputs::kAxes));
    EXPECT_EQ(needed_inputs.GetNeededInputIndexes(),
              (std::vector<size_t>{SliceKernelRefNeededInputs::kData, SliceKernelRefNeededInputs::kStart,
                                   SliceKernelRefNeededInputs::kStep, SliceKernelRefNeededInputs::kAxes}));
}

TEST(slice_gpu_jit_kernel, constant_inputs_with_nonconst_mask_not_cached) {
    run_slice_gpu_constant_inputs_with_nonconst_mask(false);
}

TEST(slice_gpu_jit_kernel, constant_inputs_with_nonconst_mask_cached) {
    run_slice_gpu_constant_inputs_with_nonconst_mask(true);
}

// Constant-folding does not replace user node, so slice still uses runtime input
TEST(slice_gpu_jit_kernel, constant_folding_inputs) {
    auto& engine = get_test_engine();
    const ov::PartialShape input_shape{1, 2, 2, 4};
    auto input = engine.allocate_memory({input_shape, data_types::i32, format::bfyx});
    set_values<int32_t>(input, helpers::GenInput<int32_t>(input_shape));

    auto start_base = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    auto start_offset = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    auto stop = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    auto step = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    auto axes = engine.allocate_memory({ov::PartialShape{3}, data_types::i64, format::bfyx});
    set_values<int64_t>(start_base, {0, 0, 0});
    set_values<int64_t>(start_offset, {1, 0, 0});
    set_values<int64_t>(stop, {2, 2, 4});
    set_values<int64_t>(step, {1, 1, 2});
    set_values<int64_t>(axes, {1, 2, 3});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("start_base", start_base));
    topology.add(data("start_offset", start_offset));
    topology.add(eltwise("start_folded", {input_info("start_base"), input_info("start_offset")}, eltwise_mode::sum));
    topology.add(data("stop", stop));
    topology.add(data("step", step));
    topology.add(data("axes", axes));
    topology.add(slice("slice_folded",
                       {input_info("input"), input_info("start_folded"), input_info("stop"), input_info("step"), input_info("axes")},
                       slice::SLICE_NON_CONST_INPUT::START,
                       {},
                       {1, 1, 2},
                       {1, 2, 3}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

    const auto slice_inst = network->get_primitive("slice_folded");
    ASSERT_NE(slice_inst, nullptr);
    const auto& slice_node_ref = slice_inst->get_node().as<slice>();
    ASSERT_TRUE(slice_node_ref.get_dependency(SliceKernelRefNeededInputs::kStart).is_constant());
    ASSERT_TRUE(slice_node_ref.get_dependency(SliceKernelRefNeededInputs::kStart).is_type<data>());

    const auto needed_inputs = SliceKernelRefNeededInputs::Create(slice_node_ref);
    EXPECT_TRUE(needed_inputs.IsInputNeededInRuntime(SliceKernelRefNeededInputs::kStart));
    EXPECT_EQ(needed_inputs.GetNeededInputIndexes(),
              (std::vector<size_t>{SliceKernelRefNeededInputs::kData, SliceKernelRefNeededInputs::kStart}));
}

// int8 Slice: dedicated small-value case (the shared TYPED_TEST data exceeds the i8 range). Covers the
// i8 output path enabled for a quantized KV cache (GroupQueryAttention int8/int4 dynamic Slice+Concat).
TEST(slice_gpu_i8, bfyx) {
    auto& engine = get_test_engine();

    // input [1,1,2,4] i8, slice axis 2 -> [1,1,1,4] (second row).
    auto input = engine.allocate_memory({ov::PartialShape{1, 1, 2, 4}, data_types::i8, format::bfyx});
    auto start = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto stop = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto step = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto axes = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});

    set_values<int8_t>(input, {1, 2, 3, 4, -5, -6, -7, -8});
    set_values<int64_t>(start, {1});
    set_values<int64_t>(stop, {2});
    set_values<int64_t>(step, {1});
    set_values<int64_t>(axes, {2});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("start", start));
    topology.add(data("stop", stop));
    topology.add(data("step", step));
    topology.add(data("axes", axes));
    topology.add(slice("slice", {input_info("input"), input_info("start"), input_info("stop"), input_info("step"), input_info("axes")}, 0, {1}, {1}, {2}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cldnn::network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("slice").get_memory();
    cldnn::mem_lock<int8_t, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<int8_t> expected_results = {-5, -6, -7, -8};
    ASSERT_EQ(output_ptr.size(), expected_results.size());
    for (size_t i = 0; i < expected_results.size(); ++i)
        ASSERT_EQ(expected_results[i], output_ptr[i]) << "i=" << i;
}

// uint8 Slice: dedicated small-value case, following the i8 test above. Covers the u8 output path
// enabled for a quantized KV cache (GroupQueryAttention int4-as-u8-packed dynamic Slice+Concat).
TEST(slice_gpu_u8, bfyx) {
    auto& engine = get_test_engine();

    // input [1,1,2,4] u8, slice axis 2 -> [1,1,1,4] (second row).
    auto input = engine.allocate_memory({ov::PartialShape{1, 1, 2, 4}, data_types::u8, format::bfyx});
    auto start = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto stop = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto step = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto axes = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});

    set_values<uint8_t>(input, {1, 2, 3, 4, 250, 251, 252, 253});
    set_values<int64_t>(start, {1});
    set_values<int64_t>(stop, {2});
    set_values<int64_t>(step, {1});
    set_values<int64_t>(axes, {2});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("start", start));
    topology.add(data("stop", stop));
    topology.add(data("step", step));
    topology.add(data("axes", axes));
    topology.add(slice("slice", {input_info("input"), input_info("start"), input_info("stop"), input_info("step"), input_info("axes")}, 0, {1}, {1}, {2}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cldnn::network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("slice").get_memory();
    cldnn::mem_lock<uint8_t, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<uint8_t> expected_results = {250, 251, 252, 253};
    ASSERT_EQ(output_ptr.size(), expected_results.size());
    for (size_t i = 0; i < expected_results.size(); ++i)
        ASSERT_EQ(expected_results[i], output_ptr[i]) << "i=" << i;
}

// f8e4m3 Slice: values are exactly representable, so the sliced (byte-moved) result is exact. Covers
// the f8 output path enabled for a quantized KV cache (GroupQueryAttention f8e4m3 dynamic Slice+Concat).
TEST(slice_gpu_f8e4m3, bfyx) {
    auto& engine = get_test_engine();

    // input [1,1,2,4] f8e4m3, slice axis 2 -> [1,1,1,4] (second row).
    auto input = engine.allocate_memory({ov::PartialShape{1, 1, 2, 4}, data_types::f8e4m3, format::bfyx});
    auto start = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto stop = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto step = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto axes = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});

    set_values<ov::float8_e4m3>(input, {1.0f, 2.0f, 3.0f, 4.0f, -0.5f, -1.5f, -2.5f, -0.25f});
    set_values<int64_t>(start, {1});
    set_values<int64_t>(stop, {2});
    set_values<int64_t>(step, {1});
    set_values<int64_t>(axes, {2});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("start", start));
    topology.add(data("stop", stop));
    topology.add(data("step", step));
    topology.add(data("axes", axes));
    topology.add(slice("slice", {input_info("input"), input_info("start"), input_info("stop"), input_info("step"), input_info("axes")}, 0, {1}, {1}, {2}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cldnn::network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("slice").get_memory();
    cldnn::mem_lock<ov::float8_e4m3, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<ov::float8_e4m3> expected_results = {-0.5f, -1.5f, -2.5f, -0.25f};
    ASSERT_EQ(output_ptr.size(), expected_results.size());
    for (size_t i = 0; i < expected_results.size(); ++i)
        ASSERT_EQ(expected_results[i], output_ptr[i]) << "i=" << i;
}

// Standalone test for bf16, as values in typed tests are too big for bf16
TEST(slice_gpu_bf16, bfyx_positive_step) {
    auto& engine = get_test_engine();

    // Input shape {1, 1, 4, 8} -> 32 elements holding values 0..31.
    const ov::PartialShape input_shape{ 1, 1, 4, 8 };
    const size_t input_size = ov::shape_size(input_shape.get_shape());
    std::vector<ov::bfloat16> input_data(input_size);
    for (size_t i = 0; i < input_size; ++i)
        input_data[i] = static_cast<ov::bfloat16>(static_cast<float>(i));

    auto input = engine.allocate_memory({ input_shape, data_types::bf16, format::bfyx });
    set_values(input, input_data);

    // Slice y in [1, 3) step 1 and x in [2, 7) step 2 (axes 2 and 3).
    auto start = engine.allocate_memory({ ov::PartialShape{ 2 }, data_types::i64, format::bfyx });
    set_values<int64_t>(start, { 1, 2 });
    auto stop = engine.allocate_memory({ ov::PartialShape{ 2 }, data_types::i64, format::bfyx });
    set_values<int64_t>(stop, { 3, 7 });
    auto step = engine.allocate_memory({ ov::PartialShape{ 2 }, data_types::i64, format::bfyx });
    set_values<int64_t>(step, { 1, 2 });
    auto axes = engine.allocate_memory({ ov::PartialShape{ 2 }, data_types::i64, format::bfyx });
    set_values<int64_t>(axes, { 2, 3 });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("start", start));
    topology.add(data("stop", stop));
    topology.add(data("step", step));
    topology.add(data("axes", axes));
    topology.add(
        slice("slice", {input_info("input"), input_info("start"), input_info("stop"), input_info("step"), input_info("axes")}, 0, {1, 2}, {1, 2}, {2, 3}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    network->set_input_data("input", input);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "slice");

    auto output = outputs.at("slice").get_memory();
    cldnn::mem_lock<ov::bfloat16, mem_lock_type::read> output_ptr(output, get_test_stream());

    // value(y, x) = y * 8 + x for y in {1, 2}, x in {2, 4, 6}.
    const std::vector<float> expected = { 10, 12, 14, 18, 20, 22 };
    ASSERT_EQ(output_ptr.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        ASSERT_TRUE(are_equal(expected[i], static_cast<float>(output_ptr[i]), 2e-3));
}

}  // anonymous namespace

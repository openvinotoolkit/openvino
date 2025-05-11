// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/slice.hpp>
#include <intel_gpu/primitives/data.hpp>

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
        topology.add(slice("slice", inputs));

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

        cldnn::mem_lock<T> output_ptr(output, get_test_stream());
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

} // anonymous namespace

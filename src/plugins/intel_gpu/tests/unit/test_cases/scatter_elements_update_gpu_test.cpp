// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "openvino/reference/scatter_elements_update.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scatter_elements_update.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;


template <typename T>
void test_d2411_axisF(bool is_caching_test) {
    //  Dictionary : 2x4x1x1
    //  Indexes : 2x2x1x1
    //  Updates : 2x2x1x1
    //  Axis : 1
    //  Output : 2x4x1x1
    //  Input values in fp16
    //
    //  Input:
    //  3.f, 6.f, 5.f, 4.f,
    //  1.f, 7.f, 2.f, 9.f
    //
    //  Indexes:
    //  0.f, 1.f
    //  2.f, 3.f
    //
    //  Updates:
    //  10.f, 11.f,
    //  12.f, 13.f
    //
    //  Output:
    //  10.f, 11.f, 5.f, 4.f,
    //  1.f, 7.f, 12.f, 13.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 2, 4, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Updates
    auto axis = 1;

    set_values(input1, {
        ov::float16(3.0f), ov::float16(6.0f), ov::float16(5.0f), ov::float16(4.0f),
        ov::float16(1.0f), ov::float16(7.0f), ov::float16(2.0f), ov::float16(9.0f)
    });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f),
        ov::float16(2.0f), ov::float16(3.0f)
    });

    set_values(input3, {
        ov::float16(10.0f), ov::float16(11.0f),
        ov::float16(12.0f), ov::float16(13.0f)
    });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_elements_update("scatter_elements_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), axis)
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("InputData", input1);
    network->set_input_data("InputIndices", input2);
    network->set_input_data("InputUpdates", input3);

    auto outputs = network->execute();

    auto output = outputs.at("scatter_elements_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<T> expected_results = {
        10.f, 11.f, 5.f, 4.f,
        1.f, 7.f, 12.f, 13.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_elements_update_gpu_fp16, d2411_axisF) {
    test_d2411_axisF<float>(false);
}

namespace {
template<typename T, typename T_IND>
struct ScatterElementsUpdateParams {
    int64_t axis;
    tensor data_tensor;
    std::vector<T> data;
    tensor indices_tensor;
    std::vector<T_IND> indices;
    std::vector<T> updates;
    std::vector<T> expected;
};

template<typename T, typename T_IND>
using ScatterElementsUpdateParamsWithFormat = std::tuple<
    ScatterElementsUpdateParams<T, T_IND>,
    format::type,     // source (plain) layout
    format::type,     // target (blocked) data layout
    format::type,     // target (blocked) indices layout
    format::type      // target (blocked) updates layout
>;

template<typename T, typename T_IND>
using ScatterElementsUpdateReduceParamsWithFormat = std::tuple<
    ScatterElementsUpdateParams<T, T_IND>,
    ScatterElementsUpdateOp::Reduction,
    bool,             // use_init_value
    format::type
>;

const std::vector<format::type> formats2D{
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
};

const std::vector<format::type> formats3D{
        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::bs_fs_zyx_bsv16_fsv16
};

const std::vector<format::type> formats4D{
        format::bfwzyx
};

template<typename T>
std::vector<T> getValues(const std::vector<float> &values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template<typename T, typename T_IND>
std::vector<ScatterElementsUpdateParams<T, T_IND>> generateScatterElementsUpdateParams2D() {
    const std::vector<ScatterElementsUpdateParams<T, T_IND> > result = {
        {   1,
            tensor{2, 4, 1, 1},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7 }),
            tensor{2, 2, 1, 1},
            getValues<T_IND>({ 0, 1, 2, 3 }),
            getValues<T>({ -10, -11, -12, -13 }),
            getValues<T>({ -10, -11, 2, 3, 4, 5, -12, -13 })
        },
        {   2,
            tensor{2, 1, 2, 2},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7 }),
            tensor{2, 1, 2, 1},
            getValues<T_IND>({ 0, 1, 0, 1 }),
            getValues<T>({ -10, -11, -12, -13 }),
            getValues<T>({ -10, 1, 2, -11, -12, 5, 6, -13 })
        },
        {   3,
            tensor{2, 1, 2, 2},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7 }),
            tensor{2, 1, 1, 2},
            getValues<T_IND>({ 0, 1, 0, 1 }),
            getValues<T>({ -10, -11, -12, -13 }),
            getValues<T>({ -10, 1, 2, -11, -12, 5, 6, -13 })
        },
    };

    return result;
}


template<typename T, typename T_IND>
std::vector<ScatterElementsUpdateParams<T, T_IND>> generateScatterElementsUpdateParams3D() {
    const std::vector<ScatterElementsUpdateParams<T, T_IND>> result = {
        {   1,
            tensor{2, 4, 1, 1, 3},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }),
            tensor{2, 1, 1, 1, 2},
            getValues<T_IND>({ 0, 3, 1, 2 }),
            getValues<T>({ -100, -110, -120, -130 }),
            getValues<T>({ -100, 1, 2, 3, 4, 5, 6, 7, 8, 9, -110, 11, 12, 13, 14, -120, 16, 17, 18, -130, 20, 21, 22, 23 })
        },
        {   4,
            tensor{2, 4, 1, 1, 3},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }),
            tensor{2, 1, 1, 1, 2},
            getValues<T_IND>({ 0, 1, 0, 1 }),
            getValues<T>({ -100, -110, -120, -130 }),
            getValues<T>({ -100, 1, -110, 3, 4, 5, 6, 7, 8, 9, 10, 11, -120, 13, -130, 15, 16, 17, 18, 19, 20, 21, 22, 23 })
        },
    };

    return result;
}

template<typename T, typename T_IND>
std::vector<ScatterElementsUpdateParams<T, T_IND>> generateScatterElementsUpdateParams4D() {
    const std::vector<ScatterElementsUpdateParams<T, T_IND>> result = {
        {   5,
            tensor{2, 4, 2, 1, 1, 3},
            getValues<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                          24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}),
            tensor{2, 1, 1, 1, 1, 2},
            getValues<T_IND>({2, 1, 1, 1, 2}),
            getValues<T>({-100, -110, -120, -130}),
            getValues<T>({0, 1, -100, -110, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                          24, -120, 26, -130, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}),
        }
    };

    return result;
}

template<typename T>
float getError() {
    return 0.0;
}

template<>
float getError<float>() {
    return 0.001;
}

template<>
float getError<ov::float16>() {
    return 0.2;
}

struct PrintToStringParamName {
    template<typename T, typename T_IND>
    std::string operator()(const testing::TestParamInfo<ScatterElementsUpdateParamsWithFormat<T, T_IND> > &param) {
        std::stringstream buf;
        ScatterElementsUpdateParams<T, T_IND> p;
        format::type plain_format;
        format::type target_data_format;
        format::type target_indices_format;
        format::type target_updates_format;
        std::tie(p, plain_format, target_data_format, target_indices_format, target_updates_format) = param.param;
        buf << "_axis=" << p.axis
            << "_data=" << p.data_tensor.to_string()
            << "_indices=" << p.indices_tensor.to_string()
            << "_plainFormat=" << fmt_to_str(plain_format)
            << "_targetDataFormat=" << fmt_to_str(target_data_format)
            << "_targetIndicesFormat=" << fmt_to_str(target_indices_format)
            << "_targetUpdatesFormat=" << fmt_to_str(target_updates_format);
        return buf.str();
    }

    template<typename T, typename T_IND>
    std::string operator()(const testing::TestParamInfo<ScatterElementsUpdateReduceParamsWithFormat<T, T_IND> > &param) {
        std::stringstream buf;
        ScatterElementsUpdateParams<T, T_IND> p;
        ScatterElementsUpdateOp::Reduction mode;
        bool use_init_value;
        format::type plain_format;
        std::tie(p, mode, use_init_value, plain_format) = param.param;
        using ov::op::operator<<;
        buf << "_axis=" << p.axis
            << "_mode=" << mode
            << "_use_init_value=" << use_init_value
            << "_data=" << p.data_tensor.to_string()
            << "_indices=" << p.indices_tensor.to_string()
            << "_plainFormat=" << fmt_to_str(plain_format);
        return buf.str();
    }
};

template<typename T, typename T_IND>
struct scatter_elements_update_gpu_formats_test
        : public ::testing::TestWithParam<ScatterElementsUpdateParamsWithFormat<T, T_IND> > {
public:
    void test(bool is_caching_test) {
        const auto data_type = ov::element::from<T>();
        const auto indices_type = ov::element::from<T_IND>();
        ScatterElementsUpdateParams<T, T_IND> params;
        format::type plain_format;
        format::type target_data_format;
        format::type target_indices_format;
        format::type target_updates_format;

        std::tie(params, plain_format, target_data_format, target_indices_format, target_updates_format) = this->GetParam();
        if (target_indices_format == format::any) {
            target_indices_format = target_data_format;
        }
        if (target_updates_format == format::any) {
            target_updates_format = target_data_format;
        }

        auto& engine = get_test_engine();
        const auto data = engine.allocate_memory({data_type, plain_format, params.data_tensor});
        const auto indices = engine.allocate_memory({indices_type, plain_format, params.indices_tensor});
        const auto updates = engine.allocate_memory({data_type, plain_format, params.indices_tensor});

        set_values(data, params.data);
        set_values(indices, params.indices);
        set_values(updates, params.updates);

        topology topology;
        topology.add(input_layout("Data", data->get_layout()));
        topology.add(input_layout("Indices", indices->get_layout()));
        topology.add(input_layout("Updates", updates->get_layout()));
        topology.add(reorder("DataReordered", input_info("Data"), target_data_format, data_type));
        topology.add(reorder("IndicesReordered", input_info("Indices"), target_indices_format, indices_type));
        topology.add(reorder("UpdatesReordered", input_info("Updates"), target_updates_format, data_type));
        topology.add(
            scatter_elements_update("ScatterEelementsUpdate", input_info("DataReordered"), input_info("IndicesReordered"),
                                    input_info("UpdatesReordered"), params.axis)
        );
        topology.add(reorder("ScatterEelementsUpdatePlain", input_info("ScatterEelementsUpdate"), plain_format, data_type));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Data", data);
        network->set_input_data("Indices", indices);
        network->set_input_data("Updates", updates);

        const auto outputs = network->execute();
        const auto output = outputs.at("ScatterEelementsUpdatePlain").get_memory();
        const cldnn::mem_lock<T> output_ptr(output, get_test_stream());

        ASSERT_EQ(params.data.size(), output_ptr.size());
        ASSERT_EQ(params.expected.size(), output_ptr.size());
        for (uint32_t i = 0; i < output_ptr.size(); i++) {
            ASSERT_NEAR(output_ptr[i], params.expected[i], getError<T>())
                                << "format=" << fmt_to_str(target_data_format) << ", i=" << i;
        }
    }
};

using scatter_elements_update_gpu_formats_test_f32 = scatter_elements_update_gpu_formats_test<float, int32_t>;
using scatter_elements_update_gpu_formats_test_f16 = scatter_elements_update_gpu_formats_test<ov::float16, int32_t>;
using scatter_elements_update_gpu_formats_test_i32 = scatter_elements_update_gpu_formats_test<int32_t, int32_t>;

template<typename T, typename T_IND>
struct scatter_elements_update_gpu_reduction_test
        : public ::testing::TestWithParam<ScatterElementsUpdateReduceParamsWithFormat<T, T_IND> > {
public:
    void test(bool is_caching_test) {
        const auto data_type = ov::element::from<T>();
        const auto indices_type = ov::element::from<T_IND>();
        ScatterElementsUpdateParams<T, T_IND> params;
        format::type plain_format;
        ScatterElementsUpdateOp::Reduction mode;
        bool use_init_value;
        std::tie(params, mode, use_init_value, plain_format) = this->GetParam();
        params.expected = generateReferenceOutput(plain_format, params, mode, use_init_value);

        auto& engine = get_test_engine();
        const auto data = engine.allocate_memory({data_type, plain_format, params.data_tensor});
        const auto indices = engine.allocate_memory({indices_type, plain_format, params.indices_tensor});
        const auto updates = engine.allocate_memory({data_type, plain_format, params.indices_tensor});

        set_values(data, params.data);
        set_values(indices, params.indices);
        set_values(updates, params.updates);

        topology topology;
        topology.add(input_layout("Data", data->get_layout()));
        topology.add(input_layout("Indices", indices->get_layout()));
        topology.add(input_layout("Updates", updates->get_layout()));
        topology.add(
            scatter_elements_update("ScatterElementsUpdate",
                                    input_info("Data"),
                                    input_info("Indices"),
                                    input_info("Updates"),
                                    params.axis,
                                    mode,
                                    use_init_value)
        );
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Data", data);
        network->set_input_data("Indices", indices);
        network->set_input_data("Updates", updates);

        const auto outputs = network->execute();
        const auto output = outputs.at("ScatterElementsUpdate").get_memory();
        const cldnn::mem_lock<T> output_ptr(output, get_test_stream());

        ASSERT_EQ(params.data.size(), output_ptr.size());
        ASSERT_EQ(params.expected.size(), output_ptr.size());
        for (uint32_t i = 0; i < output_ptr.size(); i++) {
            ASSERT_NEAR(output_ptr[i], params.expected[i], getError<T>()) << ", i=" << i;
        }
    }
private:
    static ov::Shape tensorToShape(const tensor &t, const format f) {
        std::vector<int> vec(cldnn::format::dimension(f));
        for (size_t i = 0; i < vec.size(); ++i) {
            vec[i] = t.sizes()[i];
        }
        std::reverse(vec.begin() + 2, vec.end());

        return ov::Shape(vec.begin(), vec.end());
    }

    static std::vector<T> generateReferenceOutput(const format fmt,
                                                  const ScatterElementsUpdateParams<T, T_IND>& p,
                                                  const ScatterElementsUpdateOp::Reduction mode,
                                                  const bool use_init_value) {
        std::vector<T> out(p.data_tensor.count());
        const auto data_shape = tensorToShape(p.data_tensor, fmt);
        const auto indices_shape = tensorToShape(p.indices_tensor, fmt);

        ov::reference::scatter_elem_update<T, T_IND>(p.data.data(),
                                                     p.indices.data(),
                                                     p.updates.data(),
                                                     p.axis,
                                                     out.data(),
                                                     data_shape,
                                                     indices_shape,
                                                     mode,
                                                     use_init_value);
        return out;
    }
};
}; // namespace

TEST_P(scatter_elements_update_gpu_formats_test_f32, basic) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

TEST_P(scatter_elements_update_gpu_formats_test_f16, basic) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

TEST_P(scatter_elements_update_gpu_formats_test_i32, basic) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}


INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_f32_2d,
                         scatter_elements_update_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<float, int32_t>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D),
                                 ::testing::Values(format::any),
                                 ::testing::Values(format::any)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_f16_2d,
                         scatter_elements_update_gpu_formats_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<ov::float16, int32_t>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D),
                                 ::testing::Values(format::any),
                                 ::testing::Values(format::any)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_i32_2d,
                         scatter_elements_update_gpu_formats_test_i32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<int32_t, int32_t>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D),
                                 ::testing::Values(format::any),
                                 ::testing::Values(format::any)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_f32_3d,
                         scatter_elements_update_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams3D<float, int32_t>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(formats3D),
                                 ::testing::Values(format::any),
                                 ::testing::Values(format::any)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_f32_4d,
                         scatter_elements_update_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams4D<float, int32_t>()),
                                 ::testing::Values(format::bfwzyx),
                                 ::testing::ValuesIn(formats4D),
                                 ::testing::ValuesIn(formats4D),
                                 ::testing::ValuesIn(formats4D)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_mixed_inputs,
                         scatter_elements_update_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<float, int32_t>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn({format::b_fs_yx_fsv16, format::b_fs_yx_fsv32}),
                                 ::testing::ValuesIn({format::bs_fs_yx_bsv16_fsv16, format::bs_fs_yx_bsv32_fsv16}),
                                 ::testing::ValuesIn({format::bs_fs_yx_bsv32_fsv32, format::bfyx})
                         ),
                         PrintToStringParamName());

using scatter_elements_update_gpu_reduction_test_f32 = scatter_elements_update_gpu_reduction_test<float, int32_t>;
using scatter_elements_update_gpu_reduction_test_i32 = scatter_elements_update_gpu_reduction_test<int32_t, int32_t>;

TEST_P(scatter_elements_update_gpu_reduction_test_f32, basic) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

TEST_P(scatter_elements_update_gpu_reduction_test_i32, basic) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

const std::vector<ov::op::v12::ScatterElementsUpdate::Reduction> reduce_modes{
    ov::op::v12::ScatterElementsUpdate::Reduction::SUM,
    ov::op::v12::ScatterElementsUpdate::Reduction::PROD,
    ov::op::v12::ScatterElementsUpdate::Reduction::MIN,
    // MAX mode omitted intentionally - see dedicated MAX tests below 
    ov::op::v12::ScatterElementsUpdate::Reduction::MEAN
};


INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_reduction_test_f32_2d,
                         scatter_elements_update_gpu_reduction_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<float, int32_t>()),
                                 ::testing::ValuesIn(reduce_modes),
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::Values(format::bfyx)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_reduction_test_i32_2d,
                         scatter_elements_update_gpu_reduction_test_i32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<int32_t, int32_t>()),
                                 ::testing::ValuesIn(reduce_modes),
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::Values(format::bfyx)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_reduction_test_f32_3d,
                         scatter_elements_update_gpu_reduction_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams3D<float, int32_t>()),
                                 ::testing::ValuesIn(reduce_modes),
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::Values(format::bfzyx)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_reduction_test_f32_4d,
                         scatter_elements_update_gpu_reduction_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams4D<float, int32_t>()),
                                 ::testing::ValuesIn(reduce_modes),
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::Values(format::bfwzyx)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_reduction_none_test_f32_2d,
                         scatter_elements_update_gpu_reduction_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<float, int32_t>()),
                                 ::testing::Values(ov::op::v12::ScatterElementsUpdate::Reduction::NONE),
                                 ::testing::Values(true),
                                 ::testing::Values(format::bfyx)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_max_reduction_use_init,
                         scatter_elements_update_gpu_reduction_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<float, int32_t>()),
                                 ::testing::Values(ov::op::v12::ScatterElementsUpdate::Reduction::MAX),
                                 ::testing::Values(true),
                                 ::testing::Values(format::bfyx)
                         ),
                         PrintToStringParamName());

// Disabled due to bug in reference implementation - see function reduction_neutral_value()
// in core/reference/include/openvino/reference/scatter_elements_update.hpp.
// For MAX reduction it returns numeric_limits<T>::min() which is minimal *positive*, not truly minimal value of type T,
// which causes wrong result on negative input/update values.
// Enable when/if reference implementation is fixed.
INSTANTIATE_TEST_SUITE_P(DISABLED_scatter_elements_update_gpu_max_reduction_dont_use_init,
                         scatter_elements_update_gpu_reduction_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<float, int32_t>()),
                                 ::testing::Values(ov::op::v12::ScatterElementsUpdate::Reduction::MAX),
                                 ::testing::Values(false),
                                 ::testing::Values(format::bfyx)
                         ),
                         PrintToStringParamName());

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(scatter_elements_update_gpu_formats_test_f32, basic_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(scatter_elements_update_gpu_formats_test_f16, basic_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(scatter_elements_update_gpu_formats_test_i32, basic_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}
#endif
TEST(scatter_elements_update_gpu_fp16, d2411_axisF_cached) {
    test_d2411_axisF<float>(true);
}

TEST_P(scatter_elements_update_gpu_reduction_test_f32, cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}


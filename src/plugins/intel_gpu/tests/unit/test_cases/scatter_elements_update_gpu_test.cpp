// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstddef>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scatter_elements_update.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <random>

#include "openvino/reference/scatter_elements_update.hpp"
#include "test_utils.h"

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
    cldnn::mem_lock<uint16_t, mem_lock_type::read> output_ptr(output, get_test_stream());

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
        },
        {   2,
            tensor{2, 1, 2, 2},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7 }),
            tensor{2, 1, 2, 1},
            getValues<T_IND>({ 0, 1, 0, 1 }),
            getValues<T>({ -10, -11, -12, -13 }),
        },
        {   3,
            tensor{2, 1, 2, 2},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7 }),
            tensor{2, 1, 1, 2},
            getValues<T_IND>({ 0, 1, 0, 1 }),
            getValues<T>({ -10, -11, -12, -13 }),
        },
        {
            1,
            tensor{1, 4, 1, 1},
            getValues<T>({10, 20, 30, 40}),
            tensor{1, 2, 1, 1},
            getValues<T_IND>({2, 3}),
            getValues<T>({99, 88}),
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
        },
        {   4,
            tensor{2, 4, 1, 1, 3},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }),
            tensor{2, 1, 1, 1, 2},
            getValues<T_IND>({ 0, 1, 0, 1 }),
            getValues<T>({ -100, -110, -120, -130 }),
        },
        {
            0,
            tensor{2, 2, 2, 1, 1},
            getValues<T>({1, 2, 3, 4, 5, 6, 7, 8}),
            tensor{2, 1, 2, 1, 1},
            getValues<T_IND>({1, 0, 0, 1}),
            getValues<T>({9, 8, 7, 6}),
        }
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
            getValues<T_IND>({2, 1, 1, 1}),
            getValues<T>({-100, -110, -120, -130}),
        },
        {
            5,
            tensor{1, 2, 2, 2, 1, 1},
            getValues<T>({1, 2, 3, 4, 5, 6, 7, 8}),
            tensor{1, 2, 1, 2, 1, 1},
            getValues<T_IND>({1, 0, 0, 1}),
            getValues<T>({10, 11, 12, 13}),
        },
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

        const auto& [p, plain_format, target_data_format, target_indices_format, target_updates_format] = param.param;
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

        const auto& [p, mode, use_init_value, plain_format] = param.param;
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

        const auto& [params, plain_format, target_data_format, _target_indices_format, _target_updates_format] = this->GetParam();
        auto target_indices_format = _target_indices_format;
        auto target_updates_format = _target_updates_format;
        auto expected = generateReferenceOutput(plain_format, params, ov::op::v12::ScatterElementsUpdate::Reduction::NONE, false);
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
        const cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());

        ASSERT_EQ(params.data.size(), output_ptr.size());
        ASSERT_EQ(expected.size(), output_ptr.size());
        for (uint32_t i = 0; i < output_ptr.size(); i++) {
            ASSERT_NEAR(output_ptr[i], expected[i], getError<T>())
                                << "format=" << fmt_to_str(target_data_format) << ", i=" << i;
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

        const auto& [params, mode, use_init_value, plain_format] = this->GetParam();
        auto expected = generateReferenceOutput(plain_format, params, mode, use_init_value);

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
        const cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());

        ASSERT_EQ(params.data.size(), output_ptr.size());
        ASSERT_EQ(expected.size(), output_ptr.size());
        for (uint32_t i = 0; i < output_ptr.size(); i++) {
            ASSERT_NEAR(output_ptr[i], expected[i], getError<T>()) << ", i=" << i;
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

using scatter_elements_update_gpu_reduction_test_f16 = scatter_elements_update_gpu_reduction_test<ov::float16, int32_t>;
using scatter_elements_update_gpu_reduction_test_f32 = scatter_elements_update_gpu_reduction_test<float, int32_t>;
using scatter_elements_update_gpu_reduction_test_i32 = scatter_elements_update_gpu_reduction_test<int32_t, int32_t>;

TEST_P(scatter_elements_update_gpu_reduction_test_f16, basic) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

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

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_reduction_test_f16_2d,
                         scatter_elements_update_gpu_reduction_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<ov::float16, int32_t>()),
                                 ::testing::ValuesIn(reduce_modes),
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::Values(format::bfyx)
                         ),
                         PrintToStringParamName());

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

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_reduction_test_f16_3d,
                         scatter_elements_update_gpu_reduction_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams3D<ov::float16, int32_t>()),
                                 ::testing::ValuesIn(reduce_modes),
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::Values(format::bfzyx)
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

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_reduction_test_f16_4d,
                         scatter_elements_update_gpu_reduction_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams4D<ov::float16, int32_t>()),
                                 ::testing::ValuesIn(reduce_modes),
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::Values(format::bfwzyx)
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

TEST(scatter_elements_update_gpu_fp32, smoke_multiple_indices_mean_big_1d_dynamic) {
    auto& engine = get_test_engine();
    int num = 100;
    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{num, 1, 1, 1 } }); // input
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{num, 1, 1, 1 } });  // indices
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{num, 1, 1, 1 } });  // updates

    std::vector<float> data(num, 0);
    std::vector<int32_t> indices(num, 0);
    std::vector<float> updates(num, 0);
    updates[0] = num;
    int32_t axis = 0;
    ScatterElementsUpdateOp::Reduction mode = ov::op::v12::ScatterElementsUpdate::Reduction::MEAN;
    bool use_init_value = false;

    set_values(input1, data);
    set_values(input2, indices);
    set_values(input3, updates);

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("indices", { ov::PartialShape{ ov::Dimension(-1) }, data_types::i32, format::bfyx }));
    topology.add(input_layout("updates", { ov::PartialShape{ ov::Dimension(-1) }, data_types::f32, format::bfyx }));
    topology.add(
        scatter_elements_update(
            "scatter_elements_update",
            input_info("input"),
            input_info("indices"),
            input_info("updates"),
            axis,
            mode,
            use_init_value));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("indices", input2);
    network.set_input_data("updates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_elements_update").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<float> expected_results(num, 0);
    expected_results.front() = 1;
    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_elements_update_gpu_fp32, smoke_multiple_indices_sum_big_1d_dynamic) {
    auto& engine = get_test_engine();
    int32_t num = 10000;
    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{num, 1, 1, 1 } }); // input
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{num, 1, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{num, 1, 1, 1 } }); // updates

    std::vector<float> data(num, 0);
    std::vector<int32_t> indices(num, 0);
    std::vector<float> updates(num, 0);

    const std::vector<int32_t> target_update_positions = { 0, 100, 200, 1000, 5000, 6000, 6001 };
    for (auto pos : target_update_positions) {
        updates[pos] = num;
        indices[pos] = pos;
    }

    int32_t axis = 0;
    ScatterElementsUpdateOp::Reduction mode = ov::op::v12::ScatterElementsUpdate::Reduction::SUM;
    bool use_init_value = true;

    set_values(input1, data);
    set_values(input2, indices);
    set_values(input3, updates);

    topology topology;
    topology.add(input_layout("input", { ov::PartialShape{ ov::Dimension(-1) }, data_types::f32, format::bfyx }));
    topology.add(input_layout("indices", { ov::PartialShape{ ov::Dimension(-1) }, data_types::i32, format::bfyx }));
    topology.add(input_layout("updates", { ov::PartialShape{ ov::Dimension(-1) }, data_types::f32, format::bfyx }));
    topology.add(
        scatter_elements_update(
            "scatter_elements_update",
            input_info("input"),
            input_info("indices"),
            input_info("updates"),
            axis,
            mode,
            use_init_value));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("indices", input2);
    network.set_input_data("updates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_elements_update").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<float> expected_results(num, 0);
    for (auto pos : target_update_positions) {
        expected_results[pos] = num;
    }

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_elements_update_gpu_fp32, smoke_sum_large_values_overflow_guard) {
    // Verifies that f32 SUM with update values >> 32768 (= INT32_MAX / FP_SCALE)
    // does not overflow the fixed-point accumulator.
    // Before the CAS fix: val * 65536 > INT32_MAX → convert_int_rte() UB → result ~0.
    auto& engine = get_test_engine();
    const int32_t num = 1;
    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{num, 1, 1, 1} });
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{3,  1, 1, 1} });
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{3,  1, 1, 1} });

    // Each update value is 1e7 (>> 32768), all scattered to index 0.
    // Expected: output[0] = 0.0 (init) + 1e7 + 1e7 + 1e7 = 3e7
    set_values(input1, std::vector<float>{0.0f});
    set_values(input2, std::vector<int32_t>{0, 0, 0});
    set_values(input3, std::vector<float>{1e7f, 1e7f, 1e7f});

    topology topology;
    topology.add(input_layout("input",   input1->get_layout()));
    topology.add(input_layout("indices", input2->get_layout()));
    topology.add(input_layout("updates", input3->get_layout()));
    topology.add(scatter_elements_update("scatter_elements_update",
                                         input_info("input"),
                                         input_info("indices"),
                                         input_info("updates"),
                                         0,
                                         ScatterElementsUpdateOp::Reduction::SUM,
                                         true));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input",   input1);
    network.set_input_data("indices", input2);
    network.set_input_data("updates", input3);
    auto outputs = network.execute();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(
        outputs.at("scatter_elements_update").get_memory(), get_test_stream());

    ASSERT_NEAR(output_ptr[0], 3e7f, 1.0f);  // must not be ~0 or inf
}

TEST(scatter_elements_update_gpu_fp32, smoke_sum_large_values_overflow_guard_dynamic) {
    // Same overflow guard as above but with dynamic (shape-agnostic) shapes,
    // which dispatch a different kernel path than static shapes.
    auto& engine = get_test_engine();
    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{1, 1, 1, 1} });
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{3, 1, 1, 1} });
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{3, 1, 1, 1} });

    set_values(input1, std::vector<float>{0.0f});
    set_values(input2, std::vector<int32_t>{0, 0, 0});
    set_values(input3, std::vector<float>{1e7f, 1e7f, 1e7f});

    topology topology;
    topology.add(input_layout("input",   { ov::PartialShape{ ov::Dimension(-1) }, data_types::f32, format::bfyx }));
    topology.add(input_layout("indices", { ov::PartialShape{ ov::Dimension(-1) }, data_types::i32, format::bfyx }));
    topology.add(input_layout("updates", { ov::PartialShape{ ov::Dimension(-1) }, data_types::f32, format::bfyx }));
    topology.add(scatter_elements_update("scatter_elements_update",
                                         input_info("input"),
                                         input_info("indices"),
                                         input_info("updates"),
                                         0,
                                         ScatterElementsUpdateOp::Reduction::SUM,
                                         true));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input",   input1);
    network.set_input_data("indices", input2);
    network.set_input_data("updates", input3);
    auto outputs = network.execute();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(
        outputs.at("scatter_elements_update").get_memory(), get_test_stream());

    ASSERT_NEAR(output_ptr[0], 3e7f, 1.0f);
}

TEST(scatter_elements_update_gpu_fp32, smoke_sum_opt_local_kernel_static_1d) {
    // Static-shape counterpart to smoke_multiple_indices_sum_big_1d_dynamic, sized to
    // force scatter_elements_update_opt_local_sum's eligibility gate (static shape,
    // SUM mode, output large enough that it can't fit in `_ref`'s own local-memory
    // path -- 20000 f32 elements is ~80KB, well over the ~64KB typical local-memory
    // budget). Mostly-zero indices/updates concentrate almost all of the 20000
    // dispatched updates on a single destination (stresses the local-window path),
    // while a handful of real updates are scattered far enough apart that at least
    // one pair is guaranteed to fall in different windows (stresses the
    // global-fallback path too). Correctness is checked exactly, not approximately --
    // the fixed-point SUM accumulator makes the true result an exact integer here.
    auto& engine = get_test_engine();
    const int32_t num = 20000;
    auto input1 = engine.allocate_memory({data_types::f32, format::bfyx, tensor{num, 1, 1, 1}});
    auto input2 = engine.allocate_memory({data_types::i32, format::bfyx, tensor{num, 1, 1, 1}});
    auto input3 = engine.allocate_memory({data_types::f32, format::bfyx, tensor{num, 1, 1, 1}});

    std::vector<float> data(num, 0.0f);
    std::vector<int32_t> indices(num, 0);
    std::vector<float> updates(num, 0.0f);

    const std::vector<int32_t> target_update_positions = {0, 100, 4095, 4096, 8191, 15000, 19999};
    for (auto pos : target_update_positions) {
        updates[pos] = 1.0f;
        indices[pos] = pos;
    }

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("indices", input2->get_layout()));
    topology.add(input_layout("updates", input3->get_layout()));
    topology.add(scatter_elements_update("scatter_elements_update",
                                         input_info("input"),
                                         input_info("indices"),
                                         input_info("updates"),
                                         0,
                                         ScatterElementsUpdateOp::Reduction::SUM,
                                         true));

    set_values(input1, data);
    set_values(input2, indices);
    set_values(input3, updates);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input1);
    network.set_input_data("indices", input2);
    network.set_input_data("updates", input3);

    auto outputs = network.execute();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(outputs.at("scatter_elements_update").get_memory(),
                                                           get_test_stream());

    // Every dispatched index defaults to 0 with update value 0.0 (a real, exact no-op
    // contribution), except the handful of target positions -- so index 0's expected
    // value is exactly 1.0 (only its own explicit contribution; the thousands of
    // default-index-0 contributions are all +0.0 and change nothing).
    std::vector<float> expected(num, 0.0f);
    for (auto pos : target_update_positions) {
        expected[pos] = 1.0f;
    }
    for (int32_t i = 0; i < num; ++i) {
        ASSERT_EQ(expected[i], output_ptr[i]) << "at index " << i;
    }
}

TEST(scatter_elements_update_gpu_fp16, smoke_sum_opt_local_kernel_dense_flat) {
    // Mirrors the real production shape that motivated
    // scatter_elements_update_opt_local_sum precisely: the real DRBA/softsplat graph
    // flattens its spatial dims to one axis *before* the scatter op (confirmed via
    // direct introspection of the real exported graph -- data/indices/updates are all
    // rank-3 (N,C,flat_spatial), not genuinely 2D), f16, SUM mode, static shape, with
    // the updates/indices tensor *larger* than the output (4x here, matching a
    // 4-corner bilinear-splat forward warp concatenated into one scatter call) --
    // exactly the shape/ratio combination that exposed the real bug this kernel's
    // dispatch logic had (SetDefault used the same gws layout for the update stage as
    // for init/finalize, instead of `_ref`'s X*Y-merged layout the update-stage
    // kernel body actually expects; every contribution silently landed on the same
    // output element). The destination for each source pixel is computed with a
    // bounded, moderate 2D neighborhood jitter (matching realistic optical-flow
    // motion magnitude) even though the tensors themselves are flat -- so most
    // updates should resolve via the local-window path, with only edge-of-buffer
    // cases falling back to global.
    auto& engine = get_test_engine();
    const int32_t out_x = 128, out_y = 128, channels = 2;
    const int32_t out_len = out_x * out_y;
    const int32_t upd_len = out_len * 4;  // 4 bilinear corners concatenated

    auto input1 = engine.allocate_memory({data_types::f16, format::bfyx, tensor{1, channels, out_len, 1}});
    auto input2 = engine.allocate_memory({data_types::i32, format::bfyx, tensor{1, channels, upd_len, 1}});
    auto input3 = engine.allocate_memory({data_types::f16, format::bfyx, tensor{1, channels, upd_len, 1}});

    std::vector<ov::float16> data(out_len * channels, ov::float16(0.0f));
    std::vector<int32_t> indices(upd_len * channels);
    std::vector<ov::float16> updates(upd_len * channels);

    // Each of the 4 "corners" scatters every source pixel to a nearby destination
    // (bounded +/-2 pixel jitter in the flattened index's implied (x,y) grid,
    // matching realistic bilinear-splat displacement), clamped into range -- real
    // spatial locality, not synthetic worst-case chaos.
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int32_t> jitter(-2, 2);
    std::vector<int32_t> expected_indices(upd_len);
    for (int32_t corner = 0; corner < 4; ++corner) {
        for (int32_t p = 0; p < out_len; ++p) {
            int32_t x = p % out_x, y = p / out_x;
            int32_t nx = std::min(std::max(x + jitter(rng), 0), out_x - 1);
            int32_t ny = std::min(std::max(y + jitter(rng), 0), out_y - 1);
            expected_indices[corner * out_len + p] = ny * out_x + nx;
        }
    }
    for (int32_t c = 0; c < channels; ++c) {
        for (int32_t i = 0; i < upd_len; ++i) {
            indices[c * upd_len + i] = expected_indices[i];
            updates[c * upd_len + i] = ov::float16(1.0f);
        }
    }

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("indices", input2->get_layout()));
    topology.add(input_layout("updates", input3->get_layout()));
    topology.add(scatter_elements_update("scatter_elements_update",
                                         input_info("input"),
                                         input_info("indices"),
                                         input_info("updates"),
                                         3,  // X axis -- the flat spatial scatter axis this kernel targets
                                         ScatterElementsUpdateOp::Reduction::SUM,
                                         true));

    set_values(input1, data);
    set_values(input2, indices);
    set_values(input3, updates);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input1);
    network.set_input_data("indices", input2);
    network.set_input_data("updates", input3);

    auto outputs = network.execute();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(outputs.at("scatter_elements_update").get_memory(),
                                                                 get_test_stream());

    for (int32_t c = 0; c < channels; ++c) {
        std::vector<float> expected(out_len, 0.0f);
        for (int32_t i = 0; i < upd_len; ++i) {
            expected[expected_indices[i]] += 1.0f;
        }
        for (int32_t p = 0; p < out_len; ++p) {
            float got = static_cast<float>(output_ptr[c * out_len + p]);
            // f16 fixed-point accumulation of up to ~4-8 unit contributions is exact
            // (well within f16's integer-exact range), so a tight tolerance is right
            // here, not a loose one.
            ASSERT_NEAR(expected[p], got, 0.05f) << "channel " << c << " pixel " << p;
        }
    }
}

TEST(scatter_elements_update_gpu_i32, smoke_sum_opt_local_kernel_integer_exact) {
    // Integer element types take scatter_elements_update_opt_local_sum's identity
    // encoding, so the local-staged sum is exact int32 arithmetic. Four updates collide
    // on every fourth destination, all with value 2^24+1 -- the first integer a float
    // cannot represent. An encoding that round-tripped through float would round each
    // contribution down to 2^24 and return 67108864 instead of 67108868.
    auto& engine = get_test_engine();
    const int32_t num = 20000;  // ~80KB of i32, past `_ref`'s own local-memory path
    const int32_t update_value = 16777217;  // 2^24 + 1
    auto input1 = engine.allocate_memory({data_types::i32, format::bfyx, tensor{num, 1, 1, 1}});
    auto input2 = engine.allocate_memory({data_types::i32, format::bfyx, tensor{num, 1, 1, 1}});
    auto input3 = engine.allocate_memory({data_types::i32, format::bfyx, tensor{num, 1, 1, 1}});

    std::vector<int32_t> data(num, 0);
    std::vector<int32_t> indices(num);
    std::vector<int32_t> updates(num, update_value);
    for (int32_t i = 0; i < num; ++i) {
        indices[i] = i & ~3;  // groups of 4 accumulate onto one destination
    }

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("indices", input2->get_layout()));
    topology.add(input_layout("updates", input3->get_layout()));
    topology.add(scatter_elements_update("scatter_elements_update",
                                         input_info("input"),
                                         input_info("indices"),
                                         input_info("updates"),
                                         0,
                                         ScatterElementsUpdateOp::Reduction::SUM,
                                         true));

    set_values(input1, data);
    set_values(input2, indices);
    set_values(input3, updates);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input1);
    network.set_input_data("indices", input2);
    network.set_input_data("updates", input3);

    auto outputs = network.execute();
    cldnn::mem_lock<int32_t, mem_lock_type::read> output_ptr(outputs.at("scatter_elements_update").get_memory(),
                                                             get_test_stream());

    for (int32_t i = 0; i < num; ++i) {
        const int32_t expected = (i % 4 == 0) ? 4 * update_value : 0;
        ASSERT_EQ(expected, output_ptr[i]) << "at index " << i;
    }
}

TEST(scatter_elements_update_gpu_fp32, smoke_sum_opt_local_kernel_f32_precision) {
    // f32 accumulates in floating point, so staging contributions locally reorders the
    // additions relative to `_ref` and the result is not bit-reproducible -- that is
    // float addition, not anything this kernel controls. What must hold is agreement
    // within the expected precision of the sum. Eight contributions land on every
    // destination with values in [-1, 1), and the reference is accumulated in double.
    auto& engine = get_test_engine();
    const int32_t out_len = 20000;      // ~80KB f32, past `_ref`'s own local-memory path
    const int32_t per_dest = 8;
    const int32_t n_updates = out_len * per_dest;

    auto input1 = engine.allocate_memory({data_types::f32, format::bfyx, tensor{out_len, 1, 1, 1}});
    auto input2 = engine.allocate_memory({data_types::i32, format::bfyx, tensor{n_updates, 1, 1, 1}});
    auto input3 = engine.allocate_memory({data_types::f32, format::bfyx, tensor{n_updates, 1, 1, 1}});

    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(out_len, 0.0f);
    std::vector<int32_t> indices(n_updates);
    std::vector<float> updates(n_updates);
    std::vector<double> expected(out_len, 0.0);
    for (int32_t i = 0; i < n_updates; ++i) {
        indices[i] = i % out_len;
        updates[i] = dist(rng);
        expected[indices[i]] += static_cast<double>(updates[i]);
    }

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("indices", input2->get_layout()));
    topology.add(input_layout("updates", input3->get_layout()));
    topology.add(scatter_elements_update("scatter_elements_update",
                                         input_info("input"),
                                         input_info("indices"),
                                         input_info("updates"),
                                         0,
                                         ScatterElementsUpdateOp::Reduction::SUM,
                                         true));

    set_values(input1, data);
    set_values(input2, indices);
    set_values(input3, updates);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input1);
    network.set_input_data("indices", input2);
    network.set_input_data("updates", input3);

    auto outputs = network.execute();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(outputs.at("scatter_elements_update").get_memory(),
                                                           get_test_stream());

    // Eight additions of magnitude < 1 in f32: the accumulated rounding error is bounded
    // well below 1e-4. A tolerance this tight still catches a dropped or double-counted
    // contribution, which is what could actually go wrong here.
    for (int32_t i = 0; i < out_len; ++i) {
        ASSERT_NEAR(static_cast<float>(expected[i]), output_ptr[i], 1e-4f) << "at index " << i;
    }
}

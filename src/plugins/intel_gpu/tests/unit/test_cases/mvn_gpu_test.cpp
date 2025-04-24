// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mvn.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/runtime/debug_configuration.hpp>

#include <iostream>

#include "mvn_inst.h"

using namespace cldnn;
using namespace ::tests;

class mvn_gpu_test : public ::testing::TestWithParam<cldnn::format> {};

template <typename T>
void mvn_compute_mean_across_channels(cldnn::memory::ptr output, bool normalize_variance) {
    auto l = output->get_layout();

    uint32_t batch_size = l.batch();
    uint32_t feature_size = l.feature();
    uint32_t z_size = l.spatial(2);
    uint32_t y_size = l.spatial(1);
    uint32_t x_size = l.spatial(0);

    cldnn::mem_lock<T> buff(output, get_test_stream());

    float err_margin = output->get_layout().data_type == data_types::f32 ? 1e-03F : 1e-02F;

    for (uint32_t b = 0; b < batch_size; ++b) {
        float sum = 0.f;
        float variance = 0.f;
        for (uint32_t f = 0; f < feature_size; ++f) {
            for (uint32_t z = 0; z < z_size; z++) {
                for (uint32_t y = 0; y < y_size; ++y) {
                    for (uint32_t x = 0; x < x_size; ++x) {
                        auto index_tensor = tensor(batch(b), feature(f), spatial(x, y, z, 0));
                        size_t data_index = output->get_layout().get_linear_offset(index_tensor);
                        float data = static_cast<float>(buff[data_index]);
                        sum += data;
                        if (normalize_variance)
                            variance += data * data;
                    }
                }
            }
        }
        sum /= feature_size * y_size * x_size * z_size;
        T result_sum = static_cast<T>(sum);
        ASSERT_NEAR(result_sum, 0.f, err_margin) << "at b=" << b;

        if (normalize_variance) {
            variance /= feature_size * y_size * x_size * z_size;
            T result_variance = static_cast<T>(variance);
            ASSERT_NEAR(result_variance, 1.f, err_margin) << " at b=" << b;
        }
    }
}

template <typename T>
void mvn_compute_mean_within_channels(cldnn::memory::ptr output, bool normalize_variance) {
    auto l = output->get_layout();

    uint32_t batch_size = l.batch();
    uint32_t feature_size = l.feature();
    uint32_t z_size = l.spatial(2);
    uint32_t y_size = l.spatial(1);
    uint32_t x_size = l.spatial(0);

    cldnn::mem_lock<T> buff(output, get_test_stream());

    float err_margin = output->get_layout().data_type == data_types::f32 ? 1e-03F : 2e-02F;

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t f = 0; f < feature_size; ++f) {
            float sum = 0.f;
            float variance = 0.f;
            for (uint32_t z = 0; z < z_size; ++z) {
                for (uint32_t y = 0; y < y_size; ++y) {
                    for (uint32_t x = 0; x < x_size; ++x) {
                        auto index_tensor = tensor(batch(b), feature(f), spatial(x, y, z, 0));
                        size_t data_index = output->get_layout().get_linear_offset(index_tensor);
                        float data = static_cast<float>(buff[data_index]);
                        sum += data;
                        if (normalize_variance)
                            variance += data * data;
                    }
                }
            }
            sum /= y_size * x_size * z_size;
            T result_sum = static_cast<T>(sum);
            ASSERT_NEAR(result_sum, 0.f, err_margin) << "at b=" << b << ", f=" << f;

            if (normalize_variance) {
                variance /= y_size * x_size * z_size;
                T result_variance = static_cast<T>(variance);
                ASSERT_NEAR(result_variance, 1.f, err_margin) << " at b=" << b << ", f=" << f;
            }
        }
    }
}

template <typename T>
void test_mvn_test_across_channels_outside_sqrt_bfyx(bool is_caching_test) {
    // mvn across channels fp32 test with normalize_variance set to false
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    cldnn::data_types input_data_type = std::is_same<T, ov::float16>::value ? data_types::f16 : data_types::f32;

    auto input = engine.allocate_memory({input_data_type, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<T>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), false, 1e-10f, false, {1, 2, 3}));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<T>(output, false);
}

TEST(mvn_gpu_test, mvn_test_across_channels_outside_sqrt_bfyx) {
    test_mvn_test_across_channels_outside_sqrt_bfyx<float>(false);
}

template <typename T>
void test_mvn_test_across_channels_inside_sqrt_bfyx(bool is_caching_test) {
    // mvn across channels fp32 test with normalize_variance set to false
    using namespace cldnn;
    using namespace tests;

    auto& engine = get_test_engine();

    cldnn::data_types input_data_type = std::is_same<T, ov::float16>::value ? data_types::f16 : data_types::f32;

    auto input = engine.allocate_memory({input_data_type, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<T>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), false, 1e-10f, true, {1, 2, 3}));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<T>(output, false);
}

TEST(mvn_gpu_test, mvn_test_across_channels_inside_sqrt_bfyx) {
    test_mvn_test_across_channels_inside_sqrt_bfyx<float>(false);
}

TEST(mvn_gpu_test, mvn_test_across_channels_outside_sqrt_bfyx_fp16) {
    test_mvn_test_across_channels_outside_sqrt_bfyx<ov::float16>(false);
}

TEST(mvn_gpu_test, mvn_test_across_channels_inside_sqrt_bfyx_fp16) {
    test_mvn_test_across_channels_inside_sqrt_bfyx<ov::float16>(false);
}

TEST(mvn_gpu_test, mvn_test_across_channels_outside_sqrt_bfyx_normalize_variance) {
    // mvn across channels fp32 test with normalize_variance set to true
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<float>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), true, 1e-10f, false, {1, 2, 3}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<float>(output, true);
}

TEST(mvn_gpu_test, mvn_test_across_channels_inside_sqrt_bfyx_normalize_variance) {
    // mvn across channels fp32 test with normalize_variance set to true
    using namespace cldnn;
    using namespace tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<float>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), true, 1e-10f, true, {1, 2, 3}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<float>(output, true);
}

TEST(mvn_gpu_test, mvn_test_across_channels_outside_sqrt_bfyx_normalize_variance_fp16) {
    // mvn across channels fp16 test with normalize_variance set to true
    using namespace cldnn;
    using namespace tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<ov::float16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), true, 1e-10f, false, {1, 2, 3}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<ov::float16>(output, true);
}

TEST(mvn_gpu_test, mvn_test_across_channels_inside_sqrt_bfyx_normalize_variance_fp16) {
    // mvn across channels fp16 test with normalize_variance set to true
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<ov::float16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), true, 1e-10f, true, {1, 2, 3}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<ov::float16>(output, true);
}

TEST(mvn_gpu_test, dynamic_across_channels_inside_sqrt_bfyx_normalize_variance_fp16) {
    // mvn across channels fp16 test with normalize_variance set to true
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    ov::Shape in_shape = {7, 10, 17, 13};
    auto in_layout = layout{ov::PartialShape::dynamic(in_shape.size()), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape(in_shape), data_types::f16, format::bfyx});

    tests::set_random_values<ov::float16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(mvn("mvn", input_info("input"), true, 1e-10f, true, {1, 2, 3}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("mvn");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<ov::float16>(output, true);
}

TEST(mvn_gpu_test, mvn_test_within_channels_outside_sqrt_bfyx) {
    // mvn within channels fp32 test with normalize_variance set to false
    using namespace cldnn;
    using namespace tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<float>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), false, 1e-10f, false, {2, 3}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<float>(output, false);
}

TEST(mvn_gpu_test, mvn_test_within_channels_inside_sqrt__bfyx) {
    // mvn within channels fp32 test with normalize_variance set to false
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<float>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), false, 1e-10f, true, {2, 3}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<float>(output, false);
}

TEST(mvn_gpu_test, mvn_test_within_channels_outside_sqrt_bfyx_fp16) {
    // mvn within channels fp16 test with normalize_variance set to false
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<ov::float16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), false, 1e-10f, false, {2, 3}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<ov::float16>(output, false);
}

TEST(mvn_gpu_test, mvn_test_within_channels_inside_sqrt_bfyx_fp16) {
    // mvn within channels fp16 test with normalize_variance set to false
    using namespace cldnn;
    using namespace tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<ov::float16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), false, 1e-10f, true, {2, 3}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<ov::float16>(output, false);
}

TEST(mvn_gpu_test, mvn_test_within_channels_outside_sqrt_bfyx_normalize_variance) {
    // mvn within channels fp32 test with normalize_variance set to true
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<float>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), true, 1e-10f, false, {2, 3}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<float>(output, true);
}

TEST(mvn_gpu_test, mvn_test_within_channels_inside_sqrt_bfyx_normalize_variance) {
    // mvn within channels fp32 test with normalize_variance set to true
    using namespace cldnn;
    using namespace tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<float>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), true, 1e-10f, true, {2, 3}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<float>(output, true);
}

TEST(mvn_gpu_test, mvn_test_within_channels_outside_sqrt_bfyx_normalize_variance_fp16) {
    // mvn within channels fp16 test with normalize_variance set to true
    using namespace cldnn;
    using namespace tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<ov::float16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), true, 1e-10f, false, {2, 3}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<ov::float16>(output, true);
}

TEST(mvn_gpu_test, mvn_test_within_channels_inside_sqrt_bfyx_normalize_variance_fp16) {
    // mvn within channels fp16 test with normalize_variance set to true
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<ov::float16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", input_info("input"), true, 1e-10f, true, {2, 3}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<ov::float16>(output, true);
}

TEST(mvn_gpu_test, dynamic_within_channels_inside_sqrt_bfyx_normalize_variance_fp16) {
    // mvn within channels fp16 test with normalize_variance set to true
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    ov::Shape in_shape = {7, 10, 17, 13};
    auto in_layout = layout{ov::PartialShape::dynamic(in_shape.size()), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape(in_shape), data_types::f16, format::bfyx});

    tests::set_random_values<ov::float16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(mvn("mvn", input_info("input"), true, 1e-10f, true, {2, 3}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("mvn");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<ov::float16>(output, true);
}

struct mvn_basic_test_params {
    format::type input_format;
    data_types input_type;
    tensor input_size;
    bool normalize_variance;
    bool eps_inside_sqrt;
    bool across_channels;
    padding output_pad;
};

struct mvn_random_test : ::testing::TestWithParam<mvn_basic_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    template <typename T>
    void fill_data(memory::ptr mem, const tests::VVVVVF<T>& data) {
        auto l = mem->get_layout();
        cldnn::mem_lock<T> ptr(mem, get_test_stream());
        for (size_t bi = 0; bi < static_cast<size_t>(l.batch()); ++bi) {
            for (size_t fi = 0; fi < static_cast<size_t>(l.feature()); ++fi) {
                for (size_t zi = 0; zi < static_cast<size_t>(l.spatial(2)); ++zi) {
                    for (size_t yi = 0; yi < static_cast<size_t>(l.spatial(1)); ++yi) {
                        for (size_t xi = 0; xi < static_cast<size_t>(l.spatial(0)); ++xi) {
                            auto tensor_addr = tensor(batch(bi), feature(fi), spatial(xi, yi, zi, 0));
                            auto offset = mem->get_layout().get_linear_offset(tensor_addr);
                            ptr[offset] = data[bi][fi][xi][yi][zi];
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void fill_random_data(memory::ptr mem, int min, int max, int k = 8) {
        auto l = mem->get_layout();
        auto input_data = rg.generate_random_5d<T>(l.batch(),
                                                   l.feature(),
                                                   l.spatial(0),
                                                   l.spatial(1),
                                                   l.spatial(2),
                                                   min,
                                                   max,
                                                   k);
        fill_data(mem, input_data);
    }

    void check_result(memory::ptr output, bool across_channels, bool normalize_variance) {
        if (output->get_layout().data_type == data_types::f32) {
            if (across_channels) {
                mvn_compute_mean_across_channels<float>(output, normalize_variance);
            } else {
                mvn_compute_mean_within_channels<float>(output, normalize_variance);
            }
        } else if (output->get_layout().data_type == data_types::f16) {
            if (across_channels) {
                mvn_compute_mean_across_channels<ov::float16>(output, normalize_variance);
            } else {
                mvn_compute_mean_within_channels<ov::float16>(output, normalize_variance);
            }
        }
    }

    void execute(const mvn_basic_test_params& params, engine& eng, bool is_caching_test) {
        auto& size = params.input_size;
        auto& output_pad = params.output_pad;

        auto input = eng.allocate_memory({params.input_type, params.input_format, size});

        switch (params.input_type) {
            case data_types::f32:
                fill_random_data<float>(input, -127, 127);
                break;
            case data_types::f16:
                fill_random_data<ov::float16>(input, -127, 127);
                break;
            case data_types::i8:
                fill_random_data<int8_t>(input, -127, 127);
                break;
            case data_types::u8:
                fill_random_data<uint8_t>(input, -127, 127);
                break;
            default:
                break;
        }
        auto axes = params.across_channels ? std::vector<int64_t>{1, 2, 3} : std::vector<int64_t>{2, 3};
        topology topo;
        topo.add(input_layout("input", input->get_layout()));
        auto prim = mvn("mvn", input_info("input"), params.normalize_variance, 1e-10f, false, axes);
        prim.output_paddings = {output_pad};
        topo.add(prim);

        cldnn::network::ptr net = get_network(eng, topo, get_test_default_config(eng), get_test_stream_ptr(), is_caching_test);

        net->set_input_data("input", input);

        auto outputs = net->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "mvn");

        auto output = outputs.begin()->second.get_memory();
        check_result(output, params.across_channels, params.normalize_variance);
    }
};

TEST_P(mvn_random_test, random) {
    auto& engine = tests::get_test_engine();
    this->execute(GetParam(), engine, false);
}

struct mvn_test_case_generator : std::vector<mvn_basic_test_params> {
    mvn_test_case_generator& add(mvn_basic_test_params params) {
        push_back(params);
        return *this;
    }

    mvn_test_case_generator& smoke_tests(format::type fmt, data_types in_dt) {
        push_back(mvn_basic_test_params{fmt, in_dt, {7, 10, 17, 13}, false, false, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {7, 10, 17, 13}, true, false, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {7, 10, 17, 13}, false, true, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {7, 10, 17, 13}, false, false, true, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {7, 10, 17, 13}, true, true, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {7, 10, 17, 13}, true, false, true, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {7, 10, 17, 13}, false, true, true, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {7, 10, 17, 13}, true, true, true, padding()});
        return *this;
    }

    mvn_test_case_generator& zyx_tests(format::type fmt, data_types in_dt) {
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 1, 67, 71}, false, false, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 1, 67, 71}, true, false, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 1, 67, 71}, false, true, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 1, 67, 71}, false, false, true, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 1, 67, 71}, true, true, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 1, 67, 71}, true, false, true, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 1, 67, 71}, false, true, true, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 1, 67, 71}, true, true, true, padding()});
        return *this;
    }

    mvn_test_case_generator& extended_tests(format::type fmt, data_types in_dt) {
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, false, false, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, true, false, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, false, true, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, false, false, true, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, true, true, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, true, false, true, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, false, true, true, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, true, true, true, padding()});
        // output padding
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, false, false, false, padding({0, 0, 1, 1})});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, true, false, false, padding({0, 0, 1, 1})});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, false, true, false, padding({0, 0, 1, 1})});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, false, false, true, padding({0, 0, 1, 1})});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, true, true, false, padding({0, 0, 1, 1})});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, true, false, true, padding({0, 0, 1, 1})});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, false, true, true, padding({0, 0, 1, 1})});
        push_back(mvn_basic_test_params{fmt, in_dt, {2, 17, 67, 71}, true, true, true, padding({0, 0, 1, 1})});

        return *this;
    }
};

INSTANTIATE_TEST_SUITE_P(smoke,
                        mvn_random_test,
                        testing::ValuesIn(mvn_test_case_generator()
                                              .smoke_tests(format::b_fs_yx_fsv16, data_types::i8)
                                              .smoke_tests(format::b_fs_yx_fsv16, data_types::u8)));

INSTANTIATE_TEST_SUITE_P(zyx,
                        mvn_random_test,
                        testing::ValuesIn(mvn_test_case_generator()
                                              .zyx_tests(format::b_fs_zyx_fsv16, data_types::i8)
                                              .zyx_tests(format::b_fs_zyx_fsv16, data_types::u8)));

INSTANTIATE_TEST_SUITE_P(extended,
                        mvn_random_test,
                        testing::ValuesIn(mvn_test_case_generator()
                                              .extended_tests(format::b_fs_yx_fsv16, data_types::i8)
                                              .extended_tests(format::b_fs_yx_fsv16, data_types::u8)));

struct mvn_random_test_bsv32 : ::testing::TestWithParam<mvn_basic_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    template <typename T>
    void fill_data(cldnn::memory::ptr mem, const tests::VVVVVF<T>& data) {
        auto l = mem->get_layout();
        cldnn::mem_lock<T> ptr(mem, get_test_stream());
        for (size_t bi = 0; bi < static_cast<size_t>(l.batch()); ++bi) {
            for (size_t fi = 0; fi < static_cast<size_t>(l.feature()); ++fi) {
                for (size_t zi = 0; zi < static_cast<size_t>(l.spatial(2)); ++zi) {
                    for (size_t yi = 0; yi < static_cast<size_t>(l.spatial(1)); ++yi) {
                        auto tensor_addr = tensor(batch(bi), feature(fi), spatial(0, yi, zi, 0));
                        auto offset = mem->get_layout().get_linear_offset(tensor_addr);
                        for (size_t xi = 0; xi < static_cast<size_t>(l.spatial(0)); ++xi) {
                            ptr[offset + xi] = data[bi][fi][xi][yi][zi];
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void fill_random_data(cldnn::memory::ptr mem, int min, int max, int k = 8) {
        auto l = mem->get_layout();
        auto input_data = rg.generate_random_5d<T>(l.batch(),
                                                   l.feature(),
                                                   l.spatial(0),
                                                   l.spatial(1),
                                                   l.spatial(2),
                                                   min,
                                                   max,
                                                   k);
        fill_data(mem, input_data);
    }

    size_t get_x_pitch(layout& layout) {
        auto tensor_x0 = tensor(batch(0), feature(0), spatial(0, 0, 0, 0));
        auto tensor_x1 = tensor(batch(0), feature(0), spatial(1, 0, 0, 0));
        auto x0 = layout.get_linear_offset(tensor_x0);
        auto x1 = layout.get_linear_offset(tensor_x1);
        return (x1 - x0);
    }

    template <typename T>
    void compare_outputs(const cldnn::memory::ptr out_ref, const cldnn::memory::ptr out_opt) {
        auto output_lay = out_ref->get_layout();
        auto opt_output_lay = out_opt->get_layout();

        size_t b = output_lay.batch();
        size_t f = output_lay.feature();
        size_t x = output_lay.spatial(0);
        size_t y = output_lay.spatial(1);
        cldnn::mem_lock<T> ref_ptr(out_ref, get_test_stream());
        cldnn::mem_lock<T> opt_ptr(out_opt, get_test_stream());

        auto ref_x_pitch = get_x_pitch(output_lay);
        auto opt_x_pitch = get_x_pitch(opt_output_lay);

        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    auto ref_out_coords = tensor(batch(bi), feature(fi), spatial(0, yi, 0, 0));
                    auto ref_out_offset = output_lay.get_linear_offset(ref_out_coords);
                    auto opt_out_offset = opt_output_lay.get_linear_offset(ref_out_coords);
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto ref_out_val = ref_ptr[ref_out_offset + xi * ref_x_pitch];
                        auto opt_out_val = opt_ptr[opt_out_offset + xi * opt_x_pitch];
                        ASSERT_NEAR(static_cast<float>(opt_out_val), static_cast<float>(ref_out_val), 1.e-1f);
                    }
                }
            }
        }
    }

    void execute(const mvn_basic_test_params& params, bool is_caching_test) {
        auto& size = params.input_size;
        auto& output_pad = params.output_pad;
        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({params.input_type, format::bfyx, params.input_size});
        switch (params.input_type) {
            case data_types::f32:
                fill_random_data<float>(input, -127, 127);
                break;
            case data_types::f16:
                fill_random_data<ov::float16>(input, -127, 127, 1);
                break;
            case data_types::i8:
                fill_random_data<int8_t>(input, -127, 127, 1);
                break;
            case data_types::u8:
                fill_random_data<uint8_t>(input, 0, 255, 1);
                break;
            default:
                break;
        }

        auto axes = params.across_channels ? std::vector<int64_t>{1, 2, 3} : std::vector<int64_t>{2, 3};
        topology topo;
        topo.add(input_layout("input", input->get_layout()));
        auto prim = mvn("mvn", input_info("input"), params.normalize_variance, 1e-10f, false, axes);
        prim.output_paddings = {output_pad};
        topo.add(prim);
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"mvn"}));
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"mvn", {format::type::bfyx, "mvn_gpu_bfyx_opt"}} }));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("input", input);

        auto outputs = net->execute();
        auto output = outputs.at("mvn").get_memory();

        topology topo_opt;
        topo_opt.add(input_layout("input", input->get_layout()));
        topo_opt.add(reorder("input_to_target_layout", input_info("input"), {params.input_type, params.input_format, size}));
        auto prim_opt = mvn("mvn_opt", input_info("input_to_target_layout"), params.normalize_variance, 1e-10f, false, axes);
        prim_opt.output_paddings = {output_pad};
        topo_opt.add(prim_opt);
        ExecutionConfig config_opt = get_test_default_config(engine);
        config_opt.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"mvn_opt", "input_to_target_layout"}));
        config_opt.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"mvn_opt", {params.input_format, "mvn_gpu_b_fs_yx_fsv16_imad"}} }));

        cldnn::network::ptr net_opt = get_network(engine, topo_opt, config_opt, get_test_stream_ptr(), is_caching_test);

        net_opt->set_input_data("input", input);

        auto outputs_opt = net_opt->execute();
        auto output_opt = outputs_opt.at("mvn_opt").get_memory();

        auto output_dtype = output->get_layout().data_type;
        auto output_opt_dtype = output_opt->get_layout().data_type;
        if (output_dtype == output_opt_dtype) {
            if(output_dtype == data_types::f32) {
                compare_outputs<float>(output, output_opt);
            } else if (output_dtype == data_types::f16) {
                compare_outputs<ov::float16>(output, output_opt);
            } else if (output_dtype == data_types::i8) {
                compare_outputs<int8_t>(output, output_opt);
            } else if (output_dtype == data_types::u8) {
                compare_outputs<uint8_t>(output, output_opt);
            } else {
                FAIL() << "Not supported data type: " << static_cast<size_t>(params.input_type);
            }
        } else {
            FAIL() << "Outputs have diffent data types: "
                << static_cast<size_t>(output_dtype) << ", "
                << static_cast<size_t>(output_opt_dtype);
        }
    }
};

TEST_P(mvn_random_test_bsv32, random) {
    this->execute(GetParam(), false);
}

struct mvn_test_case_generator_bsv32 : std::vector<mvn_basic_test_params> {
    mvn_test_case_generator_bsv32& add(mvn_basic_test_params params) {
        push_back(params);
        return *this;
    }

    mvn_test_case_generator_bsv32& bsv32_tests(format::type fmt, data_types in_dt) {
        push_back(mvn_basic_test_params{fmt, in_dt, {32, 32, 10, 10}, true, false, false, padding()});
        push_back(mvn_basic_test_params{fmt, in_dt, {32, 32, 10, 10}, false, false, false, padding()});
        return *this;
    }
};

INSTANTIATE_TEST_SUITE_P(mvn_bsv32_fsv32,
                        mvn_random_test_bsv32,
                        testing::ValuesIn(mvn_test_case_generator_bsv32()
                                              .bsv32_tests(format::bs_fs_yx_bsv32_fsv32, data_types::i8)));


INSTANTIATE_TEST_SUITE_P(mvn_bsv32_fsv16,
                        mvn_random_test_bsv32,
                        testing::ValuesIn(mvn_test_case_generator_bsv32()
                                              .bsv32_tests(format::bs_fs_yx_bsv32_fsv16, data_types::f16)));

INSTANTIATE_TEST_SUITE_P(mvn_fsv16,
                        mvn_random_test_bsv32,
                        testing::ValuesIn(mvn_test_case_generator_bsv32()
                                              .bsv32_tests(format::b_fs_yx_fsv16, data_types::i8)));

TEST(mvn_gpu_test, mvn_test_across_channels_outside_sqrt_bfyx_cached) {
    test_mvn_test_across_channels_outside_sqrt_bfyx<float>(true);
}
#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(mvn_gpu_test, mvn_test_across_channels_inside_sqrt_bfyx_cached) {
    test_mvn_test_across_channels_inside_sqrt_bfyx<float>(true);
}

TEST(mvn_gpu_test, mvn_test_across_channels_outside_sqrt_bfyx_fp16_cached) {
    test_mvn_test_across_channels_outside_sqrt_bfyx<ov::float16>(true);
}

TEST(mvn_gpu_test, mvn_test_across_channels_inside_sqrt_bfyx_fp16_cached) {
    test_mvn_test_across_channels_inside_sqrt_bfyx<ov::float16>(true);
}

TEST_P(mvn_random_test, random_cached) {
    auto& engine = tests::get_test_engine();
    this->execute(GetParam(), engine, true);
}

TEST_P(mvn_random_test_bsv32, random_cached) {
    this->execute(GetParam(), true);
}
#endif

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/mvn.hpp>
#include <cldnn/primitives/reorder.hpp>

#include <iostream>

using namespace cldnn;
using namespace ::tests;

class mvn_gpu_test : public ::testing::TestWithParam<cldnn::format> {};

template <typename T>
void mvn_compute_mean_across_channels(cldnn::memory::ptr output, bool normalize_variance) {
    auto output_size = output->get_layout().size;

    uint32_t batch_size = output_size.batch[0];
    uint32_t feature_size = output_size.feature[0];
    uint32_t z_size = output_size.spatial[2];
    uint32_t y_size = output_size.spatial[1];
    uint32_t x_size = output_size.spatial[0];

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
        EXPECT_NEAR(result_sum, 0.f, err_margin) << "at b=" << b;

        if (normalize_variance) {
            variance /= feature_size * y_size * x_size * z_size;
            T result_variance = static_cast<T>(variance);
            EXPECT_NEAR(result_variance, 1.f, err_margin) << " at b=" << b;
        }
    }
}

template <typename T>
void mvn_compute_mean_within_channels(cldnn::memory::ptr output, bool normalize_variance) {
    auto output_size = output->get_layout().size;

    uint32_t batch_size = output_size.batch[0];
    uint32_t feature_size = output_size.feature[0];
    uint32_t z_size = output_size.spatial[2];
    uint32_t y_size = output_size.spatial[1];
    uint32_t x_size = output_size.spatial[0];

    cldnn::mem_lock<T> buff(output, get_test_stream());

    float err_margin = output->get_layout().data_type == data_types::f32 ? 1e-03F : 1e-02F;

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
            EXPECT_NEAR(result_sum, 0.f, err_margin) << "at b=" << b << ", f=" << f;

            if (normalize_variance) {
                variance /= y_size * x_size * z_size;
                T result_variance = static_cast<T>(variance);
                EXPECT_NEAR(result_variance, 1.f, err_margin) << " at b=" << b << ", f=" << f;
            }
        }
    }
}

TEST(mvn_gpu_test, mvn_test_across_channels_outside_sqrt_bfyx) {
    // mvn across channels fp32 test with normalize_variance set to false
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<float>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", "input", false, 1e-10f, false, true));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<float>(output, false);
}

TEST(mvn_gpu_test, mvn_test_across_channels_inside_sqrt_bfyx) {
    // mvn across channels fp32 test with normalize_variance set to false
    using namespace cldnn;
    using namespace tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<float>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", "input", false, 1e-10f, true, true));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<float>(output, false);
}

TEST(mvn_gpu_test, mvn_test_across_channels_bfyx_outside_sqrt_fp16) {
    // mvn across channels fp16 test with normalize_variance set to false
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<FLOAT16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", "input", false, 1e-10f, false, true));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<FLOAT16>(output, false);
}

TEST(mvn_gpu_test, mvn_test_across_channels_inside_sqrt_bfyx_fp16) {
    // mvn across channels fp16 test with normalize_variance set to false
    using namespace cldnn;
    using namespace tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<FLOAT16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", "input", false, 1e-10f, true, true));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<FLOAT16>(output, false);
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
    topology.add(mvn("mvn", "input", true, 1e-10f, false, true));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

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
    topology.add(mvn("mvn", "input", true, 1e-10f, true, true));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<float>(output, true);
}

TEST(mvn_gpu_test, mvn_test_across_channels_outside_sqrt_bfyx_normalize_variance_fp16) {
    // mvn across channels fp16 test with normalize_variance set to true
    using namespace cldnn;
    using namespace tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<FLOAT16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", "input", true, 1e-10f, false, true));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<FLOAT16>(output, true);
}

TEST(mvn_gpu_test, mvn_test_across_channels_inside_sqrt_bfyx_normalize_variance_fp16) {
    // mvn across channels fp16 test with normalize_variance set to true
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<FLOAT16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", "input", true, 1e-10f, true, true));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_across_channels<FLOAT16>(output, true);
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
    topology.add(mvn("mvn", "input", false, 1e-10f, false, false));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

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
    topology.add(mvn("mvn", "input", false, 1e-10f, true, false));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<float>(output, false);
}

TEST(mvn_gpu_test, mvn_test_within_channels_outside_sqrt_bfyx_fp16) {
    // mvn within channels fp16 test with normalize_variance set to false
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<FLOAT16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", "input", false, 1e-10f, false, false));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<FLOAT16>(output, false);
}

TEST(mvn_gpu_test, mvn_test_within_channels_inside_sqrt_bfyx_fp16) {
    // mvn within channels fp16 test with normalize_variance set to false
    using namespace cldnn;
    using namespace tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<FLOAT16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", "input", false, 1e-10f, true, false));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<FLOAT16>(output, false);
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
    topology.add(mvn("mvn", "input", true, 1e-10f, false, false));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

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
    topology.add(mvn("mvn", "input", true, 1e-10f, true, false));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<float>(output, true);
}

TEST(mvn_gpu_test, mvn_test_within_channels_outside_sqrt_bfyx_normalize_variance_fp16) {
    // mvn within channels fp16 test with normalize_variance set to true
    using namespace cldnn;
    using namespace tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<FLOAT16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", "input", true, 1e-10f, false, false));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<FLOAT16>(output, true);
}

TEST(mvn_gpu_test, mvn_test_within_channels_inside_sqrt_bfyx_normalize_variance_fp16) {
    // mvn within channels fp16 test with normalize_variance set to true
    using namespace cldnn;
    using namespace ::tests;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {7, 10, 17, 13}});

    tests::set_random_values<FLOAT16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mvn("mvn", "input", true, 1e-10f, true, false));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels<FLOAT16>(output, true);
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
    template <typename T>
    void fill_data(memory::ptr mem, const tests::VVVVVF<T>& data) {
        auto size = mem->get_layout().size;
        cldnn::mem_lock<T> ptr(mem, get_test_stream());
        for (size_t bi = 0; bi < static_cast<size_t>(size.batch[0]); ++bi) {
            for (size_t fi = 0; fi < static_cast<size_t>(size.feature[0]); ++fi) {
                for (size_t zi = 0; zi < static_cast<size_t>(size.spatial[2]); ++zi) {
                    for (size_t yi = 0; yi < static_cast<size_t>(size.spatial[1]); ++yi) {
                        for (size_t xi = 0; xi < static_cast<size_t>(size.spatial[0]); ++xi) {
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
        auto size = mem->get_layout().size;
        auto input_data = tests::generate_random_5d<T>(size.batch[0],
                                                       size.feature[0],
                                                       size.spatial[0],
                                                       size.spatial[1],
                                                       size.spatial[2],
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
                mvn_compute_mean_across_channels<FLOAT16>(output, normalize_variance);
            } else {
                mvn_compute_mean_within_channels<FLOAT16>(output, normalize_variance);
            }
        }
    }

    void execute(const mvn_basic_test_params& params, engine& eng) {
        auto& size = params.input_size;
        auto& output_pad = params.output_pad;

        auto input = eng.allocate_memory({params.input_type, params.input_format, size});

        switch (params.input_type) {
            case data_types::f32:
                fill_random_data<float>(input, -127, 127);
                break;
            case data_types::f16:
                fill_random_data<FLOAT16>(input, -127, 127);
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

        topology topo;
        topo.add(input_layout("input", input->get_layout()));
        auto prim = mvn("mvn", "input", params.normalize_variance, 1e-10f, false, params.across_channels);
        prim.output_padding = output_pad;
        topo.add(prim);

        network net(eng, topo);

        net.set_input_data("input", input);

        auto outputs = net.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "mvn");

        auto output = outputs.begin()->second.get_memory();
        check_result(output, params.across_channels, params.normalize_variance);
    }
};

TEST_P(mvn_random_test, random) {
    auto& eng = tests::get_test_engine();
    this->execute(GetParam(), eng);
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

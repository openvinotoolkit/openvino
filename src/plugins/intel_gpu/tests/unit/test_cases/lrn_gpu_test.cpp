// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/lrn.hpp>
#include <intel_gpu/primitives/input_layout.hpp>

using namespace cldnn;
using namespace ::tests;

template <typename T>
void test_fp32_basic(bool is_caching_test) {
    //  input : 1x16x1x1
    //  Output : 1x16x1x1
    auto& engine = get_test_engine();

    const size_t b = 1;
    const size_t f = 16;
    const size_t y = 1;
    const size_t x = 1;

    auto input = engine.allocate_memory({ data_types::f32, format::b_fs_yx_fsv16, { b, f, x, y } });
    std::vector<T> inputVals(b * f * y * x);
    T n = 0;
    std::generate(inputVals.begin(), inputVals.end(), [n]() mutable { return n++; });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    uint32_t size = 2;
    float k = 0.5f;
    float alpha = 9.9e-05f;
    float beta = 1.f;
    topology.add(lrn("lrn", input_info("input"), size, k, alpha, beta, cldnn::lrn_norm_region_across_channel));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);

    auto outputs = network->execute();

    auto output = outputs.at("lrn").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 1.99901f, 3.99486f, 5.98519f,
        7.96766f, 9.93997f, 11.8999f, 13.8451f,
        15.7736f, 17.6831f, 19.5718f, 21.4376f,
        23.2787f, 25.0933f, 26.8797f, 29.3463f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i])) << i;
    }
}

TEST(lrn_fp32_gpu, basic) {
    test_fp32_basic<float>(false);
}

template <typename T>
void test_fp32_basic2(bool is_caching_test) {
    //  input : 1x16x1x1
    //  Output : 1x16x1x1
    auto& engine = get_test_engine();

    const size_t b = 1;
    const size_t f = 16;
    const size_t y = 1;
    const size_t x = 1;

    auto input = engine.allocate_memory({ data_types::f32, format::b_fs_yx_fsv16, { b, f, x, y } });
    std::vector<T> inputVals(b * f * y * x);
    T n = 0;
    std::generate(inputVals.begin(), inputVals.end(), [n]() mutable { return n++; });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    uint32_t size = 5;
    float k = 0.5f;
    float alpha = 9.9e-05f;
    float beta = 1.f;
    topology.add(lrn("lrn", input_info("input"), size, k, alpha, beta, cldnn::lrn_norm_region_across_channel));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);

    auto outputs = network->execute();

    auto output = outputs.at("lrn").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 1.99889f, 3.99525f, 5.98696f,
        7.97159f, 9.94682f, 11.9104f, 13.86f,
        15.7936f, 17.709f, 19.6041f, 21.4769f,
        23.3257f, 25.1485f, 27.2091f, 29.3151f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i])) << i;
    }
}

TEST(lrn_fp32_gpu, basic2) {
    test_fp32_basic2<float>(false);
}

template <typename T>
void test_fp16_basic1(bool is_caching_test) {
    //  input : 1x16x1x1
    //  Output : 1x16x1x1
    auto& engine = get_test_engine();

    const size_t b = 1;
    const size_t f = 16;
    const size_t y = 1;
    const size_t x = 1;

    auto input = engine.allocate_memory({ data_types::f16, format::b_fs_yx_fsv16, { b, f, x, y } });
    std::vector<T> inputVals(b * f * y * x);
    float n = 0;
    std::generate(inputVals.begin(), inputVals.end(), [n]() mutable { return T(n++); });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    uint32_t size = 5;
    float k = 0.5f;
    float alpha = 9.9e-05f;
    float beta = 1.f;
    topology.add(lrn("lrn", input_info("input"), size, k, alpha, beta, cldnn::lrn_norm_region_across_channel));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);

    auto outputs = network->execute();

    auto output = outputs.at("lrn").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 1.99889f, 3.99525f, 5.98696f,
        7.97159f, 9.94682f, 11.9104f, 13.86f,
        15.7936f, 17.709f, 19.6041f, 21.4769f,
        23.3257f, 25.1485f, 27.2091f, 29.3151f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]))) << i;
    }
}

TEST(lrn_fp16_gpu, basic1) {
    test_fp16_basic1<ov::float16>(false);
}

template <typename T>
void test_fp32_basic3(bool is_caching_test) {
    //  input : 2x16x4x4
    //  Output : 2x16x4x4
    auto& engine = get_test_engine();

    const size_t b = 2;
    const size_t f = 16;
    const size_t y = 4;
    const size_t x = 4;

    auto input = engine.allocate_memory({ data_types::f32, format::b_fs_yx_fsv16, { b, f, x, y } });
    std::vector<T> inputVals(b * f * y * x);
    T n = 0;
    std::generate(inputVals.begin(), inputVals.end(), [n]() mutable { return n++; });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    uint32_t size = 5;
    float k = 1.f;
    float alpha = 9.89999971e-05f;
    float beta = 0.75f;
    topology.add(lrn("lrn", input_info("input"), size, k, alpha, beta, cldnn::lrn_norm_region_across_channel));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);

    auto outputs = network->execute();

    auto output = outputs.at("lrn").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f,      0.999792f, 1.99911f, 2.99755f, 3.99466f, 4.99f,    5.98313f, 6.97361f, 7.96102f, 8.94493f, 9.92493f,
        10.9006f, 11.8715f,  12.8374f, 13.8493f, 14.8699f, 15.7966f, 16.696f,  17.5763f, 18.5035f, 19.4231f, 20.3347f,
        21.2381f, 22.133f,   23.019f,  23.8959f, 24.7635f, 25.6216f, 26.4698f, 27.308f,  28.5352f, 29.8116f, 30.5296f,
        30.9563f, 31.342f,   32.1164f, 32.8795f, 33.6315f, 34.372f,  35.1011f, 35.8188f, 36.5248f, 37.2192f, 37.9019f,
        38.573f,  39.2323f,  41.0464f, 43.0053f, 43.4314f, 42.8943f, 42.3534f, 42.9426f, 43.5202f, 44.0863f, 44.641f,
        45.1843f, 45.7162f,  46.2369f, 46.7464f, 47.2449f, 47.7324f, 48.209f,  50.9356f, 53.9937f, 54.1048f, 52.193f,
        50.4329f, 50.8466f,  51.2503f, 51.6441f, 52.0281f, 52.4024f, 52.7672f, 53.1226f, 53.4688f, 53.8059f, 54.1339f,
        54.4532f, 58.2539f,  62.6803f, 62.5006f, 58.9907f, 55.9219f, 56.1912f, 56.4526f, 56.7063f, 56.9524f, 57.1911f,
        57.4225f, 57.6468f,  57.864f,  58.0744f, 58.278f,  58.4751f, 63.3482f, 69.2298f, 68.8098f, 63.6702f, 59.3655f,
        59.5255f, 59.6799f,  59.8286f, 59.9718f, 60.1096f, 60.2422f, 60.3697f, 60.4921f, 60.6096f, 60.7223f, 60.8303f,
        66.6672f, 73.9448f,  73.3429f, 66.6814f, 61.3032f, 61.3851f, 61.463f,  61.537f,  61.6071f, 61.6735f, 61.7363f,
        61.7954f, 61.8511f,  61.9035f, 61.9524f, 61.9982f, 68.6425f, 77.1679f, 76.4387f, 68.44f,   62.181f,  62.2089f,
        62.2341f, 62.2566f,  62.2765f, 62.2939f, 62.3088f, 62.3213f, 62.3314f, 62.3393f, 62.345f,  62.3484f, 69.6361f,
        79.2224f, 78.4118f,  69.288f,  62.335f,  62.3266f, 62.3163f, 62.3042f, 62.2905f, 62.275f,  62.258f,  62.2393f,
        62.2191f, 62.1974f,  62.1742f, 62.1496f, 69.9303f, 80.3855f, 79.5293f, 69.4891f, 62.0065f, 61.9741f, 61.9405f,
        61.9058f, 61.87f,    61.8331f, 61.7951f, 61.7561f, 61.7162f, 61.6752f, 61.6333f, 61.5906f, 69.7361f, 80.882f,
        80.0067f, 69.2391f,  61.3639f, 61.3161f, 61.2677f, 61.2185f, 61.1686f, 61.1181f, 61.0669f, 61.015f,  60.9625f,
        60.9095f, 60.8559f,  60.8017f, 69.207f,  80.8876f, 80.0121f, 68.6802f, 60.523f,  60.4658f, 60.4082f, 60.3501f,
        60.2917f, 60.2328f,  60.1736f, 60.1139f, 60.054f,  59.9937f, 59.933f,  59.872f,  68.4534f, 80.5369f, 79.6741f,
        67.9147f, 59.5628f,  59.5002f, 59.4373f, 59.3742f, 59.3109f, 59.2473f, 59.1836f, 59.1197f, 59.0555f, 58.9912f,
        58.9268f, 58.8622f,  67.5537f, 79.9313f, 79.0897f, 67.0151f, 58.537f,  58.4716f, 58.4061f, 58.3405f, 58.2748f,
        58.209f,  58.1431f,  58.0772f, 58.0112f, 57.9451f, 57.879f,  57.8128f, 66.5636f, 79.1467f, 78.3317f, 66.0328f,
        57.4815f, 57.4151f,  57.3488f, 57.2824f, 57.216f,  57.1496f, 57.0832f, 57.0169f, 56.9505f, 56.8842f, 56.8178f,
        56.7515f, 65.5222f,  78.2394f, 77.4541f, 65.0043f, 56.4205f, 56.3544f, 56.2884f, 56.2224f, 56.1564f, 56.0905f,
        56.0247f, 55.9589f,  55.8932f, 55.8275f, 55.762f,  55.6965f, 64.4569f, 77.2513f, 76.4972f, 63.955f,  55.37f,
        55.305f,  55.24f,    55.1751f, 55.1104f, 55.0457f, 54.9811f, 54.9165f, 54.8521f, 54.7878f, 54.7236f, 54.6595f,
        63.3869f, 76.2131f,  75.4906f, 62.9026f, 54.3404f, 54.2769f, 54.2135f, 54.1503f, 54.0871f, 54.0241f, 53.9611f,
        53.8983f, 53.8356f,  53.773f,  53.7105f, 53.6482f, 62.3253f, 75.1476f, 74.4564f, 61.8595f, 53.3382f, 53.2766f,
        53.2151f, 53.1537f,  53.0924f, 53.0313f, 52.9703f, 52.9094f, 52.8487f, 52.788f,  52.7275f, 52.6672f, 61.2811f,
        74.0714f, 73.4107f,  60.834f,  52.3673f, 52.3077f, 52.2482f, 52.1889f, 52.1297f, 52.0706f, 52.0117f, 51.9529f,
        51.8942f, 51.8357f,  51.7773f, 51.719f,  60.2603f, 72.9966f, 72.3652f, 59.8317f, 51.4297f, 51.3722f, 51.3149f,
        51.2577f, 51.2006f,  51.1437f, 51.0869f, 51.0302f, 50.9737f, 50.9173f, 50.861f,  50.8049f, 59.2667f, 71.9317f,
        71.3285f, 58.8561f,  50.5263f, 50.4709f, 50.4157f, 50.3607f, 50.3058f, 50.251f,  50.1963f, 50.1418f, 50.0874f,
        50.0331f, 49.979f,   49.925f,  58.3025f, 70.8832f, 70.3067f, 57.9093f, 49.657f,  49.6038f, 49.5508f, 49.4978f,
        49.445f,  49.3923f,  49.3398f, 49.2873f, 49.2351f, 49.1829f, 49.1309f, 49.079f,  57.3689f, 69.8552f, 69.3041f,
        56.9924f, 48.8214f,  48.7703f, 48.7193f, 48.6684f, 48.6177f, 48.567f,  48.5165f, 48.4662f, 48.4159f, 48.3658f,
        48.3158f, 48.266f,   56.4664f, 68.8508f, 68.3236f, 56.1058f, 48.0185f, 47.9694f, 47.9204f, 47.8715f, 47.8228f,
        47.7741f, 47.7256f,  47.6772f, 47.629f,  47.5808f, 47.5328f, 47.4849f, 55.5949f, 67.8717f, 67.3672f, 55.2492f,
        47.2472f, 47.2f,     47.1529f, 47.106f,  47.0591f, 47.0124f, 46.9658f, 46.9193f, 46.8729f, 46.8267f, 46.7806f,
        46.7345f, 54.7537f,  66.9191f, 66.436f,  54.4224f, 46.5061f, 46.4608f, 46.4156f, 46.3705f, 46.3254f, 46.2806f,
        46.2358f, 46.1911f,  46.1466f, 46.1021f, 46.0578f, 46.0135f, 53.9422f, 65.9936f, 65.5306f, 53.6244f, 45.7941f,
        45.7505f, 45.707f,   45.6636f, 45.6204f, 45.5772f, 45.5342f, 45.4913f, 45.4484f, 45.4057f, 45.3631f, 45.3206f,
        53.1594f, 65.0953f,  64.6512f, 52.8544f, 45.1096f, 45.0677f, 45.0259f, 44.9842f, 44.9426f, 44.9011f, 44.8597f,
        44.8184f, 44.7773f,  44.7362f, 44.6952f, 44.6543f, 52.4043f, 64.2239f, 63.7976f, 52.1113f, 44.4513f, 44.411f,
        44.3708f, 44.3307f,  44.2907f, 44.2508f, 44.211f,  44.1713f, 44.1317f, 44.0921f, 44.0527f, 44.0134f, 51.6758f,
        63.379f,  62.9695f,  51.3943f, 43.8181f, 43.7793f, 43.7406f, 43.702f,  43.6635f, 43.6251f, 43.5867f, 43.5485f,
        43.5104f, 43.4723f,  43.4343f, 43.3965f, 50.9729f, 62.56f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i])) << i;
    }
}

TEST(lrn_fp32_gpu, basic3) {
    test_fp32_basic3<float>(false);
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(lrn_fp32_gpu, basic_cached) {
    test_fp32_basic<float>(true);
}

TEST(lrn_fp32_gpu, basic2_cached) {
    test_fp32_basic2<float>(true);
}

TEST(lrn_fp16_gpu, basic1_cached) {
    test_fp16_basic1<ov::float16>(true);
}
#endif
TEST(lrn_fp32_gpu, basic3_cached) {
    test_fp32_basic3<float>(true);
}

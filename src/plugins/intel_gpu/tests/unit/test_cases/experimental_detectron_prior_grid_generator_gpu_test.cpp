// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/experimental_detectron_prior_grid_generator.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <string>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

template <typename T>
struct ExperimentalDetectronPriorGridGeneratorParams {
    std::vector<T> priors;
    tensor priorsTensor;
    int h;
    int w;
    float strideX;
    float strideY;
    bool flatten;
    std::pair<int, int> featureShape;
    std::pair<int, int> imageShape;
    tensor outputTensor;
    std::vector<T> expectedOutput;
    bool is_caching_test;
};

template <typename T>
struct experimental_detectron_prior_grid_generator_test
    : public ::testing::TestWithParam<ExperimentalDetectronPriorGridGeneratorParams<T>> {
public:
    void test() {
        auto data_type = ov::element::from<T>();
        ExperimentalDetectronPriorGridGeneratorParams<T> params =
            testing::TestWithParam<ExperimentalDetectronPriorGridGeneratorParams<T>>::GetParam();
        auto& engine = get_test_engine();

        auto prior_input = engine.allocate_memory({data_type, format::bfyx, params.priorsTensor});

        set_values(prior_input, params.priors);

        const std::string priors_id = "priors";
        const std::string experimental_detectron_prior_grid_generator_id =
            "experimental_detectron_prior_grid_generator";
        topology topology;
        topology.add(input_layout(priors_id, prior_input->get_layout()));

        cldnn::layout outLayout{data_type, cldnn::format::bfyx, params.outputTensor};
        topology.add(experimental_detectron_prior_grid_generator(experimental_detectron_prior_grid_generator_id,
                                                                 { input_info(priors_id) },
                                                                 params.flatten,
                                                                 params.h,
                                                                 params.w,
                                                                 params.strideX,
                                                                 params.strideY,
                                                                 params.featureShape.first,
                                                                 params.featureShape.second,
                                                                 params.imageShape.first,
                                                                 params.imageShape.second));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), params.is_caching_test);

        network->set_input_data(priors_id, prior_input);

        auto result = network->execute();

        auto out_mem = result.at(experimental_detectron_prior_grid_generator_id).get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.outputTensor.count(), out_ptr.size());
        for (size_t i = 0; i < params.expectedOutput.size(); ++i) {
            ASSERT_NEAR(params.expectedOutput[i], out_ptr[i], 0.0001) << "at i = " << i;
        }
    }
};

template <typename T>
std::vector<T> getValues(const std::vector<float>& values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template <typename T>
std::vector<ExperimentalDetectronPriorGridGeneratorParams<T>> generateExperimentalPGGParams(bool is_caching_test=false) {
    std::vector<ExperimentalDetectronPriorGridGeneratorParams<T>> experimentalPGGParams{
        {getValues<T>({-24.5, -12.5, 24.5, 12.5, -16.5, -16.5, 16.5, 16.5, -12.5, -24.5, 12.5, 24.5}),
         tensor(batch(3), feature(4)),
         0,
         0,
         4.0,
         4.0,
         true,
         {4, 5},
         {100, 200},
         tensor(1, 1, 4, 60),
         getValues<T>(
             {-22.5, -10.5, 26.5, 14.5, -14.5, -14.5, 18.5, 18.5, -10.5, -22.5, 14.5, 26.5, -18.5, -10.5, 30.5, 14.5,
              -10.5, -14.5, 22.5, 18.5, -6.5,  -22.5, 18.5, 26.5, -14.5, -10.5, 34.5, 14.5, -6.5,  -14.5, 26.5, 18.5,
              -2.5,  -22.5, 22.5, 26.5, -10.5, -10.5, 38.5, 14.5, -2.5,  -14.5, 30.5, 18.5, 1.5,   -22.5, 26.5, 26.5,
              -6.5,  -10.5, 42.5, 14.5, 1.5,   -14.5, 34.5, 18.5, 5.5,   -22.5, 30.5, 26.5, -22.5, -6.5,  26.5, 18.5,
              -14.5, -10.5, 18.5, 22.5, -10.5, -18.5, 14.5, 30.5, -18.5, -6.5,  30.5, 18.5, -10.5, -10.5, 22.5, 22.5,
              -6.5,  -18.5, 18.5, 30.5, -14.5, -6.5,  34.5, 18.5, -6.5,  -10.5, 26.5, 22.5, -2.5,  -18.5, 22.5, 30.5,
              -10.5, -6.5,  38.5, 18.5, -2.5,  -10.5, 30.5, 22.5, 1.5,   -18.5, 26.5, 30.5, -6.5,  -6.5,  42.5, 18.5,
              1.5,   -10.5, 34.5, 22.5, 5.5,   -18.5, 30.5, 30.5, -22.5, -2.5,  26.5, 22.5, -14.5, -6.5,  18.5, 26.5,
              -10.5, -14.5, 14.5, 34.5, -18.5, -2.5,  30.5, 22.5, -10.5, -6.5,  22.5, 26.5, -6.5,  -14.5, 18.5, 34.5,
              -14.5, -2.5,  34.5, 22.5, -6.5,  -6.5,  26.5, 26.5, -2.5,  -14.5, 22.5, 34.5, -10.5, -2.5,  38.5, 22.5,
              -2.5,  -6.5,  30.5, 26.5, 1.5,   -14.5, 26.5, 34.5, -6.5,  -2.5,  42.5, 22.5, 1.5,   -6.5,  34.5, 26.5,
              5.5,   -14.5, 30.5, 34.5, -22.5, 1.5,   26.5, 26.5, -14.5, -2.5,  18.5, 30.5, -10.5, -10.5, 14.5, 38.5,
              -18.5, 1.5,   30.5, 26.5, -10.5, -2.5,  22.5, 30.5, -6.5,  -10.5, 18.5, 38.5, -14.5, 1.5,   34.5, 26.5,
              -6.5,  -2.5,  26.5, 30.5, -2.5,  -10.5, 22.5, 38.5, -10.5, 1.5,   38.5, 26.5, -2.5,  -2.5,  30.5, 30.5,
              1.5,   -10.5, 26.5, 38.5, -6.5,  1.5,   42.5, 26.5, 1.5,   -2.5,  34.5, 30.5, 5.5,   -10.5, 30.5, 38.5}),
         is_caching_test},
        {getValues<T>({-44.5, -24.5, 44.5, 24.5, -32.5, -32.5, 32.5, 32.5, -24.5, -44.5, 24.5, 44.5}),
         tensor(batch(3), feature(4)),
         0,
         0,
         8.0,
         8.0,
         false,
         {3, 7},
         {100, 200},
         tensor(3, 7, 4, 3),
         getValues<T>(
             {-40.5, -20.5, 48.5, 28.5, -28.5, -28.5, 36.5, 36.5, -20.5, -40.5, 28.5, 48.5, -32.5, -20.5, 56.5, 28.5,
              -20.5, -28.5, 44.5, 36.5, -12.5, -40.5, 36.5, 48.5, -24.5, -20.5, 64.5, 28.5, -12.5, -28.5, 52.5, 36.5,
              -4.5,  -40.5, 44.5, 48.5, -16.5, -20.5, 72.5, 28.5, -4.5,  -28.5, 60.5, 36.5, 3.5,   -40.5, 52.5, 48.5,
              -8.5,  -20.5, 80.5, 28.5, 3.5,   -28.5, 68.5, 36.5, 11.5,  -40.5, 60.5, 48.5, -0.5,  -20.5, 88.5, 28.5,
              11.5,  -28.5, 76.5, 36.5, 19.5,  -40.5, 68.5, 48.5, 7.5,   -20.5, 96.5, 28.5, 19.5,  -28.5, 84.5, 36.5,
              27.5,  -40.5, 76.5, 48.5, -40.5, -12.5, 48.5, 36.5, -28.5, -20.5, 36.5, 44.5, -20.5, -32.5, 28.5, 56.5,
              -32.5, -12.5, 56.5, 36.5, -20.5, -20.5, 44.5, 44.5, -12.5, -32.5, 36.5, 56.5, -24.5, -12.5, 64.5, 36.5,
              -12.5, -20.5, 52.5, 44.5, -4.5,  -32.5, 44.5, 56.5, -16.5, -12.5, 72.5, 36.5, -4.5,  -20.5, 60.5, 44.5,
              3.5,   -32.5, 52.5, 56.5, -8.5,  -12.5, 80.5, 36.5, 3.5,   -20.5, 68.5, 44.5, 11.5,  -32.5, 60.5, 56.5,
              -0.5,  -12.5, 88.5, 36.5, 11.5,  -20.5, 76.5, 44.5, 19.5,  -32.5, 68.5, 56.5, 7.5,   -12.5, 96.5, 36.5,
              19.5,  -20.5, 84.5, 44.5, 27.5,  -32.5, 76.5, 56.5, -40.5, -4.5,  48.5, 44.5, -28.5, -12.5, 36.5, 52.5,
              -20.5, -24.5, 28.5, 64.5, -32.5, -4.5,  56.5, 44.5, -20.5, -12.5, 44.5, 52.5, -12.5, -24.5, 36.5, 64.5,
              -24.5, -4.5,  64.5, 44.5, -12.5, -12.5, 52.5, 52.5, -4.5,  -24.5, 44.5, 64.5, -16.5, -4.5,  72.5, 44.5,
              -4.5,  -12.5, 60.5, 52.5, 3.5,   -24.5, 52.5, 64.5, -8.5,  -4.5,  80.5, 44.5, 3.5,   -12.5, 68.5, 52.5,
              11.5,  -24.5, 60.5, 64.5, -0.5,  -4.5,  88.5, 44.5, 11.5,  -12.5, 76.5, 52.5, 19.5,  -24.5, 68.5, 64.5,
              7.5,   -4.5,  96.5, 44.5, 19.5,  -12.5, 84.5, 52.5, 27.5,  -24.5, 76.5, 64.5}),
         is_caching_test
        },
        {getValues<T>({-364.5, -184.5, 364.5, 184.5, -256.5, -256.5, 256.5, 256.5, -180.5, -360.5, 180.5, 360.5}),
         tensor(batch(3), feature(4)),
         3,
         6,
         64.0,
         64.0,
         true,
         {100, 100},
         {100, 200},
         tensor(1, 1, 4, 30000),
         getValues<T>({-332.5, -152.5, 396.5, 216.5, -224.5, -224.5, 288.5, 288.5, -148.5, -328.5, 212.5, 392.5,
                       -268.5, -152.5, 460.5, 216.5, -160.5, -224.5, 352.5, 288.5, -84.5,  -328.5, 276.5, 392.5,
                       -204.5, -152.5, 524.5, 216.5, -96.5,  -224.5, 416.5, 288.5, -20.5,  -328.5, 340.5, 392.5,
                       -140.5, -152.5, 588.5, 216.5, -32.5,  -224.5, 480.5, 288.5, 43.5,   -328.5, 404.5, 392.5,
                       -76.5,  -152.5, 652.5, 216.5, 31.5,   -224.5, 544.5, 288.5, 107.5,  -328.5, 468.5, 392.5,
                       -12.5,  -152.5, 716.5, 216.5, 95.5,   -224.5, 608.5, 288.5, 171.5,  -328.5, 532.5, 392.5,
                       -332.5, -88.5,  396.5, 280.5, -224.5, -160.5, 288.5, 352.5, -148.5, -264.5, 212.5, 456.5,
                       -268.5, -88.5,  460.5, 280.5, -160.5, -160.5, 352.5, 352.5, -84.5,  -264.5, 276.5, 456.5,
                       -204.5, -88.5,  524.5, 280.5, -96.5,  -160.5, 416.5, 352.5, -20.5,  -264.5, 340.5, 456.5,
                       -140.5, -88.5,  588.5, 280.5, -32.5,  -160.5, 480.5, 352.5, 43.5,   -264.5, 404.5, 456.5,
                       -76.5,  -88.5,  652.5, 280.5, 31.5,   -160.5, 544.5, 352.5, 107.5,  -264.5, 468.5, 456.5,
                       -12.5,  -88.5,  716.5, 280.5, 95.5,   -160.5, 608.5, 352.5, 171.5,  -264.5, 532.5, 456.5,
                       -332.5, -24.5,  396.5, 344.5, -224.5, -96.5,  288.5, 416.5, -148.5, -200.5, 212.5, 520.5,
                       -268.5, -24.5,  460.5, 344.5, -160.5, -96.5,  352.5, 416.5, -84.5,  -200.5, 276.5, 520.5,
                       -204.5, -24.5,  524.5, 344.5, -96.5,  -96.5,  416.5, 416.5, -20.5,  -200.5, 340.5, 520.5,
                       -140.5, -24.5,  588.5, 344.5, -32.5,  -96.5,  480.5, 416.5, 43.5,   -200.5, 404.5, 520.5,
                       -76.5,  -24.5,  652.5, 344.5, 31.5,   -96.5,  544.5, 416.5, 107.5,  -200.5, 468.5, 520.5,
                       -12.5,  -24.5,  716.5, 344.5, 95.5,   -96.5,  608.5, 416.5, 171.5,  -200.5, 532.5, 520.5}),
         is_caching_test},
        {getValues<T>({-180.5, -88.5, 180.5, 88.5, -128.5, -128.5, 128.5, 128.5, -92.5, -184.5, 92.5, 184.5}),
         tensor(batch(3), feature(4)),
         5,
         3,
         32.0,
         32.0,
         false,
         {100, 100},
         {100, 200},
         tensor(100, 100, 4, 3),
         getValues<T>({-164.5, -72.5, 196.5, 104.5, -112.5, -112.5, 144.5, 144.5, -76.5, -168.5, 108.5, 200.5,
                       -132.5, -72.5, 228.5, 104.5, -80.5,  -112.5, 176.5, 144.5, -44.5, -168.5, 140.5, 200.5,
                       -100.5, -72.5, 260.5, 104.5, -48.5,  -112.5, 208.5, 144.5, -12.5, -168.5, 172.5, 200.5,
                       -164.5, -40.5, 196.5, 136.5, -112.5, -80.5,  144.5, 176.5, -76.5, -136.5, 108.5, 232.5,
                       -132.5, -40.5, 228.5, 136.5, -80.5,  -80.5,  176.5, 176.5, -44.5, -136.5, 140.5, 232.5,
                       -100.5, -40.5, 260.5, 136.5, -48.5,  -80.5,  208.5, 176.5, -12.5, -136.5, 172.5, 232.5,
                       -164.5, -8.5,  196.5, 168.5, -112.5, -48.5,  144.5, 208.5, -76.5, -104.5, 108.5, 264.5,
                       -132.5, -8.5,  228.5, 168.5, -80.5,  -48.5,  176.5, 208.5, -44.5, -104.5, 140.5, 264.5,
                       -100.5, -8.5,  260.5, 168.5, -48.5,  -48.5,  208.5, 208.5, -12.5, -104.5, 172.5, 264.5,
                       -164.5, 23.5,  196.5, 200.5, -112.5, -16.5,  144.5, 240.5, -76.5, -72.5,  108.5, 296.5,
                       -132.5, 23.5,  228.5, 200.5, -80.5,  -16.5,  176.5, 240.5, -44.5, -72.5,  140.5, 296.5,
                       -100.5, 23.5,  260.5, 200.5, -48.5,  -16.5,  208.5, 240.5, -12.5, -72.5,  172.5, 296.5,
                       -164.5, 55.5,  196.5, 232.5, -112.5, 15.5,   144.5, 272.5, -76.5, -40.5,  108.5, 328.5,
                       -132.5, 55.5,  228.5, 232.5, -80.5,  15.5,   176.5, 272.5, -44.5, -40.5,  140.5, 328.5,
                       -100.5, 55.5,  260.5, 232.5, -48.5,  15.5,   208.5, 272.5, -12.5, -40.5,  172.5, 328.5}),
         is_caching_test}};
    return experimentalPGGParams;
}

struct PrintToStringParamName {
    template <class T>
    std::string operator()(const testing::TestParamInfo<ExperimentalDetectronPriorGridGeneratorParams<T>>& param) {
        std::stringstream buf;
        buf << " priors tensor " << param.param.priorsTensor.to_string() << " h " << param.param.h << " w "
            << param.param.w << " strideX " << param.param.strideX << " strideY " << param.param.strideY << " flatten "
            << param.param.flatten << " is_caching_test " << param.param.is_caching_test;
        return buf.str();
    }
};

using experimental_detectron_prior_grid_generator_test_f32 = experimental_detectron_prior_grid_generator_test<float>;
using experimental_detectron_prior_grid_generator_test_f16 = experimental_detectron_prior_grid_generator_test<ov::float16>;

TEST_P(experimental_detectron_prior_grid_generator_test_f32, experimental_detectron_prior_grid_generator_test_f32) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(experimental_detectron_prior_grid_generator_test_f16, experimental_detectron_prior_grid_generator_test_f16) {
    ASSERT_NO_FATAL_FAILURE(test());
}

INSTANTIATE_TEST_SUITE_P(smoke_experimental_detectron_prior_grid_generator_test_f32,
                         experimental_detectron_prior_grid_generator_test_f32,
                         ::testing::ValuesIn(generateExperimentalPGGParams<float>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_experimental_detectron_prior_grid_generator_test_f16,
                         experimental_detectron_prior_grid_generator_test_f16,
                         ::testing::ValuesIn(generateExperimentalPGGParams<ov::float16>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(export_import,
                         experimental_detectron_prior_grid_generator_test_f16,
                         ::testing::Values(generateExperimentalPGGParams<ov::float16>(true)[0]),
                         PrintToStringParamName());

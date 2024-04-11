// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <thread>
#include <type_traits>
#include <fstream>
#include <tuple>

#include "convolution_inst.h"

using namespace cldnn;
using namespace ::tests;

namespace {
const std::string no_bias = "";
template<typename T>
T kahan_summation(std::vector<T> &input) {
    T sum = 0;
    T c = 0;
    for (T x : input) {
        T y = x - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}
}  // namespace

template <typename InputT>
struct convolution_accumulator {
    using type = InputT;
};

template <>
struct convolution_accumulator<int8_t> {
    using type = int;
};

template <>
struct convolution_accumulator<uint8_t> {
    using type = int;
};

template<typename InputT, typename OutputT = InputT, typename WeightsT = InputT,  typename AccT = typename convolution_accumulator<InputT>::type>
VVVF<OutputT> reference_convolve(VVVVF<InputT> &input, VVVVF<WeightsT> &filter,
        int stride_z, int stride_y, int stride_x,
        float bias,
        int dilation_z = 1, int dilation_y = 1, int dilation_x = 1,
        int input_padding_z = 0, int input_padding_y = 0, int input_padding_x = 0,
        int output_padding_z = 0, int output_padding_y = 0, int output_padding_x = 0,
        size_t f_begin = 0, size_t f_end = 0, bool depthwise = false, bool grouped = false,
        const VF<InputT>& data_zp = {}, const WeightsT& weights_zp = 0)
{
    size_t kernel_extent_z = dilation_z * (filter[0].size() - 1) + 1;
    size_t kernel_extent_y = dilation_y * (filter[0][0].size() - 1) + 1;
    size_t kernel_extent_x = dilation_x * (filter[0][0][0].size() - 1) + 1;

    size_t output_z = 1 + (input[0].size() - kernel_extent_z + 2 * input_padding_z) / stride_z + 2 * output_padding_z;
    size_t output_y = 1 + (input[0][0].size() - kernel_extent_y + 2 * input_padding_y) / stride_y + 2 * output_padding_y;
    size_t output_x = 1 + (input[0][0][0].size() - kernel_extent_x + 2 * input_padding_x) / stride_x + 2 * output_padding_x;

    bool asymm_data = !data_zp.empty();
    bool asymm_weights = weights_zp != static_cast<WeightsT>(0);
    VVVF<OutputT> output(output_z, VVF<OutputT>(output_y, VF<OutputT>(output_x, 0)));
    size_t filter_begin = f_begin ? f_begin : 0;
    size_t filter_end = f_end ? f_end : filter.size();
    for (size_t f = filter_begin; f < filter_end; ++f) {
        for (size_t z = 0; z < (output_z - 2 * output_padding_z); ++z) {
            for (size_t y = 0; y < (output_y - 2 * output_padding_y); ++y) {
                for (size_t x = 0; x < (output_x - 2 * output_padding_x); ++x) {
                    VF<AccT> values;
                    values.reserve(filter[0].size() * filter[0][0].size() * filter[0][0][0].size());
                    for (size_t zf = 0; zf < filter[0].size(); ++zf) {
                        int zi = -input_padding_z + (int)zf * dilation_z + stride_z * (int)z;
                        bool zi_inside = zi >= 0 && (int)input[0].size() > zi;
                        if (!zi_inside) continue;
                        for (size_t yf = 0; yf < filter[0][0].size(); ++yf) {
                            int yi = -input_padding_y + (int)yf * dilation_y + stride_y * (int)y;
                            bool yi_inside = yi >= 0 && (int)input[0][0].size() > yi;
                            if (!yi_inside) continue;
                            for (size_t xf = 0; xf < filter[0][0][0].size(); ++xf) {
                                int xi = -input_padding_x + (int)xf * dilation_x + stride_x * (int)x;
                                bool xi_inside = xi >= 0 && (int)input[0][0][0].size() > xi;
                                if (!xi_inside) continue;

                                auto input_val = static_cast<AccT>(input[f][zi][yi][xi]);

                                if (asymm_data) {
                                    input_val = input_val - static_cast<AccT>(data_zp[f]);
                                }

                                AccT weights_val;
                                if (!depthwise && !grouped) {
                                    weights_val = static_cast<AccT>(filter[f][zf][yf][xf]);
                                } else if (grouped) {
                                    weights_val = static_cast<AccT>(filter[f - filter_begin][zf][yf][xf]);
                                } else {
                                    weights_val = static_cast<AccT>(filter[0][zf][yf][xf]);
                                }

                                if (asymm_weights) {
                                    weights_val = weights_val - static_cast<AccT>(weights_zp);
                                }

                                values.push_back(input_val * weights_val);
                            }
                        }
                    }
                    output[z + output_padding_z][y + output_padding_y][x + output_padding_x] += static_cast<OutputT>(kahan_summation<AccT>(values));
                }
            }
        }
    }

    for (size_t z = 0; z < (output_z - 2 * output_padding_z); ++z) {
        for (size_t y = 0; y < (output_y - 2 * output_padding_y); ++y) {
            for (size_t x = 0; x < (output_x - 2 * output_padding_x); ++x) {
                output[z + output_padding_z][y + output_padding_y][x + output_padding_x] += static_cast<OutputT>(bias);
            }
        }
    }
    return output;
}

template<typename InputT, typename OutputT = InputT, typename WeightsT = InputT,  typename AccT = typename convolution_accumulator<InputT>::type>
VVF<OutputT> reference_convolve(VVVF<InputT> &input, VVVF<WeightsT> &filter, int stride_y, int stride_x, float bias, int dilation_y = 1, int dilation_x = 1,
        int input_padding_y = 0, int input_padding_x = 0, int output_padding_y = 0,
        int output_padding_x = 0, size_t f_begin = 0, size_t f_end = 0, bool depthwise = false, bool grouped = false,
        const VF<InputT>& data_zp = {}, const WeightsT& weights_zp = 0)
{
    VVVVF<InputT> input_extended(input.size(), VVVF<InputT>(1, VVF<InputT>(input[0].size(), VF<InputT>(input[0][0].size(), 0))));
    for (size_t fi = 0; fi < input.size(); fi++) {
        for (size_t yi = 0; yi < input[0].size(); yi++) {
            for (size_t xi = 0; xi < input[0][0].size(); xi++) {
                input_extended[fi][0][yi][xi] = input[fi][yi][xi];
            }
        }
    }

    VVVVF<WeightsT> filter_extended(filter.size(), VVVF<WeightsT>(1, VVF<WeightsT>(filter[0].size(), VF<WeightsT>(filter[0][0].size(), 0))));
    for (size_t fi = 0; fi < filter.size(); fi++) {
        for (size_t yi = 0; yi < filter[0].size(); yi++) {
            for (size_t xi = 0; xi < filter[0][0].size(); xi++) {
                filter_extended[fi][0][yi][xi] = filter[fi][yi][xi];
            }
        }
    }

    VVVF<OutputT> output = reference_convolve<InputT, OutputT, WeightsT, AccT>(input_extended, filter_extended,
        1, stride_y, stride_x,
        bias,
        1, dilation_y, dilation_x,
        0, input_padding_y, input_padding_x,
        0, output_padding_y, output_padding_x,
        f_begin, f_end, depthwise, grouped,
        data_zp, weights_zp);

    VVF<OutputT> output_shrinked(output[0].size(), VF<OutputT>(output[0][0].size(), 0));

    for (size_t yi = 0; yi < output[0].size(); yi++) {
        for (size_t xi = 0; xi < output[0][0].size(); xi++) {
            output_shrinked[yi][xi] = output[0][yi][xi];
        }
    }

    return output_shrinked;
}

template <typename T>
VVVF<T> reference_scale_post_op(const VVVF<T>& input, const T& scale, const T& shift) {
    auto output = input;
    auto size_z = input.size();
    auto size_y = input[0].size();
    auto size_x = input[0][0].size();
    for (size_t zi = 0; zi < size_z; ++zi) {
        for (size_t yi = 0; yi < size_y; ++yi) {
            for (size_t xi = 0; xi < size_x; ++xi) {
                output[zi][yi][xi] = output[zi][yi][xi] * scale + shift;
            }
        }
    }
    return output;
}


template <typename T>
VVF<T> reference_scale_post_op(const VVF<T>& input, const T& scale, const T& shift) {
    VVVF<T> input_extended(1, VVF<T>(input.size(), VF<T>(input[0].size(), 0)));
    for (size_t yi = 0; yi < input.size(); yi++) {
        for (size_t xi = 0; xi < input[0].size(); xi++) {
            input_extended[0][yi][xi] = input[yi][xi];
        }
    }
    VVVF<T> output = reference_scale_post_op<T>(input_extended, scale, shift);
    VVF<T> output_shrinked(output[0].size(), VF<T>(output[0][0].size(), 0));
    for (size_t yi = 0; yi < output[0].size(); yi++) {
        for (size_t xi = 0; xi < output[0][0].size(); xi++) {
            output_shrinked[yi][xi] = output[0][yi][xi];
        }
    }

    return output_shrinked;
}

TEST(deformable_convolution_f32_fw_gpu, basic_deformable_convolution_def_group1_2) {
    //  Input    : 4x4x4
    //  Trans    : 18x4x4
    //  Output   : 4x4x4
    //  In_offset: 1x1
    //  Filter   : 3x3
    //  Stride   : 1x1
    //  Dilation : 1x1
    //  Group    : 1
    //  Def_group: 1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 4, 4 } });
    auto trans = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 18, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 4, 4, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 1, 1 } });

    set_values(input, { 0.680375f, -0.211234f, 0.566198f, 0.59688f, 0.823295f, -0.604897f, -0.329554f, 0.536459f,
                        -0.444451f, 0.10794f, -0.0452059f, 0.257742f, -0.270431f, 0.0268018f, 0.904459f, 0.83239f,
                        0.271423f, 0.434594f, -0.716795f, 0.213938f, -0.967399f, -0.514226f, -0.725537f, 0.608353f,
                        -0.686642f, -0.198111f, -0.740419f, -0.782382f, 0.997849f, -0.563486f, 0.0258648f, 0.678224f,
                        0.22528f, -0.407937f, 0.275105f, 0.0485743f, -0.012834f, 0.94555f, -0.414966f, 0.542715f,
                        0.0534899f, 0.539828f, -0.199543f, 0.783059f, -0.433371f, -0.295083f, 0.615449f, 0.838053f,
                        -0.860489f, 0.898654f, 0.0519907f, -0.827888f, -0.615572f, 0.326454f, 0.780465f, -0.302214f,
                        -0.871657f, -0.959954f, -0.0845965f, -0.873808f, -0.52344f, 0.941268f, 0.804416f, 0.70184f });

    set_values(trans, { -0.466668f, 0.0795207f, -0.249586f, 0.520497f, 0.0250708f, 0.335448f, 0.0632129f, -0.921439f, -0.124725f,
                        0.86367f, 0.86162f, 0.441905f, -0.431413f, 0.477069f, 0.279958f, -0.291903f, 0.375723f, -0.668052f,
                        -0.119791f, 0.76015f, 0.658402f, -0.339326f, -0.542064f, 0.786745f, -0.29928f, 0.37334f, 0.912936f,
                        0.17728f, 0.314608f, 0.717353f, -0.12088f, 0.84794f, -0.203127f, 0.629534f, 0.368437f, 0.821944f,
                        -0.0350187f, -0.56835f, 0.900505f, 0.840256f, -0.70468f, 0.762124f, 0.282161f, -0.136093f, 0.239193f,
                        -0.437881f, 0.572004f, -0.385084f, -0.105933f, -0.547787f, -0.624934f, -0.447531f, 0.112888f, -0.166997f,
                        -0.660786f, 0.813608f, -0.793658f, -0.747849f, -0.00911188f, 0.52095f, 0.969503f, 0.870008f, 0.36889f,
                        -0.233623f, 0.499542f, -0.262673f, -0.411679f, -0.535477f, 0.168977f, -0.511175f, -0.69522f, 0.464297f,
                        -0.74905f, 0.586941f, -0.671796f, 0.490143f, -0.85094f, 0.900208f, -0.894941f, 0.0431267f, -0.647579f,
                        -0.519875f, 0.595596f, 0.465309f, 0.313127f, 0.93481f, 0.278917f, 0.51947f, -0.813039f, -0.730195f,
                        0.0404202f, -0.843536f, -0.860187f, -0.59069f, -0.077159f, 0.639355f, 0.146637f, 0.511162f, -0.896122f,
                        -0.684386f, 0.999987f, -0.591343f, 0.779911f, -0.749063f, 0.995598f, -0.891885f, 0.74108f, -0.855342f,
                        -0.991677f, 0.846138f, 0.187784f, -0.639255f, -0.673737f, -0.21662f, 0.826053f, 0.63939f, -0.281809f,
                        0.10497f, 0.15886f, -0.0948483f, 0.374775f, -0.80072f, 0.0616159f, 0.514588f, -0.39141f, 0.984457f,
                        0.153942f, 0.755228f, 0.495619f, 0.25782f, -0.929158f, 0.495606f, 0.666477f, 0.850753f, 0.746543f,
                        0.662075f, 0.958868f, 0.487622f, 0.806733f, 0.967191f, 0.333761f, -0.00548297f, -0.672064f, 0.660024f,
                        0.777897f, -0.846011f, 0.299414f, -0.503912f, 0.258959f, -0.541726f, 0.40124f, -0.366266f, -0.342446f,
                        -0.537144f, -0.851678f, 0.266144f, -0.552687f, 0.302264f, 0.021372f, 0.942931f, -0.439916f, 0.0922137f,
                        0.438537f, -0.773439f, -0.0570331f, 0.18508f, 0.888636f, -0.0981649f, -0.327298f, 0.695369f, -0.130973f,
                        -0.993537f, -0.310114f, 0.196963f, 0.666487f, -0.532217f, 0.350952f, -0.0340995f, -0.0361283f, -0.390089f,
                        0.424175f, -0.634888f, 0.243646f, -0.918271f, -0.172033f, 0.391968f, 0.347873f, 0.27528f, -0.305768f,
                        -0.630755f, 0.218212f, 0.254316f, 0.461459f, -0.343251f, 0.480877f, -0.595574f, 0.841829f, 0.369513f,
                        0.306261f, -0.485469f, 0.0648819f, -0.824713f, -0.479006f, 0.754768f, 0.37225f, -0.81252f, -0.777449f,
                        -0.276798f, 0.153381f, 0.186423f, 0.333113f, -0.422444f, 0.551535f, -0.423241f, -0.340716f, -0.620498f,
                        0.968726f, -0.992843f, 0.654782f, -0.337042f, -0.623598f, -0.127006f, 0.917274f, 0.837861f, 0.529743f,
                        0.398151f, -0.757714f, 0.371572f, -0.232336f, 0.548547f, 0.886103f, 0.832546f, 0.723834f, -0.592904f,
                        0.587314f, 0.0960841f, -0.405423f, 0.809865f, 0.819286f, 0.747958f, -0.00371218f, 0.152399f, -0.674487f,
                        -0.452178f, 0.729158f, -0.0152023f, -0.0726757f, 0.697884f, -0.0080452f, -0.417893f, -0.639158f, 0.368357f,
                        0.455101f, -0.721884f, 0.206218f, -0.0151566f, 0.676267f, 0.448504f, -0.643585f, -0.556069f, -0.00294906f,
                        -0.757482f, -0.723523f, -0.279115f, -0.350386f, 0.863791f, 0.816969f, 0.244191f, 0.673656f, 0.636255f,
                        -0.00785118f, -0.330057f, -0.211346f, 0.317662f, 0.217766f, -0.482188f, -0.69754f, -0.85491f, -0.784303f,
                        0.294415f, -0.272803f, -0.423461f, -0.337228f, -0.817703f, -0.145345f, 0.868989f, 0.167141f, -0.469077f });

    set_values(weights, { 0.0f, 0.841471f, 0.909297f, 0.14112f, -0.756802f, -0.958924f, -0.279415f, 0.656987f, 0.989358f,
                          0.412118f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.912945f, 0.836656f,
                          -0.00885131f, -0.84622f, -0.905578f, -0.132352f, 0.762558f, 0.956376f, 0.270906f, -0.663634f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.745113f, -0.158623f, -0.916522f,
                          -0.831775f, 0.0177019f, 0.850904f, 0.901788f, 0.123573f, -0.768255f, -0.953753f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.304811f, -0.966118f, -0.739181f, 0.167356f,
                          0.920026f, 0.826829f, -0.0265512f, -0.85552f, -0.897928f, -0.114785f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.993889f, -0.629888f, 0.313229f, 0.968364f, 0.73319f,
                          -0.176076f, -0.923458f, -0.821818f, 0.0353983f, 0.860069f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, -0.506366f, 0.452026f, 0.994827f, 0.622989f, -0.321622f, -0.970535f,
                          -0.727143f, 0.184782f, 0.926818f, 0.816743f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.580611f, 0.998815f, 0.498713f, -0.459903f, -0.995687f, -0.61604f, 0.329991f,
                          0.97263f, 0.721038f, -0.193473f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.98024f, 0.363171f, -0.587795f, -0.998345f });

    set_values(biases, { -0.491022f, 0.467745f, 0.996469f, 0.609044f });

    std::vector<float> output_vec = {
            -0.0742483f, -2.09984f, -0.850381f, 0.0398813f, -1.06922f, -0.0233979f, -1.15264f, -0.970688f, -0.0428347f,
            -1.73668f, 0.613717f, 2.12469f, 0.450835f, -1.82602f, -0.57416f, -0.682909f, -0.211437f, 0.351543f,
            0.930845f, 0.412505f, 1.23318f, 0.894126f, 1.56587f, 0.882479f, 0.640997f, -1.94229f, -1.00846f, -3.64185f,
            -0.383791f, -0.274041f, -1.49479f, -2.82027f, 0.858848f, 1.7228f, 0.51184f, 0.537693f, 1.40331f, 0.192823f,
            0.325383f, 0.814044f, 1.19015f, 0.403436f, 1.40995f, 0.42931f, 0.131369f, 2.01262f, 0.253117f, 0.018361f,
            1.3469f, 1.15957f, 0.599044f, 1.48224f, 2.16468f, 0.504246f, -1.52044f, 0.10271f, 0.0379517f, 0.942841f,
            -2.6067f, 0.562893f, 0.671884f, 0.404735f, 1.45044f, 0.950113f };

    topology topology(
            input_layout("input", input->get_layout()),
            input_layout("trans", trans->get_layout()),
            data("weights", weights),
            data("biases", biases),
            convolution(
                    "conv",
                    { input_info("input"), input_info("trans") },
                    "weights",
                    "biases",
                    true,
                    1,
                    1,
                    { 1, 1 },
                    { 1, 1 },
                    { 1, 1 },
                    { 1, 1 })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    network.set_input_data("trans", trans);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 4);
    ASSERT_EQ(f_size, 4);
    ASSERT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        ASSERT_NEAR(output_vec[i], output_ptr[i], 0.1);
    }
}

TEST(deformable_convolution_f32_fw_gpu, basic_deformable_convolution_def_group1) {
    //  Input    : 4x4x4
    //  Trans    : 18x4x4
    //  Output   : 4x4x4
    //  In_offset: 2x2
    //  Filter   : 3x3
    //  Stride   : 1x1
    //  Dilation : 2x2
    //  Group    : 1
    //  Def_group: 1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 4, 4 } });
    auto trans = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 18, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 4, 4, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 1, 1 } });

    set_values(input, { 0.680375f, -0.211234f, 0.566198f, 0.59688f, 0.823295f, -0.604897f, -0.329554f, 0.536459f,
                        -0.444451f, 0.10794f, -0.0452059f, 0.257742f, -0.270431f, 0.0268018f, 0.904459f, 0.83239f,
                        0.271423f, 0.434594f, -0.716795f, 0.213938f, -0.967399f, -0.514226f, -0.725537f, 0.608353f,
                        -0.686642f, -0.198111f, -0.740419f, -0.782382f, 0.997849f, -0.563486f, 0.0258648f, 0.678224f,
                        0.22528f, -0.407937f, 0.275105f, 0.0485743f, -0.012834f, 0.94555f, -0.414966f, 0.542715f,
                        0.0534899f, 0.539828f, -0.199543f, 0.783059f, -0.433371f, -0.295083f, 0.615449f, 0.838053f,
                        -0.860489f, 0.898654f, 0.0519907f, -0.827888f, -0.615572f, 0.326454f, 0.780465f, -0.302214f,
                        -0.871657f, -0.959954f, -0.0845965f, -0.873808f, -0.52344f, 0.941268f, 0.804416f, 0.70184f });

    set_values(trans, { -0.466668f, 0.0795207f, -0.249586f, 0.520497f, 0.0250708f, 0.335448f, 0.0632129f, -0.921439f, -0.124725f,
                        0.86367f, 0.86162f, 0.441905f, -0.431413f, 0.477069f, 0.279958f, -0.291903f, 0.375723f, -0.668052f,
                        -0.119791f, 0.76015f, 0.658402f, -0.339326f, -0.542064f, 0.786745f, -0.29928f, 0.37334f, 0.912936f,
                        0.17728f, 0.314608f, 0.717353f, -0.12088f, 0.84794f, -0.203127f, 0.629534f, 0.368437f, 0.821944f,
                        -0.0350187f, -0.56835f, 0.900505f, 0.840256f, -0.70468f, 0.762124f, 0.282161f, -0.136093f, 0.239193f,
                        -0.437881f, 0.572004f, -0.385084f, -0.105933f, -0.547787f, -0.624934f, -0.447531f, 0.112888f, -0.166997f,
                        -0.660786f, 0.813608f, -0.793658f, -0.747849f, -0.00911188f, 0.52095f, 0.969503f, 0.870008f, 0.36889f,
                        -0.233623f, 0.499542f, -0.262673f, -0.411679f, -0.535477f, 0.168977f, -0.511175f, -0.69522f, 0.464297f,
                        -0.74905f, 0.586941f, -0.671796f, 0.490143f, -0.85094f, 0.900208f, -0.894941f, 0.0431267f, -0.647579f,
                        -0.519875f, 0.595596f, 0.465309f, 0.313127f, 0.93481f, 0.278917f, 0.51947f, -0.813039f, -0.730195f,
                        0.0404202f, -0.843536f, -0.860187f, -0.59069f, -0.077159f, 0.639355f, 0.146637f, 0.511162f, -0.896122f,
                        -0.684386f, 0.999987f, -0.591343f, 0.779911f, -0.749063f, 0.995598f, -0.891885f, 0.74108f, -0.855342f,
                        -0.991677f, 0.846138f, 0.187784f, -0.639255f, -0.673737f, -0.21662f, 0.826053f, 0.63939f, -0.281809f,
                        0.10497f, 0.15886f, -0.0948483f, 0.374775f, -0.80072f, 0.0616159f, 0.514588f, -0.39141f, 0.984457f,
                        0.153942f, 0.755228f, 0.495619f, 0.25782f, -0.929158f, 0.495606f, 0.666477f, 0.850753f, 0.746543f,
                        0.662075f, 0.958868f, 0.487622f, 0.806733f, 0.967191f, 0.333761f, -0.00548297f, -0.672064f, 0.660024f,
                        0.777897f, -0.846011f, 0.299414f, -0.503912f, 0.258959f, -0.541726f, 0.40124f, -0.366266f, -0.342446f,
                        -0.537144f, -0.851678f, 0.266144f, -0.552687f, 0.302264f, 0.021372f, 0.942931f, -0.439916f, 0.0922137f,
                        0.438537f, -0.773439f, -0.0570331f, 0.18508f, 0.888636f, -0.0981649f, -0.327298f, 0.695369f, -0.130973f,
                        -0.993537f, -0.310114f, 0.196963f, 0.666487f, -0.532217f, 0.350952f, -0.0340995f, -0.0361283f, -0.390089f,
                        0.424175f, -0.634888f, 0.243646f, -0.918271f, -0.172033f, 0.391968f, 0.347873f, 0.27528f, -0.305768f,
                        -0.630755f, 0.218212f, 0.254316f, 0.461459f, -0.343251f, 0.480877f, -0.595574f, 0.841829f, 0.369513f,
                        0.306261f, -0.485469f, 0.0648819f, -0.824713f, -0.479006f, 0.754768f, 0.37225f, -0.81252f, -0.777449f,
                        -0.276798f, 0.153381f, 0.186423f, 0.333113f, -0.422444f, 0.551535f, -0.423241f, -0.340716f, -0.620498f,
                        0.968726f, -0.992843f, 0.654782f, -0.337042f, -0.623598f, -0.127006f, 0.917274f, 0.837861f, 0.529743f,
                        0.398151f, -0.757714f, 0.371572f, -0.232336f, 0.548547f, 0.886103f, 0.832546f, 0.723834f, -0.592904f,
                        0.587314f, 0.0960841f, -0.405423f, 0.809865f, 0.819286f, 0.747958f, -0.00371218f, 0.152399f, -0.674487f,
                        -0.452178f, 0.729158f, -0.0152023f, -0.0726757f, 0.697884f, -0.0080452f, -0.417893f, -0.639158f, 0.368357f,
                        0.455101f, -0.721884f, 0.206218f, -0.0151566f, 0.676267f, 0.448504f, -0.643585f, -0.556069f, -0.00294906f,
                        -0.757482f, -0.723523f, -0.279115f, -0.350386f, 0.863791f, 0.816969f, 0.244191f, 0.673656f, 0.636255f,
                        -0.00785118f, -0.330057f, -0.211346f, 0.317662f, 0.217766f, -0.482188f, -0.69754f, -0.85491f, -0.784303f,
                        0.294415f, -0.272803f, -0.423461f, -0.337228f, -0.817703f, -0.145345f, 0.868989f, 0.167141f, -0.469077f });

    set_values(weights, { 0.0f, 0.841471f, 0.909297f, 0.14112f, -0.756802f, -0.958924f, -0.279415f, 0.656987f,
                          0.989358f, 0.412118f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.912945f,
                          0.836656f, -0.00885131f, -0.84622f, -0.905578f, -0.132352f, 0.762558f, 0.956376f, 0.270906f,
                          -0.663634f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.745113f, -0.158623f,
                          -0.916522f, -0.831775f, 0.0177019f, 0.850904f, 0.901788f, 0.123573f, -0.768255f, -0.953753f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.304811f, -0.966118f, -0.739181f,
                          0.167356f, 0.920026f, 0.826829f, -0.0265512f, -0.85552f, -0.897928f, -0.114785f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.993889f, -0.629888f, 0.313229f, 0.968364f,
                          0.73319f, -0.176076f, -0.923458f, -0.821818f, 0.0353983f, 0.860069f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.506366f, 0.452026f, 0.994827f, 0.622989f, -0.321622f,
                          -0.970535f, -0.727143f, 0.184782f, 0.926818f, 0.816743f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.580611f, 0.998815f, 0.498713f, -0.459903f, -0.995687f, -0.61604f,
                          0.329991f, 0.97263f, 0.721038f, -0.193473f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.98024f, 0.363171f, -0.587795f, -0.998345f });

    set_values(biases, { -0.491022f, 0.467745f, 0.996469f, 0.609044f });

    std::vector<float> output_vec = {
            0.304297f, -0.385894f, -1.44155f, -0.954556f, -0.260702f, -0.0162079f, 1.05196f, -0.129013f, 0.668587f,
            -1.4058f, -0.0966965f, -0.45043f, -2.23299f, -1.56306f, 0.083207f, -0.42985f, -0.00589353f, 1.08037f,
            1.06648f, 0.0936709f, 1.62321f, 1.50433f, 0.00480294f, -0.0550415f, 0.165425f, 0.146279f, -0.45487f,
            0.370202f, 0.177222f, -1.03958f, -0.744073f, -0.375273f, 0.587801f, 0.120338f, 1.17536f, 1.88443f,
            0.119988f, -0.540461f, -1.7228f, 1.54217f, 0.962263f, -0.0363407f, 0.762274f, 1.32504f, 1.43954f,
            -0.143791f, 1.21981f, 1.71111f, 0.195772f, 0.650412f, 0.474924f, 0.929919f, -0.442715f, 0.462916f,
            -0.210789f, -0.973089f, -0.407542f, 1.11818f, 0.843776f, 0.628229f, 1.29095f, 1.18637f, 0.808982f, 1.43841f };

    topology topology(
            input_layout("input", input->get_layout()),
            input_layout("trans", trans->get_layout()),
            data("weights", weights),
            data("biases", biases),
            convolution(
                    "conv",
                    { input_info("input"), input_info("trans") },
                    "weights",
                    "biases",
                    true,
                    1,
                    1,
                    { 1, 1 },
                    { 2, 2 },
                    { 2, 2 },
                    { 2, 2 })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    network.set_input_data("trans", trans);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 4);
    ASSERT_EQ(f_size, 4);
    ASSERT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        ASSERT_NEAR(output_vec[i], output_ptr[i], 0.1);
    }
}

TEST(deformable_convolution_f32_fw_gpu, basic_deformable_convolution) {
    //  Input    : 4x4x4
    //  Trans    : 36x4x4
    //  Output   : 4x4x4
    //  In_offset: 2x2
    //  Filter   : 3x3
    //  Stride   : 1x1
    //  Dilation : 2x2
    //  Group    : 1
    //  Def_group: 2

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 4, 4 } });
    auto trans = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 36, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 4, 4, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 1, 1 } });

    set_values(input, { 0.680375f, -0.211234f, 0.566198f, 0.59688f, 0.823295f, -0.604897f, -0.329554f, 0.536459f,
                        -0.444451f, 0.10794f, -0.0452059f, 0.257742f, -0.270431f, 0.0268018f, 0.904459f, 0.83239f,
                        0.271423f, 0.434594f, -0.716795f, 0.213938f, -0.967399f, -0.514226f, -0.725537f, 0.608353f,
                        -0.686642f, -0.198111f, -0.740419f, -0.782382f, 0.997849f, -0.563486f, 0.0258648f, 0.678224f,
                        0.22528f, -0.407937f, 0.275105f, 0.0485743f, -0.012834f, 0.94555f, -0.414966f, 0.542715f,
                        0.0534899f, 0.539828f, -0.199543f, 0.783059f, -0.433371f, -0.295083f, 0.615449f, 0.838053f,
                        -0.860489f, 0.898654f, 0.0519907f, -0.827888f, -0.615572f, 0.326454f, 0.780465f, -0.302214f,
                        -0.871657f, -0.959954f, -0.0845965f, -0.873808f, -0.52344f, 0.941268f, 0.804416f, 0.70184f });

    set_values(trans, { -0.466668f, 0.0795207f, -0.249586f, 0.520497f, 0.0250708f, 0.335448f, 0.0632129f, -0.921439f, -0.124725f,
                        0.86367f, 0.86162f, 0.441905f, -0.431413f, 0.477069f, 0.279958f, -0.291903f, 0.375723f, -0.668052f,
                        -0.119791f, 0.76015f, 0.658402f, -0.339326f, -0.542064f, 0.786745f, -0.29928f, 0.37334f, 0.912936f,
                        0.17728f, 0.314608f, 0.717353f, -0.12088f, 0.84794f, -0.203127f, 0.629534f, 0.368437f, 0.821944f,
                        -0.0350187f, -0.56835f, 0.900505f, 0.840256f, -0.70468f, 0.762124f, 0.282161f, -0.136093f, 0.239193f,
                        -0.437881f, 0.572004f, -0.385084f, -0.105933f, -0.547787f, -0.624934f, -0.447531f, 0.112888f, -0.166997f,
                        -0.660786f, 0.813608f, -0.793658f, -0.747849f, -0.00911188f, 0.52095f, 0.969503f, 0.870008f, 0.36889f,
                        -0.233623f, 0.499542f, -0.262673f, -0.411679f, -0.535477f, 0.168977f, -0.511175f, -0.69522f, 0.464297f,
                        -0.74905f, 0.586941f, -0.671796f, 0.490143f, -0.85094f, 0.900208f, -0.894941f, 0.0431267f, -0.647579f,
                        -0.519875f, 0.595596f, 0.465309f, 0.313127f, 0.93481f, 0.278917f, 0.51947f, -0.813039f, -0.730195f,
                        0.0404202f, -0.843536f, -0.860187f, -0.59069f, -0.077159f, 0.639355f, 0.146637f, 0.511162f, -0.896122f,
                        -0.684386f, 0.999987f, -0.591343f, 0.779911f, -0.749063f, 0.995598f, -0.891885f, 0.74108f, -0.855342f,
                        -0.991677f, 0.846138f, 0.187784f, -0.639255f, -0.673737f, -0.21662f, 0.826053f, 0.63939f, -0.281809f,
                        0.10497f, 0.15886f, -0.0948483f, 0.374775f, -0.80072f, 0.0616159f, 0.514588f, -0.39141f, 0.984457f,
                        0.153942f, 0.755228f, 0.495619f, 0.25782f, -0.929158f, 0.495606f, 0.666477f, 0.850753f, 0.746543f,
                        0.662075f, 0.958868f, 0.487622f, 0.806733f, 0.967191f, 0.333761f, -0.00548297f, -0.672064f, 0.660024f,
                        0.777897f, -0.846011f, 0.299414f, -0.503912f, 0.258959f, -0.541726f, 0.40124f, -0.366266f, -0.342446f,
                        -0.537144f, -0.851678f, 0.266144f, -0.552687f, 0.302264f, 0.021372f, 0.942931f, -0.439916f, 0.0922137f,
                        0.438537f, -0.773439f, -0.0570331f, 0.18508f, 0.888636f, -0.0981649f, -0.327298f, 0.695369f, -0.130973f,
                        -0.993537f, -0.310114f, 0.196963f, 0.666487f, -0.532217f, 0.350952f, -0.0340995f, -0.0361283f, -0.390089f,
                        0.424175f, -0.634888f, 0.243646f, -0.918271f, -0.172033f, 0.391968f, 0.347873f, 0.27528f, -0.305768f,
                        -0.630755f, 0.218212f, 0.254316f, 0.461459f, -0.343251f, 0.480877f, -0.595574f, 0.841829f, 0.369513f,
                        0.306261f, -0.485469f, 0.0648819f, -0.824713f, -0.479006f, 0.754768f, 0.37225f, -0.81252f, -0.777449f,
                        -0.276798f, 0.153381f, 0.186423f, 0.333113f, -0.422444f, 0.551535f, -0.423241f, -0.340716f, -0.620498f,
                        0.968726f, -0.992843f, 0.654782f, -0.337042f, -0.623598f, -0.127006f, 0.917274f, 0.837861f, 0.529743f,
                        0.398151f, -0.757714f, 0.371572f, -0.232336f, 0.548547f, 0.886103f, 0.832546f, 0.723834f, -0.592904f,
                        0.587314f, 0.0960841f, -0.405423f, 0.809865f, 0.819286f, 0.747958f, -0.00371218f, 0.152399f, -0.674487f,
                        -0.452178f, 0.729158f, -0.0152023f, -0.0726757f, 0.697884f, -0.0080452f, -0.417893f, -0.639158f, 0.368357f,
                        0.455101f, -0.721884f, 0.206218f, -0.0151566f, 0.676267f, 0.448504f, -0.643585f, -0.556069f, -0.00294906f,
                        -0.757482f, -0.723523f, -0.279115f, -0.350386f, 0.863791f, 0.816969f, 0.244191f, 0.673656f, 0.636255f,
                        -0.00785118f, -0.330057f, -0.211346f, 0.317662f, 0.217766f, -0.482188f, -0.69754f, -0.85491f, -0.784303f,
                        0.294415f, -0.272803f, -0.423461f, -0.337228f, -0.817703f, -0.145345f, 0.868989f, 0.167141f, -0.469077f,
                        0.317493f, 0.523556f, -0.0251462f, -0.685456f, 0.766073f, 0.251331f, 0.0354295f, -0.584313f, 0.115121f,
                        -0.147601f, 0.659878f, -0.211223f, -0.511346f, -0.347973f, 0.45872f, 0.277308f, 0.969689f, -0.323514f,
                        0.79512f, -0.727851f, -0.178424f, -0.989183f, 0.566564f, 0.548772f, -0.412644f, -0.770664f, 0.73107f,
                        0.442012f, -0.901675f, -0.10179f, 0.972934f, 0.415818f, -0.578234f, -0.0522121f, 0.730362f, -0.812161f,
                        -0.800881f, -0.234208f, -0.396474f, 0.31424f, 0.618191f, -0.736596f, -0.896983f, -0.893155f, -0.0845688f,
                        0.561737f, 0.384153f, -0.11488f, -0.761777f, 0.179273f, 0.15727f, 0.0597985f, 0.190091f, -0.276166f,
                        -0.391429f, 0.777447f, -0.0468307f, -0.66036f, 0.219458f, 0.0514942f, 0.237851f, 0.192392f, -0.532688f,
                        0.659617f, -0.85982f, -0.802325f, 0.847456f, -0.660701f, -0.0365333f, -0.549018f, 0.653539f, -0.418343f,
                        -0.285614f, 0.756555f, -0.311498f, 0.629817f, 0.318292f, -0.927345f, -0.485062f, 0.556515f, 0.251928f,
                        0.672207f, -0.383687f, -0.557981f, -0.603959f, 0.224884f, -0.780535f, 0.349211f, 0.564525f, 0.438924f,
                        -0.599295f, -0.197625f, -0.368684f, -0.131983f, -0.538008f, -0.228504f, 0.0656919f, -0.690552f, 0.110795f,
                        -0.970841f, -0.239571f, -0.235666f, -0.389184f, 0.474815f, -0.47911f, 0.299318f, 0.104633f, 0.839182f,
                        0.371973f, 0.619571f, 0.395696f, -0.376099f, 0.291778f, -0.98799f, 0.0659196f, 0.687819f, 0.236894f,
                        0.285385f, 0.0370297f, -0.198582f, -0.275691f, 0.437734f, 0.603793f, 0.355625f, -0.694248f, -0.934215f,
                        -0.872879f, 0.371444f, -0.624767f, 0.237917f, 0.400602f, 0.135662f, -0.997749f, -0.988582f, -0.389522f,
                        -0.476859f, 0.310736f, 0.71511f, -0.637678f, -0.317291f, 0.334681f, 0.758019f, 0.30661f, -0.373541f,
                        0.770028f, -0.62747f, -0.685722f, 0.00692201f, 0.657915f, 0.351308f, 0.80834f, -0.617777f, -0.210957f,
                        0.412133f, 0.737848f, 0.0947942f, 0.477919f, 0.864969f, -0.533762f, 0.853152f, 0.102886f, 0.86684f,
                        -0.0111862f, 0.105137f, 0.878258f, 0.599291f, 0.628277f, 0.188995f, 0.314402f, 0.9906f, 0.871704f,
                        -0.350917f, 0.748619f, 0.178313f, 0.275542f, 0.518647f, 0.550843f, 0.58982f, -0.474431f, 0.208758f,
                        -0.0588716f, -0.666091f, 0.590981f, 0.730171f, 0.746043f, 0.328829f, -0.175035f, 0.223961f, 0.193798f,
                        0.291203f, 0.077113f, -0.703316f, 0.158043f, -0.934073f, 0.401821f, 0.0363014f, 0.665218f, 0.0300982f,
                        -0.774704f, -0.02038f, 0.0206981f, -0.903001f, 0.628703f, -0.230683f, 0.275313f, -0.0957552f, -0.712036f,
                        -0.173844f, -0.505935f, -0.186467f, -0.965087f, 0.435194f, 0.147442f, 0.625894f, 0.165365f, -0.106515f,
                        -0.0452772f, 0.99033f, -0.882554f, -0.851479f, 0.281533f, 0.19456f, -0.554795f, -0.560424f, 0.260486f,
                        0.847025f, 0.475877f, -0.0742955f, -0.122876f, 0.701173f, 0.905324f, 0.897822f, 0.798172f, 0.534027f,
                        -0.332862f, 0.073485f, -0.561728f, -0.0448976f, 0.899641f, -0.0676628f, 0.768636f, 0.934554f, -0.632469f,
                        -0.083922f, 0.560448f, 0.532895f, 0.809563f, -0.484829f, 0.523225f, 0.927009f, -0.336308f, -0.195242f,
                        0.121569f, 0.108896f, 0.244333f, -0.617945f, -0.0440782f, -0.27979f, 0.30776f, 0.833045f, -0.578617f,
                        0.213084f, 0.730867f, -0.780445f, -0.252888f, -0.601995f, 0.29304f, 0.185384f, 0.353108f, 0.192681f,
                        -0.882279f, 0.121744f, 0.127235f, -0.514748f, -0.962178f, -0.312317f, -0.981853f, 0.847385f, 0.202853f,
                        0.541372f, 0.774394f, 0.866545f, -0.653871f, -0.104037f, -0.0245586f, 0.590463f, 0.278018f, 0.931363f });

    set_values(weights, { 0.0f, 0.841471f, 0.909297f, 0.14112f, -0.756802f, -0.958924f, -0.279415f, 0.656987f,
                          0.989358f, 0.412118f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.912945f,
                          0.836656f, -0.00885131f, -0.84622f, -0.905578f, -0.132352f, 0.762558f, 0.956376f, 0.270906f,
                          -0.663634f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.745113f, -0.158623f,
                          -0.916522f, -0.831775f, 0.0177019f, 0.850904f, 0.901788f, 0.123573f, -0.768255f, -0.953753f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.304811f, -0.966118f, -0.739181f,
                          0.167356f, 0.920026f, 0.826829f, -0.0265512f, -0.85552f, -0.897928f, -0.114785f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.993889f, -0.629888f, 0.313229f, 0.968364f,
                          0.73319f, -0.176076f, -0.923458f, -0.821818f, 0.0353983f, 0.860069f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.506366f, 0.452026f, 0.994827f, 0.622989f, -0.321622f,
                          -0.970535f, -0.727143f, 0.184782f, 0.926818f, 0.816743f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.580611f, 0.998815f, 0.498713f, -0.459903f, -0.995687f, -0.61604f,
                          0.329991f, 0.97263f, 0.721038f, -0.193473f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.98024f, 0.363171f, -0.587795f, -0.998345f });

    set_values(biases, { -0.491022f, 0.467745f, 0.996469f, 0.609044f });

    std::vector<float> output_vec = {
            -0.119782f, -1.19244f, -0.495079f, -0.760084f, 0.066627f, 0.42253f, 0.365703f, 0.62729f, 0.380708f,
            0.0900432f, -0.866731f, -0.784469f, -2.18692f, -1.73267f, 0.251761f, -0.791547f, 0.634862f, 0.646589f,
            -0.321454f, 0.575675f, 0.98983f, -0.445829f, 0.523965f, -0.346374f, 0.127803f, -2.13572f, -0.796409f,
            1.5734f, -0.972705f, -1.88344f, -1.04588f, -0.0209212f, 0.78641f, -0.3878f, 0.151791f, 2.08673f, 0.698802f,
            0.584678f, -0.78199f, -0.352576f, 1.03862f, 0.229792f, -0.223219f, 1.02365f, 1.45293f, 0.0561579f, 1.95182f,
            1.59586f, 0.773778f, 0.648415f, 1.65464f, 1.311f, 0.326254f, -0.447391f, -0.858153f, -0.702836f, -0.589441f,
            1.18929f, 0.382556f, 0.499048f, 1.16212f, 1.62688f, 1.31246f, 1.82684f };

    topology topology(
            input_layout("input", input->get_layout()),
            input_layout("trans", trans->get_layout()),
            data("weights", weights),
            data("biases", biases),
            convolution(
                    "conv",
                    { input_info("input"), input_info("trans") },
                    "weights",
                    "biases",
                    true,
                    1,
                    2,
                    { 1, 1 },
                    { 2, 2 },
                    { 2, 2 },
                    { 2, 2 })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    network.set_input_data("trans", trans);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 4);
    ASSERT_EQ(f_size, 4);
    ASSERT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        ASSERT_NEAR(output_vec[i], output_ptr[i], 0.1);
    }
}

TEST(deformable_convolution_f32_fw_gpu, basic_deformable_convolution_group2) {
    //  Input    : 4x4x4
    //  Trans    : 36x4x4
    //  Output   : 4x4x4
    //  In_offset: 2x2
    //  Filter   : 3x3
    //  Stride   : 1x1
    //  Dilation : 2x2
    //  Group    : 2
    //  Def_group: 1

    auto& engine = get_test_engine();
    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::allow_static_input_reorder(true));

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 4, 4 } });
    auto trans = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 18, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 6, 2, 3, 3 } });

    set_values(input, { 0.041230f, 0.571591f, 0.214803f, 0.920063f, 0.710891f, 0.590901f, 0.115938f, 0.903375f,
                        0.836745f, 0.069482f, 0.116163f, 0.846875f, 0.326503f, 0.798514f, 0.633905f, 0.037332f,
                        0.040814f, 0.783123f, 0.964177f, 0.412064f, 0.855732f, 0.856115f, 0.143356f, 0.934137f,
                        0.433961f, 0.869262f, 0.245320f, 0.333171f, 0.593450f, 0.394133f, 0.373404f, 0.979914f,
                        0.298339f, 0.100796f, 0.924156f, 0.092880f, 0.904197f, 0.797754f, 0.588189f, 0.650559f,
                        0.259428f, 0.505293f, 0.784722f, 0.098935f, 0.578755f, 0.632079f, 0.023725f, 0.777252f,
                        0.067884f, 0.777330f, 0.382775f, 0.445708f, 0.639517f, 0.234484f, 0.749223f, 0.856886f,
                        0.232946f, 0.954170f, 0.772150f, 0.246118f, 0.289955f, 0.954520f, 0.122751f, 0.934407f });

    set_values(trans, { -0.029787f, -0.039425f, -0.013103f, 0.020071f, -0.005835f, -0.012874f, -0.017001f, 0.049894f,
                        0.029097f, -0.009970f, -0.022878f, -0.047258f, 0.020377f, 0.046071f, 0.041883f, 0.006665f,
                        -0.017147f, -0.019289f, 0.033301f, 0.035090f, -0.034251f, 0.019805f, 0.046993f, 0.004558f,
                        -0.008064f, -0.006918f, 0.025468f, 0.028423f, 0.047196f, 0.033526f, -0.030316f, -0.047028f,
                        0.039442f, 0.031440f, 0.002416f, -0.015918f, -0.017300f, 0.039576f, 0.049565f, 0.028253f,
                        -0.043583f, 0.034208f, -0.036632f, 0.013871f, 0.003656f, -0.047041f, 0.013766f, 0.027901f,
                        0.038762f, 0.028222f, -0.006699f, 0.004693f, 0.039893f, -0.000607f, 0.007333f, -0.021024f,
                        0.004991f, -0.022489f, -0.004437f, 0.027367f, -0.032716f, 0.028834f, -0.046343f, -0.009434f,
                        0.027402f, 0.000910f, 0.033518f, 0.039552f, 0.006292f, 0.037292f, 0.004389f, -0.001523f,
                        -0.035141f, 0.038158f, -0.039541f, 0.044087f, 0.035931f, 0.007894f, -0.033210f, -0.033552f,
                        0.007346f, 0.049394f, -0.030189f, 0.003506f, -0.001177f, -0.042913f, 0.002683f, 0.025040f,
                        -0.043625f, 0.032408f, 0.020641f, 0.000095f, -0.048939f, -0.036238f, -0.013676f, -0.007374f,
                        0.022051f, -0.039209f, -0.016096f, 0.027290f, -0.040269f, -0.010659f, -0.000863f, 0.029705f,
                        0.038543f, -0.002328f, -0.014264f, -0.035752f, 0.034962f, 0.044321f, 0.021213f, 0.027358f,
                        -0.027710f, -0.043171f, -0.022418f, 0.038013f, 0.043587f, -0.010271f, 0.045589f, -0.003316f,
                        -0.012297f, -0.046647f, -0.036306f, 0.001580f, -0.012444f, 0.015811f, -0.044044f, -0.032120f,
                        -0.036106f, 0.046225f, 0.026619f, -0.007977f, 0.031591f, 0.010924f, -0.016834f, 0.011595f,
                        0.035778f, -0.035381f, 0.020276f, -0.028779f, 0.020281f, -0.048127f, -0.023946f, 0.019259f,
                        -0.033553f, -0.031123f, -0.022753f, 0.038951f, -0.025033f, 0.043039f, 0.022913f, 0.023352f,
                        -0.027936f, 0.005870f, -0.006296f, 0.013130f, 0.049880f, 0.002440f, 0.019296f, -0.021542f,
                        0.042561f, 0.013915f, -0.004669f, 0.017946f, 0.029617f, -0.012364f, 0.013526f, 0.037983f,
                        0.026276f, -0.012129f, -0.030703f, -0.004503f, -0.034408f, 0.024183f, 0.042694f, -0.021127f,
                        -0.000750f, 0.015751f, -0.006451f, -0.027637f, -0.017465f, 0.029969f, -0.042837f, 0.000506f,
                        -0.019212f, 0.009024f, 0.015760f, -0.036041f, -0.023333f, -0.037582f, -0.004512f, 0.022489f,
                        -0.025172f, 0.047346f, -0.022394f, 0.038075f, 0.000998f, 0.032706f, -0.035312f, -0.030298f,
                        -0.045665f, 0.048576f, 0.014447f, 0.038639f, 0.014167f, -0.013453f, 0.014609f, -0.043763f,
                        -0.048391f, 0.035368f, -0.039751f, 0.035484f, 0.013826f, -0.020628f, 0.046977f, -0.042135f,
                        0.030593f, -0.003539f, -0.025274f, -0.024285f, 0.003617f, 0.007755f, -0.035767f, 0.011002f,
                        0.037875f, 0.009636f, 0.038495f, 0.008297f, 0.002958f, 0.012495f, 0.018759f, -0.045469f,
                        -0.031202f, -0.000778f, -0.031627f, -0.046655f, 0.000654f, 0.045007f, -0.000535f, -0.006192f,
                        -0.025460f, 0.000590f, 0.002009f, -0.005429f, 0.038686f, -0.025255f, 0.035947f, -0.034878f,
                        -0.013894f, 0.037885f, 0.029620f, -0.043924f, -0.000509f, -0.025247f, 0.046582f, 0.030141f,
                        0.037285f, 0.025552f, 0.038033f, -0.038338f, -0.001596f, 0.000761f, -0.025951f, -0.011722f,
                        0.002122f, 0.006186f, -0.019645f, -0.012891f, -0.042582f, 0.027550f, 0.031169f, -0.006275f,
                        -0.001020f, 0.049283f, 0.024151f, -0.001658f, -0.008065f, -0.004660f, -0.036786f, 0.007594f,
                        0.010722f, -0.016304f, 0.018261f, -0.045930f, 0.019360f, 0.011701f, 0.027280f, 0.032347f });

    set_values(weights, { 0.570468f, 0.164136f, 0.776458f, 0.114689f, 0.419818f, 0.175685f, 0.800586f, 0.955510f,
                            0.172536f, 0.221061f, 0.461171f, 0.231518f, 0.061834f, 0.739035f, 0.125626f, 0.847827f,
                            0.495315f, 0.628024f, 0.050780f, 0.074000f, 0.046359f, 0.312483f, 0.728176f, 0.303299f,
                            0.951573f, 0.958894f, 0.553264f, 0.369629f, 0.485841f, 0.870666f, 0.064903f, 0.113451f,
                            0.589594f, 0.246304f, 0.813138f, 0.967322f, 0.449608f, 0.783049f, 0.537315f, 0.914755f,
                            0.737451f, 0.330159f, 0.248510f, 0.384153f, 0.995351f, 0.734002f, 0.674795f, 0.912569f,
                            0.779339f, 0.227471f, 0.536402f, 0.811472f, 0.149300f, 0.974410f, 0.635431f, 0.190016f,
                            0.511683f, 0.327507f, 0.475696f, 0.981653f, 0.690011f, 0.964964f, 0.137833f, 0.178613f,
                            0.457133f, 0.519849f, 0.738906f, 0.043956f, 0.086127f, 0.569203f, 0.818306f, 0.154500f,
                            0.178339f, 0.309057f, 0.798050f, 0.520464f, 0.891095f, 0.935555f, 0.588479f, 0.399601f,
                            0.834115f, 0.284009f, 0.148420f, 0.374753f, 0.781456f, 0.792354f, 0.593585f, 0.156931f,
                            0.065250f, 0.533898f, 0.385339f, 0.628777f, 0.623290f, 0.343597f, 0.661532f, 0.781720f,
                            0.630958f, 0.696810f, 0.409430f, 0.305062f, 0.516459f, 0.113841f, 0.089201f, 0.637092f,
                            0.138792f, 0.397683f, 0.516309f, 0.751039f });

    std::vector<float> output_vec = { 1.970516f, 3.382788f, 3.232190f, 2.336809f, 3.345522f, 3.613247f, 3.728191f, 3.020900f,
                                    2.941497f, 4.088605f, 4.330465f, 2.704222f, 1.416622f, 1.978842f, 1.908965f, 1.283542f,
                                    3.169041f, 3.549672f, 3.379996f, 2.635512f, 4.070296f, 4.150237f, 3.749203f, 2.677094f,
                                    4.008467f, 3.803390f, 4.592338f, 2.864883f, 1.829590f, 2.018164f, 2.118501f, 0.737497f,
                                    2.472763f, 2.802295f, 4.671433f, 2.367759f, 3.826317f, 5.116533f, 6.496192f, 3.471319f,
                                    4.765404f, 5.568552f, 5.531154f, 3.182333f, 2.685795f, 3.444699f, 3.776934f, 2.262869f,
                                    1.818490f, 3.280025f, 3.267537f, 2.819613f, 2.515392f, 4.808901f, 4.601172f, 3.314271f,
                                    2.718597f, 4.912858f, 4.508485f, 3.421931f, 1.883965f, 2.337842f, 2.899203f, 1.303477f,
                                    2.060152f, 3.820107f, 3.858294f, 1.984789f, 3.717936f, 5.461279f, 4.718024f, 3.145632f,
                                    3.807969f, 4.831135f, 5.490211f, 2.392655f, 2.803777f, 3.287232f, 3.329570f, 1.954188f,
                                    1.874439f, 3.715954f, 3.360692f, 2.248626f, 3.255451f, 4.810193f, 4.815291f, 3.047631f,
                                    3.744109f, 4.851815f, 4.863840f, 2.928771f, 1.937537f, 2.858193f, 2.596052f, 1.853897f };

    topology topology(
            input_layout("input", input->get_layout()),
            input_layout("trans", trans->get_layout()),
            data("weights", weights),
            convolution(
                    "conv",
                    { input_info("input"), input_info("trans") },
                    "weights",
                    no_bias,
                    true,
                    2,
                    1,
                    { 1, 1 },
                    { 1, 1 },
                    { 1, 1 },
                    { 1, 1 },
                    true)
    );

    network network(engine, topology, cfg);
    network.set_input_data("input", input);
    network.set_input_data("trans", trans);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 4);
    ASSERT_EQ(f_size, 6);
    ASSERT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        ASSERT_NEAR(output_vec[i], output_ptr[i], 0.01);
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_no_bias) {
    //  Filter : 2x3
    //  Stride : 2x1
    //  Input  : 4x5
    //  Output : 2x3
    //
    //  Input:
    //  1  2  3  4  5
    //  2  2  3  4  6
    //  3  3  3  5  1
    //  1  1  1  1  1
    //
    //  Filter:
    //  1  2  1
    //  2  1  2
    //
    //  Output:
    // 21  28  39
    // 18  20  20

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 3, 2 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });
    VVF<float> output_vec = {
        { 20.0f, 27.0f, 38.0f },
        { 17.0f, 19.0f, 19.0f } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        convolution("conv", input_info("input"), "weights", no_bias, 1, { 2, 1 }, {1, 1}, {0, 0}, {0, 0}, false));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }

    //VVF temp_vec(y_size, VF(x_size, 0.0f));
    //for (int y = 0; y < y_size; ++y) {
    //    for (int x = 0; x < x_size; ++x) {
    //        temp_vec[y][x] = output_ptr[y * x_size + x];
    //    }
    //}
    //print_2d(temp_vec);
}


TEST(convolution_f32_fw_gpu, basic_convolution_no_bias_dynamic) {
    auto& engine = get_test_engine();

    ov::Shape in0_shape = { 1, 1, 4, 5 };

    auto in0_dyn_layout = layout{ov::PartialShape::dynamic(in0_shape.size()), data_types::f32, format::bfyx};
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 3, 2 } });

    set_values(weights, {
        1.0f, 2.0f, 1.0f,
        2.0f, 1.0f, 2.0f
    });

    topology topology(
        input_layout("input", in0_dyn_layout),
        data("weights", weights),
        convolution("conv", input_info("input"), "weights", no_bias, 1, { 2, 1 }, {1, 1}, {0, 0}, {0, 0}, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    // first execute
    {
        auto input0 = engine.allocate_memory({ in0_shape, data_types::f32, format::bfyx });
        set_values(input0, {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            2.0f, 2.0f, 3.0f, 4.0f, 6.0f,
            3.0f, 3.0f, 3.0f, 5.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f
        });
        network.set_input_data("input", input0);

        auto inst = network.get_primitive("conv");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "conv");

        auto output_memory = outputs.at("conv").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

        int y_size = output_layout.spatial(1);
        int x_size = output_layout.spatial(0);
        int f_size = output_layout.feature();
        int b_size = output_layout.batch();
        ASSERT_EQ(output_layout.format, format::bfyx);
        ASSERT_EQ(y_size, 2);
        ASSERT_EQ(x_size, 3);
        ASSERT_EQ(f_size, 1);
        ASSERT_EQ(b_size, 1);

        VVF<float> output_vec = {
            { 20.0f, 27.0f, 38.0f },
            { 17.0f, 19.0f, 19.0f }
        };
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
            }
        }
    }

    // second execute
    {
        in0_shape = { 1, 1, 6, 4 };
        auto input0 = engine.allocate_memory({ in0_shape, data_types::f32, format::bfyx });
        set_values(input0, {
            1.0f, 2.0f, 3.0f, 4.0f,
            2.0f, 2.0f, 3.0f, 4.0f,
            3.0f, 3.0f, 3.0f, 5.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            5.0f, 4.0f, 3.0f, 2.0f,
            4.0f, 4.0f, 3.0f, 3.0f,
        });
        network.set_input_data("input", input0);

        auto inst = network.get_primitive("conv");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "conv");

        auto output_memory = outputs.at("conv").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

        int y_size = output_layout.spatial(1);
        int x_size = output_layout.spatial(0);
        int f_size = output_layout.feature();
        int b_size = output_layout.batch();
        ASSERT_EQ(output_layout.format, format::bfyx);
        ASSERT_EQ(y_size, 3);
        ASSERT_EQ(x_size, 2);
        ASSERT_EQ(f_size, 1);
        ASSERT_EQ(b_size, 1);

        VVF<float> output_vec = {
            { 20.f, 27.f },
            { 17.f, 19.f },
            { 34.f, 29.f }
        };
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
            }
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_int8_no_bias) {
    //  Filter : 2x3
    //  Stride : 2x1
    //  Input  : 4x5
    //  Output : 2x3
    //
    //  Input:
    //  1  2  3  4  5
    //  2  2  3  4  6
    //  3  3  3  5  1
    //  1  1  1  1  1
    //
    //  Filter:
    //  1  2  1
    //  2  1  2
    //
    //  Output:
    // 21  28  39
    // 18  20  20

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, { 1, 1, 3, 2 } });

    set_values(input, { 1.1f, 2.4f, 3.5f, 4.5f, 5.8f,
                        2.9f, 2.3f, 3.5f, 4.4f, 6.6f,
                        3.8f, 3.9f, 3.4f, 5.1f, 1.4f,
                        1.8f, 1.1f, 1.2f, 1.2f, 1.9f });
    set_values<int8_t>(weights, { 1, 2, 1,
                                  2, 1, 2 });
    VVF<float> output_vec = {
        { 20.0f, 27.0f, 38.0f },
        { 17.0f, 19.0f, 19.0f } };

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("to_int", input_info("input"), { data_types::i8, format::bfyx, { 1, 1, 5, 4 } }),
        data("weights", weights),
        convolution("conv", input_info("to_int"), "weights", no_bias, 1, {2, 1}, {1, 1}, {0, 0}, {0, 0}, false),
        reorder("output", input_info("conv"), { data_types::f32, format::bfyx, { 1, 1, 3, 2 } }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution3D_no_bias) {
    //  data is similar as in basic_convolution_no_bias

    //  Filter : 2x3x1
    //  Stride : 2x1x1
    //  Input  : 4x5x1
    //  Output : 2x3x1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 3, 2 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });

    VVF<float> output_vec = {
        { 20.0f, 27.0f, 38.0f },
        { 17.0f, 19.0f, 19.0f } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        convolution("conv", input_info("input"), "weights", no_bias, 1, {2, 1}, {1, 1}, {0, 0}, {0, 0}, false));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int z_size = output_layout.spatial(2);
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(z_size, 1);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution3D) {
    //  Input  : 4x4x4
    //  Filter : 2x2x2
    //  Output : 3x3x3
    //
    //  Input:
    //  1  0  1  0
    //  1  1  3  1
    //  1  1  0  2
    //  0  2  1  1
    //
    //  1  0  0  1
    //  2  0  1  2
    //  3  1  1  1
    //  0  0  3  1
    //
    //  2  0  1  1
    //  3  3  1  0
    //  2  1  1  0
    //  3  2  1  2
    //
    //  1  0  2  0
    //  1  0  3  3
    //  3  1  0  0
    //  1  1  0  2
    //
    //  Filter:
    //  0  1
    //  0  0
    //
    //  2  1
    //  0  0

    //  Output:
    //  2  1  1
    //  5  4  5
    //  8  3  5
    //
    //  4  1  4
    //  9  8  4
    //  6  4  3
    //
    //  2  3  5
    //  5  4  9
    //  8  3  0
    //
    //  Bias:
    //  1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { 1, 1, 4, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfzyx, { 1, 1, 2, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1, 1 } });

    set_values(input, {
        1.0f,  0.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  3.0f,  1.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        0.0f,  2.0f,  1.0f,  1.0f,
        1.0f,  0.0f,  0.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  2.0f,
        3.0f,  1.0f,  1.0f,  1.0f,
        0.0f,  0.0f,  3.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  1.0f,
        3.0f,  3.0f,  1.0f,  0.0f,
        2.0f,  1.0f,  1.0f,  0.0f,
        3.0f,  2.0f,  1.0f,  2.0f,
        1.0f,  0.0f,  2.0f,  0.0f,
        1.0f,  0.0f,  3.0f,  3.0f,
        3.0f,  1.0f,  0.0f,  0.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
    });

    set_values(weights, {
        0.0f,  1.0f,
        0.0f,  0.0f,
        2.0f,  1.0f,
        0.0f,  0.0f,
    });

    set_values(biases, { 1.0f });

    VVVF<float> output_vec = {
        {
            { 3.0f,   2.0f,   2.0f },
            { 6.0f,   5.0f,   6.0f },
            { 9.0f,   4.0f,   6.0f }
        },
        {
            { 5.0f,   2.0f,   5.0f },
            { 10.0f,   9.0f,   5.0f },
            { 7.0f,   5.0f,   4.0f }
        },
        {
            { 3.0f,   4.0f,   6.0f },
            { 6.0f,   5.0f,   10.0f },
            { 9.0f,   4.0f,   1.0f }
        }
    };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, false));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int z_size = output_layout.spatial(2);
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfzyx);
    ASSERT_EQ(z_size, 3);
    ASSERT_EQ(y_size, 3);
    ASSERT_EQ(x_size, 3);

    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);
    for (int z = 0; z < z_size; ++z) {
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_EQ(output_vec[z][y][x], output_ptr[z * y_size * x_size + y * x_size + x]);
            }
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution3D_group2) {
    //  data is similar as in basic_convolution3D_split2
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 2, 4, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goizyx, tensor(format::goizyx,  {2, 1, 1, 2, 2, 2 }) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 2, 1, 1, 1 } });

    set_values(input, {
        1.0f,  0.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  3.0f,  1.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        0.0f,  2.0f,  1.0f,  1.0f,
        1.0f,  0.0f,  0.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  2.0f,
        3.0f,  1.0f,  1.0f,  1.0f,
        0.0f,  0.0f,  3.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  1.0f,
        3.0f,  3.0f,  1.0f,  0.0f,
        2.0f,  1.0f,  1.0f,  0.0f,
        3.0f,  2.0f,  1.0f,  2.0f,
        1.0f,  0.0f,  2.0f,  0.0f,
        1.0f,  0.0f,  3.0f,  3.0f,
        3.0f,  1.0f,  0.0f,  0.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        1.0f,  0.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  3.0f,  1.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        0.0f,  2.0f,  1.0f,  1.0f,
        1.0f,  0.0f,  0.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  2.0f,
        3.0f,  1.0f,  1.0f,  1.0f,
        0.0f,  0.0f,  3.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  1.0f,
        3.0f,  3.0f,  1.0f,  0.0f,
        2.0f,  1.0f,  1.0f,  0.0f,
        3.0f,  2.0f,  1.0f,  2.0f,
        1.0f,  0.0f,  2.0f,  0.0f,
        1.0f,  0.0f,  3.0f,  3.0f,
        3.0f,  1.0f,  0.0f,  0.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
    });

    set_values(weights, {
        0.0f,  1.0f,
        0.0f,  0.0f,
        2.0f,  1.0f,
        0.0f,  0.0f,
        0.0f,  1.0f,
        0.0f,  0.0f,
        2.0f,  1.0f,
        0.0f,  0.0f,
    });

    set_values(biases, { 1.0f, 2.0f });

    VVVVF<float> output_vec = {
        {
            {
                { 3.0f,   2.0f,   2.0f },
                { 6.0f,   5.0f,   6.0f },
                { 9.0f,   4.0f,   6.0f }
            },
            {
                { 5.0f,   2.0f,   5.0f },
                { 10.0f,   9.0f,   5.0f },
                { 7.0f,   5.0f,   4.0f }
            },
            {
                { 3.0f,   4.0f,   6.0f },
                { 6.0f,   5.0f,   10.0f },
                { 9.0f,   4.0f,   1.0f }
            },
        },
        {
            {
                { 4.0f,   3.0f,   3.0f },
                { 7.0f,   6.0f,   7.0f },
                { 10.0f,  5.0f,   7.0f }
            },
            {
                { 6.0f,   3.0f,   6.0f },
                { 11.0f,  10.0f,  6.0f },
                { 8.0f,   6.0f,   5.0f }
            },
            {
                { 4.0f,   5.0f,   7.0f },
                { 7.0f,   6.0f,  11.0f },
                { 10.0f,  5.0f,   2.0f }
            },
        }
    };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 2, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, true));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int z_size = output_layout.spatial(2);
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfzyx);
    ASSERT_EQ(b_size, 1);
    ASSERT_EQ(f_size, 2);
    ASSERT_EQ(z_size, 3);
    ASSERT_EQ(y_size, 3);
    ASSERT_EQ(x_size, 3);
    for (int f = 0; f < f_size; ++f) {
        for (int z = 0; z < z_size; ++z) {
            for (int y = 0; y < y_size; ++y) {
                for (int x = 0; x < x_size; ++x) {
                    ASSERT_EQ(output_vec[f][z][y][x],
                        output_ptr[f * z_size * y_size * x_size + z * y_size * x_size + y * x_size + x]);
                }
            }
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution3D_group2_dynamic) {
    //  data is similar as in basic_convolution3D_split2
    auto& engine = get_test_engine();

    ov::Shape in0_shape = { 1, 2, 4, 4, 4 };
    auto in0_dyn_layout = layout{ov::PartialShape::dynamic(in0_shape.size()), data_types::f32, format::bfzyx};

    auto input0 = engine.allocate_memory({ in0_shape, data_types::f32, format::bfzyx });
    auto weights = engine.allocate_memory({ data_types::f32, format::goizyx, tensor(format::goizyx, {2, 1, 1, 2, 2, 2 }) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfzyx, { 1, 2, 1, 1, 1 } });

    set_values(input0, {
        1.0f,  0.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  3.0f,  1.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        0.0f,  2.0f,  1.0f,  1.0f,
        1.0f,  0.0f,  0.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  2.0f,
        3.0f,  1.0f,  1.0f,  1.0f,
        0.0f,  0.0f,  3.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  1.0f,
        3.0f,  3.0f,  1.0f,  0.0f,
        2.0f,  1.0f,  1.0f,  0.0f,
        3.0f,  2.0f,  1.0f,  2.0f,
        1.0f,  0.0f,  2.0f,  0.0f,
        1.0f,  0.0f,  3.0f,  3.0f,
        3.0f,  1.0f,  0.0f,  0.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        1.0f,  0.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  3.0f,  1.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        0.0f,  2.0f,  1.0f,  1.0f,
        1.0f,  0.0f,  0.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  2.0f,
        3.0f,  1.0f,  1.0f,  1.0f,
        0.0f,  0.0f,  3.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  1.0f,
        3.0f,  3.0f,  1.0f,  0.0f,
        2.0f,  1.0f,  1.0f,  0.0f,
        3.0f,  2.0f,  1.0f,  2.0f,
        1.0f,  0.0f,  2.0f,  0.0f,
        1.0f,  0.0f,  3.0f,  3.0f,
        3.0f,  1.0f,  0.0f,  0.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
    });

    set_values(weights, {
        0.0f,  1.0f,
        0.0f,  0.0f,
        2.0f,  1.0f,
        0.0f,  0.0f,
        0.0f,  1.0f,
        0.0f,  0.0f,
        2.0f,  1.0f,
        0.0f,  0.0f,
    });

    set_values(biases, { 1.0f, 2.0f });

    VVVVF<float> output_vec = {
        {
            {
                { 3.0f,   2.0f,   2.0f },
                { 6.0f,   5.0f,   6.0f },
                { 9.0f,   4.0f,   6.0f }
            },
            {
                { 5.0f,   2.0f,   5.0f },
                { 10.0f,   9.0f,   5.0f },
                { 7.0f,   5.0f,   4.0f }
            },
            {
                { 3.0f,   4.0f,   6.0f },
                { 6.0f,   5.0f,   10.0f },
                { 9.0f,   4.0f,   1.0f }
            },
        },
        {
            {
                { 4.0f,   3.0f,   3.0f },
                { 7.0f,   6.0f,   7.0f },
                { 10.0f,  5.0f,   7.0f }
            },
            {
                { 6.0f,   3.0f,   6.0f },
                { 11.0f,  10.0f,  6.0f },
                { 8.0f,   6.0f,   5.0f }
            },
            {
                { 4.0f,   5.0f,   7.0f },
                { 7.0f,   6.0f,  11.0f },
                { 10.0f,  5.0f,   2.0f }
            },
        }
    };

    topology topology(
        input_layout("input", in0_dyn_layout),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 2, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, true));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    network.set_input_data("input", input0);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int z_size = output_layout.spatial(2);
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfzyx);
    ASSERT_EQ(b_size, 1);
    ASSERT_EQ(f_size, 2);
    ASSERT_EQ(z_size, 3);
    ASSERT_EQ(y_size, 3);
    ASSERT_EQ(x_size, 3);
    for (int f = 0; f < f_size; ++f) {
        for (int z = 0; z < z_size; ++z) {
            for (int y = 0; y < y_size; ++y) {
                for (int x = 0; x < x_size; ++x) {
                    ASSERT_EQ(output_vec[f][z][y][x],
                        output_ptr[f * z_size * y_size * x_size + z * y_size * x_size + y * x_size + x]);
                }
            }
        }
    }
}

TEST(convolution_f32_fw_gpu, three_convolutions_same_weights) {
    //  Filter : 1x1
    //  Input  : 2x2
    //  Output : 2x2
    //
    //  Input:
    //  1  1   1  1
    //  1  1   1  1
    //
    //  Filter:
    //  1
    //
    //  Output:
    //  8  8   8  8
    //  8  8   8  8

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 1 } });

    set_values(input, { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        convolution("conv1", input_info("input"), "weights", no_bias, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false),
        convolution("conv2", input_info("conv1"), "weights", no_bias, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false),
        convolution("conv3", input_info("conv2"), "weights", no_bias, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_memory = outputs.at("conv3").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();

    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 2);
    ASSERT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_FLOAT_EQ(8.0f, output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution) {
    //  Filter : 2x3
    //  Stride : 2x1
    //  Input  : 4x5
    //  Output : 2x3
    //
    //  Input:
    //  1  2  3  4  5
    //  2  2  3  4  6
    //  3  3  3  5  1
    //  1  1  1  1  1
    //
    //  Filter:
    //  1  2  1
    //  2  1  2
    //
    //  Output:
    // 21  28  39
    // 18  20  20
    //
    //  Bias:
    //  1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 3, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 21.0f, 28.0f, 39.0f },
        { 18.0f, 20.0f, 20.0f } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution( "conv", input_info("input"), "weights", "biases", 1, {2, 1}, {1, 1}, {0, 0}, {0, 0}, false));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, convolution_big_size_weights) {
    auto& engine = get_test_engine();

    const std::vector<int> filter_size_data = {
        33, 33,
    };

    const std::vector<std::string> impl_kernel_data = {
        "convolution_gpu_ref__f32"
    };

    for (size_t m = 0 ; m < filter_size_data.size() / 2; m++) {
        const int in_y = filter_size_data[m * 2];
        const int in_x = filter_size_data[m * 2 + 1];

        auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, in_y, in_x } });
        auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, in_y, in_x } });
        auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

        tests::random_generator rg(GET_SUITE_NAME);
        VVVVF<float> input_rnd = rg.generate_random_4d<float>(1, 1, in_y, in_x, -10, 10);
        VF<float> input_rnd_vec = flatten_4d<float>(format::yxfb, input_rnd);
        VVVVF<float> filter_rnd = rg.generate_random_4d<float>(1, 1, in_y, in_x, -10, 10);
        VF<float> filter_rnd_vec = flatten_4d<float>(format::bfyx, filter_rnd);

        set_values(biases, { 0.0f });
        set_values(input, input_rnd_vec);
        set_values(weights, filter_rnd_vec);

        float output_sum = 0.f;
        size_t idx = 0;
        for (int i = 0 ; i < in_y; i++) {
            for (int k = 0 ; k < in_x; k++) {
                idx = i * in_x + k;
                output_sum += input_rnd_vec[idx] * filter_rnd_vec[idx];
            }
        }

        topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            data("biases", biases),
            convolution( "conv", input_info("input"), "weights", "biases", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network network(engine, topology, config);

        auto impl_info = network.get_implementation_info("conv");
        ASSERT_EQ(impl_info, impl_kernel_data[m]);

        network.set_input_data("input", input);

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "conv");

        auto output_memory = outputs.at("conv").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

        int y_size = output_layout.spatial(1);
        int x_size = output_layout.spatial(0);
        int f_size = output_layout.feature();
        int b_size = output_layout.batch();

        ASSERT_EQ(y_size, 1);
        ASSERT_EQ(x_size, 1);
        ASSERT_EQ(f_size, 1);
        ASSERT_EQ(b_size, 1);

        ASSERT_EQ(output_sum, output_ptr[0]);
    }

}

TEST(convolution_f16_fw_gpu, convolution_big_size_weights) {
    auto& engine = get_test_engine();

    const std::vector<int> filter_size_data = {
        65, 33,
        65, 32,
    };

    const std::vector<std::string> impl_kernel_data = {
        "convolution_gpu_ref__f16",
        "convolution_gpu_bfyx_gemm_like__f16",
    };

    for (size_t m = 0 ; m < filter_size_data.size() / 2; m++) {
        const int in_y = filter_size_data[m * 2];
        const int in_x = filter_size_data[m * 2 + 1];

        auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, in_x, in_y } });
        auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, in_x, in_y } });

        tests::random_generator rg(GET_SUITE_NAME);
        VVVVF<ov::float16> input_rnd = rg.generate_random_4d<ov::float16>(1, 1, in_y, in_x, -10, 10);
        VF<ov::float16> input_rnd_vec = flatten_4d<ov::float16>(format::bfyx, input_rnd);
        VVVVF<ov::float16> filter_rnd = rg.generate_random_4d<ov::float16>(1, 1, in_y, in_x, -10, 10);
        VF<ov::float16> filter_rnd_vec = flatten_4d<ov::float16>(format::bfyx, filter_rnd);

        set_values(input, input_rnd_vec);
        set_values(weights, filter_rnd_vec);

        topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            convolution( "conv", input_info("input"), "weights", no_bias, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network network(engine, topology, config);

        auto impl_info = network.get_implementation_info("conv");
        ASSERT_EQ(impl_info, impl_kernel_data[m]);

        network.set_input_data("input", input);

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "conv");

        auto output_memory = outputs.at("conv").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<ov::float16> output_ptr(output_memory, get_test_stream());

        int y_size = output_layout.spatial(1);
        int x_size = output_layout.spatial(0);
        int f_size = output_layout.feature();
        int b_size = output_layout.batch();

        ASSERT_EQ(y_size, 1);
        ASSERT_EQ(x_size, 1);
        ASSERT_EQ(f_size, 1);
        ASSERT_EQ(b_size, 1);
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_bfyx_weights_as_input_layout) {
    //Same params as convolution_f32_fw_gpu, basic_convolution but with bfyx optimized data and weights set as input_layout
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 5, 4 }});
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 3, 2 }});
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 }});
    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });
    set_values(biases, { 1.0f });

    VVF<float> output_vec = {
        { 21.0f, 28.0f, 39.0f },
        { 18.0f, 20.0f, 20.0f }
    };
    topology topology(
        input_layout("input", input->get_layout()),
        input_layout("weights", weights->get_layout()),
        input_layout("biases", biases->get_layout()),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 2, 1 }, { 1, 1 }, {0, 0}, {0, 0}, false));
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);
    network.set_input_data("weights", weights);
    network.set_input_data("biases", biases);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_bfyx_weights_as_input_layout_non_opt_build) {
    //Same params as convolution_f32_fw_gpu, basic_convolution but with bfyx optimized data and weights set as input_layout
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 5, 4 }});
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 3, 2 }});
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 }});
    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });
    set_values(biases, { 1.0f });

    VVF<float> output_vec = {
        { 21.0f, 28.0f, 39.0f },
        { 18.0f, 20.0f, 20.0f }
    };

    topology topology(
        input_layout("input", input->get_layout()),
        input_layout("weights", weights->get_layout()),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 2, 1 }, {1, 1}, { 0, 0 }, {0, 0}, false));
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(false));
    network network(engine, topology, config, true);
    network.set_input_data("input", input);
    network.set_input_data("weights", weights);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_input_padding) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : 2x1
    //  Output : 6x5
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //  z  1  2  3  4  z
    //  z  2  2  3  4  z
    //  z  3  3  3  5  z
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1
    //  2  4  6  8  5
    //  4  8 11 15  9
    //  6 11 12 16 10
    //  4  7  7  9  6
    //  1  1  1  1  1
    //
    //  Bias:
    //  1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 4, 3 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 2.0f, 4.0f, 6.0f, 8.0f, 5.0f },
        { 4.0f, 8.0f, 11.0f, 15.0f, 9.0f },
        { 6.0f, 11.0f, 12.0f, 16.0f, 10.0f },
        { 4.0f, 7.0f, 7.0f, 9.0f, 6.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            input_info("input"),
            "weights",
            "biases",
            1,
            { 1, 1 },
            { 1, 1 },
            { 2, 1 },
            { 2, 1 },
            false,
            ov::op::PadType::EXPLICIT,
            padding{ { 0, 0, 0, 0 }, 0 })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 6);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }

    //VVF temp_vec(y_size, VF(x_size, 0.0f));
    //for (int y = 0; y < y_size; ++y) {
    //    for (int x = 0; x < x_size; ++x) {
    //        temp_vec[y][x] = output_ptr[y * x_size + x];
    //    }
    //}
    //print_2d(temp_vec);
}

TEST(convolution_f32_fw_gpu, basic_convolution_sym_input_padding) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : above 2x1, below 2x1
    //  Output : 6x5
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //  z  1  2  3  4  z
    //  z  2  2  3  4  z
    //  z  3  3  3  5  z
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1
    //  2  4  6  8  5
    //  4  8 11 15  9
    //  6 11 12 16 10
    //  4  7  7  9  6
    //  1  1  1  1  1
    //
    //  Bias:
    //  1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 4, 3 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 2.0f, 4.0f, 6.0f, 8.0f, 5.0f },
        { 4.0f, 8.0f, 11.0f, 15.0f, 9.0f },
        { 6.0f, 11.0f, 12.0f, 16.0f, 10.0f },
        { 4.0f, 7.0f, 7.0f, 9.0f, 6.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            input_info("input"),
            "weights",
            "biases",
            1,
            { 1, 1 },
            { 1, 1 },
            { 2, 1 },
            { 2, 1 },
            false,
            ov::op::PadType::EXPLICIT,
            padding{ { 0, 0, 0, 0 }, 0 })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 6);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_asym_input_padding) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : above 2x1, below 3x2
    //  Output : 7x6
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  1  2  3  4  z  z
    //  z  2  2  3  4  z  z
    //  z  3  3  3  5  z  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1  1
    //  2  4  6  8  5  1
    //  4  8 11 15  9  1
    //  6 11 12 16 10  1
    //  4  7  7  9  6  1
    //  1  1  1  1  1  1
    //  1  1  1  1  1  1
    //
    //  Bias:
    //  1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 4, 3 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 2.0f, 4.0f, 6.0f, 8.0f, 5.0f, 1.0f },
        { 4.0f, 8.0f, 11.0f, 15.0f, 9.0f, 1.0f },
        { 6.0f, 11.0f, 12.0f, 16.0f, 10.0f, 1.0f },
        { 4.0f, 7.0f, 7.0f, 9.0f, 6.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            input_info("input"),
            "weights",
            "biases",
            1,
            { 1, 1 },
            { 1, 1 },
            { 2, 1 },
            { 3, 2 },
            false,
            ov::op::PadType::EXPLICIT,
            padding{ { 0, 0, 0, 0 }, 0 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 7);
    ASSERT_EQ(x_size, 6);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_sym_input_padding_with_pad) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : above 2x1, below 2x1
    //  Input offset: 2x1
    //  Output : 10x7
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z
    //  z  z  1  2  3  4  z  z
    //  z  z  2  2  3  4  z  z
    //  z  z  3  3  3  5  z  z
    //  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1
    //  1  2  4  6  8  5  1
    //  1  4  8 11 15  9  1
    //  1  6 11 12 16 10  1
    //  1  4  7  7  9  6  1
    //  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1
    //
    //  Bias:
    //  1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 4, 3 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 2.0f, 4.0f, 6.0f, 8.0f, 5.0f, 1.0f },
        { 1.0f, 4.0f, 8.0f, 11.0f, 15.0f, 9.0f, 1.0f },
        { 1.0f, 6.0f, 11.0f, 12.0f, 16.0f, 10.0f, 1.0f },
        { 1.0f, 4.0f, 7.0f, 7.0f, 9.0f, 6.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            input_info("input"),
            "weights",
            "biases",
            1,
            { 1, 1 },
            { 1, 1 },
            ov::CoordinateDiff{ 4, 2 },
            ov::CoordinateDiff{ 4, 2 },
            false,
            ov::op::PadType::EXPLICIT,
            padding{ { 0, 0, 0, 0 }, 0 })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 10);
    ASSERT_EQ(x_size, 7);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_asym_input_padding_with_pad) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : above 2x1, below 3x2
    //  Input offset: 2x1
    //  Output : 11x8
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  1  2  3  4  z  z  z
    //  z  z  2  2  3  4  z  z  z
    //  z  z  3  3  3  5  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1
    //  1  2  4  6  8  5  1  1
    //  1  4  8 11 15  9  1  1
    //  1  6 11 12 16 10  1  1
    //  1  4  7  7  9  6  1  1
    //  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1
    //
    //  Bias:
    //  1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 4, 3 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 2.0f, 4.0f, 6.0f, 8.0f, 5.0f, 1.0f, 1.0f },
        { 1.0f, 4.0f, 8.0f, 11.0f, 15.0f, 9.0f, 1.0f, 1.0f },
        { 1.0f, 6.0f, 11.0f, 12.0f, 16.0f, 10.0f, 1.0f, 1.0f },
        { 1.0f, 4.0f, 7.0f, 7.0f, 9.0f, 6.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            input_info("input"),
            "weights",
            "biases",
            1,
            { 1, 1 },
            { 1, 1 },
            { 4, 2 },
            { 5, 3 },
            false,
            ov::op::PadType::EXPLICIT,
            padding{ { 0, 0, 0, 0 }, 0 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 11);
    ASSERT_EQ(x_size, 8);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_input_and_output_padding) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : 2x1
    //  Output : 8x9
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //  z  1  2  3  4  z
    //  z  2  2  3  4  z
    //  z  3  3  3  5  z
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1  1
    //  1  1  2  4  6  8  5  1  1
    //  1  1  4  8 11 15  9  1  1
    //  1  1  6 11 12 16 10  1  1
    //  1  1  4  7  7  9  6  1  1
    //  1  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1  1
    //
    //  Bias:
    //  1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 4, 3 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 2.0f, 4.0f, 6.0f, 8.0f, 5.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 4.0f, 8.0f, 11.0f, 15.0f, 9.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 6.0f, 11.0f, 12.0f, 16.0f, 10.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 4.0f, 7.0f, 7.0f, 9.0f, 6.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    const int x_pad = 2;
    const int y_pad = 1;
    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            input_info("input"),
            "weights",
            "biases",
            1,
            { 1, 1 },
            { 1, 1 },
            { 2, 1 },
            { 2, 1 },
            false,
            ov::op::PadType::EXPLICIT,
            padding{ { 0,0,x_pad,y_pad }, 0 })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    auto padded_dims = output_layout.get_padded_dims();
    auto non_padded_dims = output_layout.get_dims();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int b_size = padded_dims[0];
    int f_size = padded_dims[1];
    int y_size = padded_dims[2];
    int x_size = padded_dims[3];
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 8);
    ASSERT_EQ(x_size, 9);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    ASSERT_EQ(non_padded_dims[0], 1);
    ASSERT_EQ(non_padded_dims[1], 1);
    ASSERT_EQ(non_padded_dims[2], 6);
    ASSERT_EQ(non_padded_dims[3], 5);

    for (int y = y_pad; y < y_size - y_pad; ++y)
    {
        for (int x = x_pad; x < x_size - x_pad; ++x)
        {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }

    //VVF temp_vec(y_size, VF(x_size, 0.0f));
    //for (int y = 0; y < y_size; ++y) {
    //    for (int x = 0; x < x_size; ++x) {
    //        temp_vec[y][x] = output_ptr[y * x_size + x];
    //    }
    //}
    //print_2d(temp_vec);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad_random) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2
    //
    //  Input:
    //  rnd  rnd  rnd  rnd
    //  rnd  rnd  rnd  rnd
    //  rnd  rnd  rnd  rnd
    //  rnd  rnd  rnd  rnd
    //
    //  Filter
    //  rnd  rnd
    //  rnd  rnd
    //
    //  Bias
    //  rnd
    //
    //  Output:
    //  rnd  rnd
    //  rnd  rnd

    size_t batch = 1, input_f = 1, input_y = 4, input_x = 4;
    tests::random_generator rg(GET_SUITE_NAME);

    VVVVF<float> input_rnd = rg.generate_random_4d<float>(batch, input_f, input_y, input_x, -10, 10);
    VF<float> input_rnd_vec = flatten_4d<float>(format::yxfb, input_rnd);
    VVVVF<float> filter_rnd = rg.generate_random_4d<float>(1, 1, 2, 2, -10, 10);
    VF<float> filter_rnd_vec = flatten_4d<float>(format::bfyx, filter_rnd);
    VF<float> bias_rnd = rg.generate_random_1d<float>(1, -10, 10);
    VVVVF<float> output_rnd(batch, VVVF<float>(filter_rnd.size()));
    for (size_t b = 0; b < output_rnd.size(); ++b) {
        for (size_t of = 0; of < filter_rnd.size(); ++of) {
            output_rnd[b][of] = reference_convolve<float>(input_rnd[b], filter_rnd[of], 2, 2, bias_rnd[of]);
        }
    }
    VF<float> output_rnd_vec = flatten_4d<float>(format::yxfb, output_rnd);

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32,  format::yxfb, { 1, 1, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32, { 1, { 2, 2 }, 1 } });
    auto weights = engine.allocate_memory({ data_types::f32,  format::bfyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32,  format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, input_rnd_vec);
    set_values(weights, filter_rnd_vec);
    set_values(biases, bias_rnd);

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 2, 2 }, {1, 1}, {0, 0}, {0, 0}, false)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    for (size_t i = 0; i < output_rnd.size(); ++i) {
        float x = float_round(output_rnd_vec[i]), y = float_round(output_ptr[i]);
        ASSERT_FLOAT_EQ(x, y) << "random seed = " << random_seed << std::endl;
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in2x2x1x2_nopad_random) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x2x1x2
    //  Output : 1x1x1x2
    //
    //  Input:
    //  rnd  rnd    rnd  rnd
    //  rnd  rnd    rnd  rnd
    //
    //  Filter:
    //  rnd  rnd
    //  rnd  rnd
    //
    //  Bias:
    //  rnd
    //
    //  Output:
    //  rnd  rnd

    size_t batch = 2, input_f = 1, input_y = 2, input_x = 2;
    tests::random_generator rg(GET_SUITE_NAME);

    VVVVF<float> input_rnd = rg.generate_random_4d<float>(batch, input_f, input_y, input_x, -10, 10);
    VF<float> input_rnd_vec = flatten_4d<float>(format::yxfb, input_rnd);
    VVVVF<float> filter_rnd = rg.generate_random_4d<float>(1, 1, 2, 2, -10, 10);
    VF<float> filter_rnd_vec = flatten_4d<float>(format::bfyx, filter_rnd);
    VF<float> bias_rnd = rg.generate_random_1d<float>(1, -10, 10);
    VVVVF<float> output_rnd(batch, VVVF<float>(filter_rnd.size()));
    for (size_t b = 0; b < output_rnd.size(); ++b) {
        for (size_t of = 0; of < filter_rnd.size(); ++of) {
            output_rnd[b][of] = reference_convolve<float>(input_rnd[b], filter_rnd[of], 2, 2, bias_rnd[of]);
        }
    }
    VF<float> output_rnd_vec = flatten_4d<float>(format::yxfb, output_rnd);

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32, { 2, { 1, 1 }, 1 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, input_rnd_vec);
    set_values(weights, filter_rnd_vec);
    set_values(biases, bias_rnd);

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 2, 2 }, {1, 1}, {0, 0}, {0, 0}, false)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    for (size_t i = 0; i < output_rnd.size(); ++i) {
        float x = float_round(output_rnd_vec[i]), y = float_round(output_ptr[i]);
        ASSERT_FLOAT_EQ(x, y) << "random seed = " << random_seed << std::endl;
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2
    //
    //  Input:
    //  -0.5   1     0.5  2
    //   1.5  -0.5   0   -1
    //   0.5   0.5  -1    1
    //   0.5   2     1.5 -0.5
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  8  0.5
    //  6  9

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32, { 1, { 2, 2 }, 1 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 2, 2 }, {1, 1}, {0, 0}, {0, 0}, false)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(8.0f, output_ptr[0]);
    ASSERT_FLOAT_EQ(0.5f, output_ptr[1]);
    ASSERT_FLOAT_EQ(6.0f, output_ptr[2]);
    ASSERT_FLOAT_EQ(9.0f, output_ptr[3]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in2x2x1x2_nopad) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x2x1x2
    //  Output : 1x1x1x2
    //
    //  Input:
    //  0.5   1.5    2.3 -0.4
    //  2.0  -4.0    1.0  3.0
    //
    //  Filter:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias:
    //  -1
    //
    //  Output:
    //  3.65 -5.36
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32, { 2, { 1, 1 }, 1 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 0.5f, 2.3f, 1.5f, -0.4f, 2.0f, 1.0f, -4.0f, 3.0f });
    set_values(weights, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases, { -1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 2, 2 }, {1, 1}, {0, 0}, {0, 0}, false)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(3.65f, output_ptr[0]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[1]);
}

TEST(convolution_f32_fw_gpu, basic_ofm_wsiz2x1x2x1_in1x2x1_nopad) {
    //  Filter : 1x2x1x2x1
    //  Input  : 1x1x2x1
    //  Output : 1x2x1x1
    //
    //  Input:
    //  1.0    2.0
    //
    // Filter:
    //   1.0    2.0  ofm=0
    //  -1.0   -2.0  ofm=1
    //
    //  Bias:
    //  0.1 -0.2
    //
    //  Output:
    //   5.1  f=0
    //  -5.2  f=1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 1, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32, { 1, { 1, 1 }, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 1, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 1.0f, 2.0f });
    set_values(weights, { 1.0f, 2.0f, -1.0f, -2.0f });
    set_values(biases, { 0.1f, -0.2f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 5, 5 }, {1, 1}, {0, 0}, {0, 0}, false)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(5.1f, output_ptr[0]);
    ASSERT_FLOAT_EQ(-5.2f, output_ptr[1]);
}

TEST(convolution_f32_fw_gpu, basic_ofm_wsiz3x2x2x1_in2x2x1_nopad) {
    //  Filter : 1x3x2x2x1
    //  Input  : 1x2x2x1
    //  Output : 1x3x1x1
    //
    //  Input:
    //  1.0    2.0  f=0
    //  3.0    4.0  f=1
    //
    // Filter:
    //   1.0    2.0  ifm=0  ofm=0
    //   3.0    4.0  ifm=1
    //
    //   5.0    6.0  ifm=0  ofm=1
    //   7.0    8.0  ifm=1
    //
    //   9.0   10.0  ifm=0  ofm=2
    //  11.0   12.0  ifm=1
    //  Bias:
    //   -5     -6     -7
    //
    //  Output:
    //   25.0  f=0
    //   64,0  f=1
    //  103.0  f=2

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 2, 1, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32, { 1, { 1, 1 }, 3 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 2, 1, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 1, 1 } });

    set_values(input, { 1.0f, 3.0f, 2.0f, 4.0f });
    set_values(weights, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f });
    set_values(biases, { -5.0f, -6.0f, -7.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 5, 5 }, {1, 1}, {0, 0}, {0, 0}, false)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(25.0f, output_ptr[0]);
    ASSERT_FLOAT_EQ(64.0f, output_ptr[1]);
    ASSERT_FLOAT_EQ(103.0f, output_ptr[2]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2x1x3_wstr2x2_in2x2x1x1_nopad) {
    //  Filter : 2x2x1x3
    //  Stride : 2x2
    //  Input  : 2x2x1x1
    //  Output : 1x1x3x1
    //
    //  Input:
    //  -2.3 -0.1
    //   3.1  1.9
    //
    //  Filter:
    //  -1.1  1.5       0.1  0.2        2.0  -1.0
    //   0.5 -0.5       0.4  0.7        2.5  -1.5
    //
    //  Bias:
    //  0.1 -0.2 0.3
    //
    //  Output:
    //   0.7
    //   2.12
    //   3.08

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32, { 1, { 1, 1 }, 3 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 1, 1 } });

    set_values(input, { -2.3f, -0.1f, 3.1f, 1.9f });
    set_values(weights, { -1.1f, 1.5f, 0.5f, -0.5f, 0.1f, 0.2f, 0.4f, 0.7f, 2.0f, -1.0f, 2.5f, -1.5f });
    set_values(biases, { 0.1f, -0.2f, 0.3f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 2, 2 }, {1, 1}, {0, 0}, {0, 0}, false)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_TRUE(are_equal(3.08f, output_ptr[0]));
    ASSERT_TRUE(are_equal(2.12f, output_ptr[1]));
    ASSERT_TRUE(are_equal(0.7f,  output_ptr[2]));
}

TEST(convolution_f32_fw_gpu, wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
    //  Filter  : 3x3
    //  Stride  : 2x2
    //  Input   : 2x2
    //  Output  : 1x1
    //  Padding : zero
    //
    //  Input:
    //  -0.5   1.0   padd
    //   0.5   2.0   padd
    //  padd  padd   padd
    //
    //  Filter
    //  -2    0.5  3.5
    //   1.5  4   -5
    //   0.5  1.5 -1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  12.25
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32, { 1, { 1, 1 }, 1 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 2, 2 }, {1, 1}, {0, 0}, {1, 1}, false)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(12.25f, output_ptr[0]);
}

TEST(convolution_f32_fw_gpu, offsets_wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
    //   Filter       : 3x3
    //   Stride       : 2x2
    //   Input        : 2x2
    //   Input offset : -1x-1
    //   Output       : 2x2
    //   Output offset: 1x1
    //   Padding      : zero
    //
    //   Input:
    //   padd padd  padd
    //   padd -0.5   1
    //   padd  0.5   2.0
    //
    //   Filter
    //   -2    0.5  3.5
    //    1.5  4   -5
    //    0.5  1.5 -1.5
    //
    //   Bias
    //   2
    //
    //   Output:
    //   rnd   rnd
    //   rnd   2.0
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32, { 1, { 2, 2 }, 1 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            input_info("input"),
            "weights",
            "biases",
            1,
            { 2, 2 },
            { 1, 1 },
            { 1, 1 },
            { 1, 1 },
            false,
            ov::op::PadType::EXPLICIT,
            padding{ { 0, 0, 1, 1 }, 0 })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(-7.25f, output_ptr[4]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_split2) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4x2
    //  Output : 2x2x2
    //
    //  Input:
    //  f0: -0.5   1     0.5  2
    //       1.5  -0.5   0   -1
    //       0.5   0.5  -1    1
    //       0.5   2     1.5 -0.5
    //
    //  f1:  0.5   1.5   2.3 -0.4
    //       2.0  -4.0   1.0  3.0
    //       0.5   1.5   2.3 -0.4
    //       2.0  -4.0   1.0  3.0
    //
    //  Filter1:
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias1:
    //  2
    //
    //  Filter2:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias2:
    //  -1

    //  Output:
    //   8  3.65 0.5 -5.36
    //   6  3.65 9   -5.36

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 2, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32, { 1, { 2, 2 }, 2 } });
    auto weights1 = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2))});
    auto biases1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, {
        -0.5f,  0.5f,  1.0f,  1.5f,  0.5f,  2.3f,  2.0f, -0.4f,
        1.5f,  2.0f, -0.5f, -4.0f,  0.0f,  1.0f, -1.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  1.5f, -1.0f,  2.3f,  1.0f, -0.4f,
        0.5f,  2.0f,  2.0f, -4.0f,  1.5f,  1.0f, -0.5f,  3.0f
    });
    set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f, -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases1, { 2.0f, -1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights1", weights1),
        data("biases1", biases1),
        convolution(
            "conv",
            input_info("input"),
            { "weights1" },
            { "biases1" },
            2,
            { 2, 2 },
            { 1, 1 },
            { 0, 0 },
            { 0, 0 },
            true)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(8.0f,   output_ptr[0]);
    ASSERT_FLOAT_EQ(3.65f,  output_ptr[1]);
    ASSERT_FLOAT_EQ(0.5f,   output_ptr[2]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[3]);
    ASSERT_FLOAT_EQ(6.0f,   output_ptr[4]);
    ASSERT_FLOAT_EQ(3.65f,  output_ptr[5]);
    ASSERT_FLOAT_EQ(9.0f,   output_ptr[6]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[7]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2) {
    //  2x Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x4x4x2
    //  Output : 2x2x2x2
    //
    //  Input:
    //  f0b0: -0.5   1     0.5  2
    //         1.5  -0.5   0   -1
    //         0.5   0.5  -1    1
    //         0.5   2     1.5 -0.5
    //
    //  f0b1: -0.5   1     0.5  2
    //         1.5  -0.5   0   -1
    //         0.5   0.5  -1    1
    //         0.5   2     1.5 -0.5
    //
    //  f1b0:  0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //         0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //
    //  f1b1:  0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //         0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //
    //
    //  Filter1:
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias1:
    //  2
    //
    //  Filter2:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias2:
    //  -1

    //  Output:
    //   8  8 3.65 3.65 0.5  0.5 -5.36 -5.36
    //   6  6 3.65 3.65 9    9   -5.36 -5.36

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32, { 2, { 2, 2 }, 2 } });
    auto weights1 = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, {
       -0.5f, -0.5f,  0.5f,  0.5f,  1.0f,  1.0f,  1.5f,  1.5f,  0.5f,  0.5f,  2.3f,  2.3f,  2.0f,  2.0f, -0.4f, -0.4f,
        1.5f,  1.5f,  2.0f,  2.0f, -0.5f, -0.5f, -4.0f, -4.0f,  0.0f,  0.0f,  1.0f,  1.0f, -1.0f, -1.0f,  3.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  1.5f,  1.5f, -1.0f, -1.0f,  2.3f,  2.3f,  1.0f,  1.0f, -0.4f, -0.4f,
        0.5f,  0.5f,  2.0f,  2.0f,  2.0f,  2.0f, -4.0f, -4.0f,  1.5f,  1.5f,  1.0f,  1.0f, -0.5f, -0.5f,  3.0f,  3.0f,
    });
    set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f, -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases1, { 2.0f, -1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights1", weights1),
        data("biases1", biases1),
        convolution(
            "conv",
            input_info("input"),
            "weights1",
            "biases1",
            2,
            { 2, 2 },
            { 1, 1 },
            { 0, 0 },
            { 0, 0 },
            true)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(8.0f,   output_ptr[0]);
    ASSERT_FLOAT_EQ(8.0f,   output_ptr[1]);
    ASSERT_FLOAT_EQ(3.65f,  output_ptr[2]);
    ASSERT_FLOAT_EQ(3.65f,  output_ptr[3]);
    ASSERT_FLOAT_EQ(0.5f,   output_ptr[4]);
    ASSERT_FLOAT_EQ(0.5f,   output_ptr[5]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[6]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[7]);
    ASSERT_FLOAT_EQ(6.0f,   output_ptr[8]);
    ASSERT_FLOAT_EQ(6.0f,   output_ptr[9]);
    ASSERT_FLOAT_EQ(3.65f,  output_ptr[10]);
    ASSERT_FLOAT_EQ(3.65f,  output_ptr[11]);
    ASSERT_FLOAT_EQ(9.0f,   output_ptr[12]);
    ASSERT_FLOAT_EQ(9.0f,   output_ptr[13]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[14]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[15]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_group2) {
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_split2
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 2, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, {
        -0.5f,  0.5f,  1.0f,  1.5f,  0.5f,  2.3f,  2.0f, -0.4f,
        1.5f,  2.0f, -0.5f, -4.0f,  0.0f,  1.0f, -1.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  1.5f, -1.0f,  2.3f,  1.0f, -0.4f,
        0.5f,  2.0f,  2.0f, -4.0f,  1.5f,  1.0f, -0.5f,  3.0f
    });
    set_values(weights, {
        -2.0f, 0.5f, 3.5f, 1.5f,
        -1.2f, 1.5f, 0.5f, -0.5f
    });
    set_values(biases, { 2.0f, -1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            input_info("input"),
            "weights",
            "biases",
            2, // number of groups
            { 2, 2 },
            { 1, 1 },
            { 0, 0 },
            { 0, 0 },
            true)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(8.0f, output_ptr[0]);
    ASSERT_FLOAT_EQ(3.65f, output_ptr[1]);
    ASSERT_FLOAT_EQ(0.5f, output_ptr[2]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[3]);
    ASSERT_FLOAT_EQ(6.0f, output_ptr[4]);
    ASSERT_FLOAT_EQ(3.65f, output_ptr[5]);
    ASSERT_FLOAT_EQ(9.0f, output_ptr[6]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[7]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_group2_bfyx) {
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_split2

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 2, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, {
        -0.5f,  0.5f,  1.0f,  1.5f,  0.5f,  2.3f,  2.0f, -0.4f,
        1.5f,  2.0f, -0.5f, -4.0f,  0.0f,  1.0f, -1.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  1.5f, -1.0f,  2.3f,  1.0f, -0.4f,
        0.5f,  2.0f,  2.0f, -4.0f,  1.5f,  1.0f, -0.5f,  3.0f
    });
    set_values(weights, {
        -2.0f, 0.5f, 3.5f, 1.5f,
        -1.2f, 1.5f, 0.5f, -0.5f
    });
    set_values(biases, { 2.0f, -1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("input_1", input_info("input"), { data_types::f32, format::bfyx, { 1, 2, 4, 4 } }),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            input_info("input_1"),
            "weights",
            "biases",
            2, // number of groups
            { 2, 2 },
            { 1, 1 },
            { 0, 0 },
            { 0, 0 },
            true)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(8.0f, output_ptr[0]);
    ASSERT_FLOAT_EQ(0.5f, output_ptr[1]);
    ASSERT_FLOAT_EQ(6.0f, output_ptr[2]);
    ASSERT_FLOAT_EQ(9.0f, output_ptr[3]);
    ASSERT_FLOAT_EQ(3.65f, output_ptr[4]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[5]);
    ASSERT_FLOAT_EQ(3.65f, output_ptr[6]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[7]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_group2) {
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, {
        -0.5f, -0.5f,  0.5f,  0.5f,  1.0f,  1.0f,  1.5f,  1.5f,  0.5f,  0.5f,  2.3f,  2.3f,  2.0f,  2.0f, -0.4f, -0.4f,
        1.5f,  1.5f,  2.0f,  2.0f, -0.5f, -0.5f, -4.0f, -4.0f,  0.0f,  0.0f,  1.0f,  1.0f, -1.0f, -1.0f,  3.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  1.5f,  1.5f, -1.0f, -1.0f,  2.3f,  2.3f,  1.0f,  1.0f, -0.4f, -0.4f,
        0.5f,  0.5f,  2.0f,  2.0f,  2.0f,  2.0f, -4.0f, -4.0f,  1.5f,  1.5f,  1.0f,  1.0f, -0.5f, -0.5f,  3.0f,  3.0f,
    });
    set_values(weights, {
        -2.0f, 0.5f, 3.5f, 1.5f,
        -1.2f, 1.5f, 0.5f, -0.5f
    });
    set_values(biases, { 2.0f, -1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            input_info("input"),
            "weights",
            "biases",
            2, // number of groups
            { 2, 2 },
            { 1, 1 },
            { 0, 0 },
            { 0, 0 },
            true)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(8.0f, output_ptr[0]);
    ASSERT_FLOAT_EQ(8.0f, output_ptr[1]);
    ASSERT_FLOAT_EQ(3.65f, output_ptr[2]);
    ASSERT_FLOAT_EQ(3.65f, output_ptr[3]);
    ASSERT_FLOAT_EQ(0.5f, output_ptr[4]);
    ASSERT_FLOAT_EQ(0.5f, output_ptr[5]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[6]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[7]);
    ASSERT_FLOAT_EQ(6.0f, output_ptr[8]);
    ASSERT_FLOAT_EQ(6.0f, output_ptr[9]);
    ASSERT_FLOAT_EQ(3.65f, output_ptr[10]);
    ASSERT_FLOAT_EQ(3.65f, output_ptr[11]);
    ASSERT_FLOAT_EQ(9.0f, output_ptr[12]);
    ASSERT_FLOAT_EQ(9.0f, output_ptr[13]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[14]);
    ASSERT_FLOAT_EQ(-5.36f, output_ptr[15]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2_depthwise_sep_opt) {
    //  Test for depthwise separable optimization, there are 16 weights and biases (split 16)
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2 but with batch 1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 16, 4, 4 } });

    set_values(input, {
        -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f,
        1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f,
        0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f,
        2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f,
        1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f,
        -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f,
        0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f,
        0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f,
        -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f,
        1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f,
        0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f,
        2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f,
        1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f,
        -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f,
    });

    auto weights1 = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(16), batch(1), feature(1), spatial(2, 2)) });
    auto biases1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 16, 1, 1 } });

    set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f });

    set_values(biases1, { 2.0f, -1.0f,
                          2.0f, -1.0f,
                          2.0f, -1.0f,
                          2.0f, -1.0f,
                          2.0f, -1.0f,
                          2.0f, -1.0f,
                          2.0f, -1.0f,
                          2.0f, -1.0f });

    primitive_id weights_id = "weights1";
    primitive_id bias_id = "biases1";

    topology topology(
        input_layout("input", input->get_layout()),
        data(weights_id, weights1),
        data(bias_id, biases1),
        convolution(
                "conv",
                input_info("input"),
                weights_id,
                bias_id,
                16,  // number of groups
                { 2, 2 },
                { 1, 1 },
                { 0, 0 },
                { 0, 0 },
                true)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f,
        0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f,
        6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f,
        9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2_depthwise_sep_opt_bfyx) {
    //  Test for depthwise separable optimization, there are 16 weights and biases (split 16)
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2 but with batch 1
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 16, 4, 4 } });

    set_values(input, {
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
    });

    auto weights1 = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(16), batch(1), feature(1), spatial(2, 2)) });
    auto biases1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 16, 1, 1 } });

    set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f,
                           -2.0f, 0.5f, 3.5f, 1.5f,  -1.2f, 1.5f, 0.5f, -0.5f });

    set_values(biases1, { 2.0f, -1.0f,
                          2.0f, -1.0f,
                          2.0f, -1.0f,
                          2.0f, -1.0f,
                          2.0f, -1.0f,
                          2.0f, -1.0f,
                          2.0f, -1.0f,
                          2.0f, -1.0f });

    primitive_id weights_id = "weights1";
    primitive_id bias_id = "biases1";

    topology topology(
            input_layout("input", input->get_layout()),
            data(weights_id, weights1),
            data(bias_id, biases1),
            convolution(
            "conv",
            input_info("input"),
            weights_id,
            bias_id,
            16,  // number of groups
            { 2, 2 },
            { 1, 1 },
            { 0, 0 },
            { 0, 0 },
            true)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_group16) {
    //  Test for grouped convolution, there are 16 joined weights and biases (group 16)
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2_depthwise_sep_opt

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 16, 4, 4 } });

    set_values(input, {
        -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f,
        1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f,
        0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f,
        2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f,
        1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f,
        -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f,
        0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f,
        0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f,
        -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f,
        1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f,
        0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f,
        2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f,
        1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f,
        -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f,
    });

    topology topology(input_layout("input", input->get_layout()));

    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(16), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 16, 1, 1 } });

    set_values(weights,
        {
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f
        }
    );
    set_values(biases, { 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f });

    topology.add(
        data("weights", weights),
        data("bias", biases)
    );

    topology.add(
        convolution(
            "conv",
            input_info("input"),
            "weights",
            "bias",
            16,
            { 2, 2 },
            { 1, 1 },
            { 0, 0 },
            { 0, 0 },
            true)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f,
        0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f,
        6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f,
        9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_group16_bfyx) {
    //  Test for grouped convolution, there are 16 joined weights and biases (group 16)
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2_depthwise_sep_opt_bfyx
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 16, 4, 4 } });

    set_values(input, {
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
    });

    topology topology(input_layout("input", input->get_layout()));

    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(16), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 16, 1, 1 } });

    set_values(weights,
        {
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f
        }
    );

    set_values(biases, { 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f });

    topology.add(
            data("weights", weights),
            data("bias", biases)
    );

    topology.add(
        convolution(
            "conv",
            input_info("input"),
            "weights",
            "bias",
            16,
            { 2, 2 },
            { 1, 1 },
            { 0, 0 },
            { 0, 0 },
            true)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(convolution_gpu, trivial_convolution_relu) {

    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2

    //  Input:
    //  -0.5   1     0.5  2
    //   1.5  -0.5   0   -1
    //   0.5   0.5  -1    1
    //   0.5   2     1.5 -0.5
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  -2
    //
    //  Output:
    //  4  0.0
    //  2  5

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32, { 1, { 2, 2 }, 1 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        -0.5f,  1.0f,  0.5f,  2.0f,
        1.5f, -0.5f,  0.0f, -1.0f,
        0.5f,  0.5f, -1.0f,  1.0f,
        0.5f,  2.0f,  1.5f, -0.5f
    });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { -2.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 2, 2 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, false),
        activation("out", input_info("conv"), activation_func::relu)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(4.0f, output_ptr[0]);
    ASSERT_FLOAT_EQ(0.0f, output_ptr[1]);
    ASSERT_FLOAT_EQ(2.0f, output_ptr[2]);
    ASSERT_FLOAT_EQ(5.0f, output_ptr[3]);
}

TEST(convolution_gpu, relu_with_negative_slope) {

    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2
    //  Negative Slope : 0.1

    //  Input:
    //  -0.5   1     0.5  2
    //   1.5  -0.5   0   -1
    //   0.5   0.5  -1    1
    //   0.5   2     1.5 -0.5
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  -2
    //
    //  Output:
    //  4  -0.35
    //  2  5

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        -0.5f,  1.0f,  0.5f,  2.0f,
        1.5f, -0.5f,  0.0f, -1.0f,
        0.5f,  0.5f, -1.0f,  1.0f,
        0.5f,  2.0f,  1.5f, -0.5f
    });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { -2.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 2, 2 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, false),
        activation("out", input_info("conv"), activation_func::relu_negative_slope, { 0.1f, 0.0f } )
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(4.0f, output_ptr[0]);
    ASSERT_FLOAT_EQ(-0.35f, output_ptr[1]);
    ASSERT_FLOAT_EQ(2.0f, output_ptr[2]);
    ASSERT_FLOAT_EQ(5.0f, output_ptr[3]);
}

TEST(convolution_gpu, DISABLED_two_1x1_kernels_after_each_other) {

    auto& engine = get_test_engine();

    extern const std::vector<float> conv_1x1_output;

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 16, 8, 16, 16 } });
    auto weights_conv_1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 8, 8, 1, 1 } });
    auto weights_conv_2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 8, 1, 1 } });

    set_random_values<float>(input);
    set_random_values<float>(weights_conv_1);
    set_random_values<float>(weights_conv_2);

    auto inp_lay = input_layout("input", input->get_layout());
    auto conv_1 = convolution("conv_1", input_info("input"), "weights_conv_1", no_bias, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false);
    auto conv_2 = convolution("conv_2", input_info("conv_1"), "weights_conv_2", no_bias, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false);

    topology topology(
        inp_lay,
        data("weights_conv_1", weights_conv_1),
        conv_1,
        data("weights_conv_2", weights_conv_2),
        conv_2
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));

    auto output_prim = outputs.at("conv_2").get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
    auto output_layout = output_prim->get_layout();

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    int f_offset = y_size * x_size;
    int b_offset = f_size * f_offset;
    for (int b = 0; b < b_size; ++b) {
        for (int f = 0; f < f_size; ++f) {
            for (int y = 0; y < y_size; ++y) {
                for (int x = 0; x < x_size; ++x) {
                    int idx = b * b_offset + f * f_offset + y * x_size + x;
                    ASSERT_TRUE(are_equal(conv_1x1_output[idx], output_ptr[idx]));
                }
            }
        }
    }
}

TEST(convolution_gpu, basic_yxfb_4_4_yxfb_2_2_b16_if2_of16_st2_2_p0_sp1_fp32)
{
#define USE_OLD_WEIGHTS_FORMAT 0

    const auto input_format   = format::yxfb;
#if USE_OLD_WEIGHTS_FORMAT
    const auto weights_format = format::bfyx;
#else
    const auto weights_format = format::yxfb;
#endif
    const auto biases_format = format::bfyx;

    const int32_t batch_size = 16;
    const int32_t input_feature_count = 2;
    const int32_t output_feature_count = 16;

    const int32_t stride_x = 2;
    const int32_t stride_y = 2;

    const int32_t input_x = 4;
    const int32_t input_y = 4;
    const int32_t weights_x = 2;
    const int32_t weights_y = 2;
    const int32_t output_x = (input_x - weights_x) / stride_x + 1;
    const int32_t output_y = (input_y - weights_y) / stride_y + 1;

    auto& engine = get_test_engine();

    auto input_size = tensor( batch_size, input_feature_count, input_x, input_y );
    auto input = engine.allocate_memory({ data_types::f32, input_format, input_size });
    auto weights_size = tensor( output_feature_count, input_feature_count, weights_x, weights_y );
    auto weights = engine.allocate_memory({ data_types::f32, weights_format, weights_size });
    auto biases = engine.allocate_memory({ data_types::f32, biases_format, { 1, output_feature_count, 1, 1 } });

    //auto output = memory::allocate({ output_format, { batch_size, { output_x, output_y }, output_feature_count } });

    // input:
    std::vector<float> input_vals_template {
        0.25f, 0.50f, 0.75f, 1.00f,
        1.25f, 1.50f, 1.75f, 2.00f,
        2.25f, 2.50f, 2.75f, 3.00f,
        3.25f, 3.50f, 3.75f, 4.00f,
    };
    input_vals_template.resize(input_y * input_x);

    std::vector<float> input_vals;
    input_vals.reserve(input_y * input_x * input_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < input_y * input_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                input_vals.push_back((bi * input_feature_count + ifi + 1) * input_vals_template[yxi]);
            }
        }
    }
    set_values(input, input_vals);

    // weights:
    std::vector<float> weights_vals_template {
        -4.0f, -2.0f,
         4.0f,  4.0f,
    };
    weights_vals_template.resize(weights_y * weights_x);

    std::vector<float> weights_vals;
    weights_vals.reserve(weights_y * weights_x * input_feature_count * output_feature_count);
#if USE_OLD_WEIGHTS_FORMAT
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#else
    for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#endif
    set_values(weights, weights_vals);

    // biases:
    std::vector<float> biases_vals;
    biases_vals.reserve(output_feature_count);
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        biases_vals.push_back(ofi * 1.0f);
    }
    set_values(biases, biases_vals);

    // output:
    std::vector<float> output_vals_template {
         9.0f, 10.0f,
        13.0f, 14.0f,
    };
    output_vals_template.resize(output_y * output_x);

    std::vector<float> output_vals;
    output_vals.reserve(output_y * output_x * output_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                uint32_t template_factor = input_feature_count * input_feature_count * input_feature_count * bi * ofi +
                    input_feature_count * input_feature_count * (input_feature_count + 1) / 2 * (bi + ofi) +
                    input_feature_count * (input_feature_count + 1) * (2 * input_feature_count + 1) / 6;
                float bias_factor = ofi * 1.0f;

                output_vals.push_back(template_factor * output_vals_template[yxi] + bias_factor);
            }
        }
    }

    // Computing convolution.
    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { stride_y, stride_x }, { 1, 1 }, { 0, 0 }, { 0, 0 }, false),
        activation("out", input_info("conv"), activation_func::relu, { 0.1f, 0.0f })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    // Checking result.
    uint32_t i = 0;
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi, ++i)
            {
                auto equal = are_equal(output_vals[i], output_ptr[i]);
                ASSERT_TRUE(equal);
                if (!equal)
                {
                    std::cout << "Failed at position (" << yxi << ", output feature = " << ofi << ", batch = " << bi << "): "
                        << output_vals[i] << " != " << output_ptr[i] << std::endl;
                    return;
                }
            }
        }
    }

#undef USE_OLD_WEIGHTS_FORMAT
}

TEST(convolution_f32_fw_gpu, byte_activation) {
    //  Filter : 2x3
    //  Stride : 2x1
    //  Input  : 4x5
    //  Output : 2x3
    //
    //  Input:
    //  1  2  3  4  5
    //  2  2  3  4  6
    //  3  3  3  5  1
    //  1  1  1  1  1
    //
    //  Filter:
    //  1  2  1
    //  2  1  2
    //
    //  19 17 -1
    // -10 32 23
    //
    //  Output:
    // 21  28  39
    // 18  20  20
    //
    // -101 -11 92
    // -114 -116 -78
    //
    //  Bias:
    //  1 -8
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::i8, format::bfyx, { 1, 1, 5, 4 } });

    VVVF<int8_t> output_vec = {
        {
            { 11, 0, 15 },
            { 0,  0, 2 }
        },
        {
            { 33, 0, 0 },
            { 0, 0, 0 }
        } };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    set_values<int8_t>(input, {  1,  2, -3,  4, -5,
                                 2, -2,  3, -4,  6,
                                -3,  3, -3,  5, -1,
                                -1, -1, -1, -1, -1 });
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, { 2, 1, 3, 2 } });

    std::vector<int8_t> weights_values = { 1,  2,  1,
                                           2,  1,  2,
                                          19, 17, -1,
                                         -10, 32, 23 };
    set_values<int8_t>(weights, weights_values);
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });
    set_values(biases, { 1.0f, -8.0f });

    topology topology(input_layout("input", input->get_layout()),
                      data("weights", weights),
                      data("biases", biases),
                      convolution("conv", input_info("input"), "weights", "biases", 1, { 2, 1 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, false));

    if (engine.get_device_info().supports_immad) {
        // Add reorder not to concern about query oneDNN (e.g. src:acdb, wei:Abcd8a, dst:acdb)
        topology.add(
            activation( "act", input_info("conv"), activation_func::relu),
            reorder("out", input_info("act"), format::bfyx, data_types::f32)
        );
    } else {
        topology.add(
            activation( "out", input_info("conv"), activation_func::relu)
        );
    }

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();

    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 2);
    ASSERT_EQ(b_size, 1);
    ASSERT_EQ(output_layout.format, format::bfyx);

    for (int f = 0; f < f_size; f++) {
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_NEAR(output_vec[f][y][x], ((float)output_ptr[f * y_size * x_size + y * x_size + x]), 3.0f);
            }
        }
    }
}

TEST(convolution_int8_fw_gpu, quantized_convolution_u8s8f32_symmetric) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 1, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, { 2, 1, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });
    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5 });
    set_values(biases, { 1.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { 52.0f, 78.0f, 3.0f },
            { 8.0f, 14.0f, 0.0f }
        },
        {
            { 20.0f, 35.0f, 31.0f },
            { 11.0f, 19.0f, 0.0f }
        } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("input"), "weights", "biases", 1, { 2, 2 }, { 1, 1 }, { 0, 0 }, { 1, 2 }, false),
        reorder("out", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    auto output_layout = output_memory->get_layout();
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 2);
    ASSERT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_NEAR(output_vec[f][y][x], ((float)output_ptr[f * y_size * x_size + y * x_size + x]), 1e-5f) <<
                " x="<<x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_int8_fw_gpu, quantized_convolution_u8s8f32_asymmetric_weight_and_activations) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 1, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, { 2, 1, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });
    auto w_zp = engine.allocate_memory({ data_types::i8, format::bfyx, { 2, 1, 1, 1 } });
    auto a_zp = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 1, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });
    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5 });
    set_values<uint8_t>(a_zp, { 2 });
    set_values<int8_t>(w_zp, { 1, -1 });
    set_values(biases, { 1.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { 12.0f, 26.0f, -19.0f },
            { 2.0f, 8.0f, 4.0f }
        },
        {
            { -8.0f, 19.0f, 21.0f },
            { -7.0f, 1.0f, -18.0f }
        } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("a_zp", a_zp),
        data("w_zp", w_zp),
        convolution("conv", input_info("input"), "weights", "biases", "w_zp", "a_zp", "", 1,
                    { 2, 2 }, { 1, 1 }, { 0, 0 }, { 1, 2 }, false, data_types::f32),
        reorder("out", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    auto output_layout = output_memory->get_layout();
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 2);
    ASSERT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_NEAR(output_vec[f][y][x], ((float)output_ptr[f * y_size * x_size + y * x_size + x]), 1e-5f) <<
                " x="<< x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_int8_fw_gpu, quantized_convolution_u8s8f32_asymmetric_activations_per_tensor) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 1, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, { 2, 1, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });
    auto a_zp = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 1, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });
    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5 });
    set_values<uint8_t>(a_zp, { 2 });
    set_values(biases, { 1.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { 16.0f, 42.0f, -13.0f },
            { 2.0f, 8.0f, 2.0f }
        },
        {
            { -12.0f, 3.0f, 15.0f },
            { -7.0f, 1.0f, -16.0f }
        } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("a_zp", a_zp),
        convolution("conv", input_info("input"), "weights", "biases", "", { "a_zp" }, "", 1,
                    { 2, 2 }, { 1, 1 }, { 0, 0 }, { 1, 2 }, false, data_types::f32),
        reorder("out", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    auto output_layout = output_memory->get_layout();
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 2);
    ASSERT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_NEAR(output_vec[f][y][x], ((float)output_ptr[f * y_size * x_size + y * x_size + x]), 1e-5f) <<
                " x="<< x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_int8_fw_gpu, quantized_convolution_u8s8f32_asymmetric_activations_per_channel) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 2, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, { 3, 2, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 1, 1 } });
    auto a_zp = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 3, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1,

                                 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });

    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5,

                                   1, 2, -1,
                                   -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                   -1, 3,  2,
                                   0, 2,  5,

                                   1, 2, -1,
                                   -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                   -1, 3,  2,
                                   0, 2,  5 });
    set_values<uint8_t>(a_zp, { 2, 5, 5 });
    set_values(biases, { 1.0f, -8.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { -36.0f, 5.0f, -14.0f },
            { -24.0f, -10.0f, -30.0f }
        },
        {
            { -45.0f, -4.0f, -23.0f },
            { -33.0f, -19.0f, -39.0f }
        },
        {
            { -45.0f, -4.0f, -23.0f },
            { -33.0f, -19.0f, -39.0f }
        } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("a_zp", a_zp),
        convolution("conv", input_info("input"), "weights", "biases", "", { "a_zp" }, "", 1,
                    { 2, 2 }, { 1, 1 }, { 0, 0 }, { 1, 2 }, false, data_types::f32),
        reorder("out", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    auto output_layout = output_memory->get_layout();
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 3);
    ASSERT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_NEAR(output_vec[f][y][x], ((float)output_ptr[f * y_size * x_size + y * x_size + x]), 1e-5f) <<
                " x="<< x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_int8_fw_gpu, quantized_convolution_u8s8f32_asymmetric_activations_per_channel_3ic_with_sub) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 3, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, { 2, 3, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });
    auto a_zp = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 3, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1,

                                 2, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1,

                                 3, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });

    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5,

                                   1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5 });

    set_values<uint8_t>(a_zp, { 2, 5, 6 });
    set_values(biases, { 1.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { -82.0f, -26.0f, -60.0f },
            { -35.0f, -15.0f, -25.0f }
        },
        {
            { -86.0f, -57.0f, -32.0f },
            { -68.0f, -46.0f, -79.0f }
        } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("a_zp", a_zp),
        activation("activation", input_info("input"), activation_func::relu),  // needed just to add padding
        eltwise("in", { input_info("activation"), input_info("a_zp") }, eltwise_mode::sub, data_types::f32),
        convolution("conv", input_info("in"), "weights", "biases", 1, { 2, 2 }, { 1, 1 }, { 0, 0 }, { 1, 2 }, false),
        reorder("out", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    auto output_layout = output_memory->get_layout();
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 2);
    ASSERT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_NEAR(output_vec[f][y][x], ((float)output_ptr[f * y_size * x_size + y * x_size + x]), 1e-5f) <<
                " x="<< x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_int8_fw_gpu, quantized_convolution_u8s8f32_asymmetric_weights_per_channel) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 1, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, { 2, 1, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });
    auto w_zp = engine.allocate_memory({ data_types::i8, format::bfyx, { 2, 1, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });
    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5 });
    set_values<int8_t>(w_zp, { 1, -1 });
    set_values(biases, { 1.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { 30.0f, 44.0f, -9.0f },
            { -4.0f, 2.0f, -2.0f }
        },
        {
            { 42.0f, 69.0f, 43.0f },
            { 23.0f, 31.0f, 2.0f }
        } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("w_zp", w_zp),
        convolution("conv", input_info("input"), "weights", "biases", "w_zp", "", "", 1, { 2, 2 }, { 1, 1 }, { 0, 0 }, { 1, 2 }, false, data_types::f32),
        reorder("out", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    auto output_layout = output_memory->get_layout();
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 2);
    ASSERT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_NEAR(output_vec[f][y][x], ((float)output_ptr[f * y_size * x_size + y * x_size + x]), 1e-5f) <<
                " x="<< x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_gpu, basic_yxfb_4_4_yxfb_2_2_b16_if2_of16_st2_2_p0_sp1_fp16)
{
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const auto input_format   = format::yxfb;
    const auto weights_format = format::yxfb;
    const auto biases_format  = format::bfyx;
    const auto output_format  = input_format;

    const int32_t batch_size = 16;
    const int32_t input_feature_count = 2;
    const int32_t output_feature_count = 16;

    const int32_t stride_x = 2;
    const int32_t stride_y = 2;

    const int32_t input_x = 4;
    const int32_t input_y = 4;
    const int32_t weights_x = 2;
    const int32_t weights_y = 2;
    const int32_t output_x = (input_x - weights_x) / stride_x + 1;
    const int32_t output_y = (input_y - weights_y) / stride_y + 1;

    auto input_size = tensor( batch_size, input_feature_count, input_x, input_y );
    auto input = engine.allocate_memory({ data_types::f32, input_format, input_size });
    auto weights_size = tensor( output_feature_count, input_feature_count, weights_x, weights_y );
    auto weights = engine.allocate_memory({ data_types::f32, weights_format, weights_size });
    auto biases_size = tensor( 1, output_feature_count, 1, 1 );
    auto biases = engine.allocate_memory({ data_types::f32, biases_format, biases_size });
    auto output_size = tensor( batch_size, output_feature_count, output_x, output_y );
    //auto output = memory::allocate({ output_format, { batch_size, { output_x, output_y }, output_feature_count } });

    //auto input_cvtd = engine.allocate_memory({ data_types::f16, input_size });
    //auto weights_cvtd = engine.allocate_memory({ data_types::f16, weights_size });
    //auto biases_cvtd = engine.allocate_memory({ data_types::f16, biases_size });
    //auto output_cvtd  = memory::allocate({ output_cvt_format, { batch_size, { output_x, output_y }, output_feature_count } });

    // input:
    std::vector<float> input_vals_template {
        0.25f, 0.50f, 0.75f, 1.00f,
        1.25f, 1.50f, 1.75f, 2.00f,
        2.25f, 2.50f, 2.75f, 3.00f,
        3.25f, 3.50f, 3.75f, 4.00f,
    };
    input_vals_template.resize(input_y * input_x);

    std::vector<float> input_vals;
    input_vals.reserve(input_y * input_x * input_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < input_y * input_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                input_vals.push_back((bi * input_feature_count + ifi + 1) * input_vals_template[yxi]);
            }
        }
    }
    set_values(input, input_vals);

    // weights:
    std::vector<float> weights_vals_template {
        -0.50f, -0.25f,
         0.50f,  0.50f,
    };
    weights_vals_template.resize(weights_y * weights_x);

    std::vector<float> weights_vals;
    weights_vals.reserve(weights_y * weights_x * input_feature_count * output_feature_count);
    for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
    set_values(weights, weights_vals);

    // biases:
    std::vector<float> biases_vals;
    biases_vals.reserve(output_feature_count);
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        biases_vals.push_back(ofi * 1.0f);
    }
    set_values(biases, biases_vals);

    // output:
    std::vector<float> output_vals_template {
        1.125f,  1.250f,
        1.625f,  1.750f,
    };
    output_vals_template.resize(output_y * output_x);

    std::vector<float> output_vals;
    output_vals.reserve(output_y * output_x * output_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                uint32_t template_factor = input_feature_count * input_feature_count * input_feature_count * bi * ofi +
                    input_feature_count * input_feature_count * (input_feature_count + 1) / 2 * (bi + ofi) +
                    input_feature_count * (input_feature_count + 1) * (2 * input_feature_count + 1) / 6;
                float bias_factor = ofi * 1.0f;

                output_vals.push_back(template_factor * output_vals_template[yxi] + bias_factor);
            }
        }
    }

    //auto expected_float = engine.allocate_memory({ data_types::f32, { format::x, { static_cast<int32_t>(output_vals.size()) } } });
    //auto expected_half  = engine.allocate_memory({ data_types::f16, { format::x, { static_cast<int32_t>(output_vals.size()) } } });
    //auto expected       = engine.allocate_memory({ data_types::f32, { format::x, { static_cast<int32_t>(output_vals.size()) } } });

//    set_values(expected_float, output_vals);
//    auto cvt_expected_f32_f16 = reorder::create({ expected_float, expected_half });
//    auto cvt_expected_f16_f32 = reorder::create({ expected_half, expected });
//    execute({ cvt_expected_f32_f16, cvt_expected_f16_f32 }).wait();
//
//    auto expected_ptr = expected.as<const memory&>().pointer<float>();

    // Computing convolution.
    topology topology(
        input_layout("input", input->get_layout()),
        reorder("cvt_input", input_info("input"), { data_types::f16, input_format, input_size }),
        data("weights", weights),
        reorder("cvt_weights", input_info("weights"), { data_types::f16, weights_format, weights_size }),
        data("biases", biases),
        reorder("cvt_biases", input_info("biases"), { data_types::f16, biases_format, biases_size }),
        convolution("conv", input_info("cvt_input"), "cvt_weights", "cvt_biases", 1, { stride_y, stride_x }, {1, 1}, {0, 0}, {0, 0}, false),
        reorder("output", input_info("conv"), { data_types::f32, output_format, output_size })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());

    // Checking result.
    uint32_t i = 0;
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi, ++i)
            {
                auto equal = are_equal(output_vals[i] /*get_value(expected_ptr, i)*/, output_ptr[i], 0.002f);
                ASSERT_TRUE(equal);
                if (!equal)
                {
                    std::cout << "Failed at position (" << yxi << ", output feature = " << ofi << ", batch = " << bi << "): "
                        << output_vals[i] /*get_value(expected_ptr, i)*/ << " != " << output_ptr[i] << std::endl;
                    return;
                }
            }
        }
    }
}

using TestParamType_convolution_gpu = ::testing::tuple<int,   // 0 - Filter size
                                                       int,   // 1 - Input features
                                                       int,   // 2 - Stride
                                                       int,   // 3 - Output padding
                                                       bool>; // 4 - With bias

using TestParamType_convolution_gpu_block_layout = ::testing::tuple<int,   // 0 -Batch size
        int,  // 1 - Input features
        int,  // 2 - Output features
        int,  // 3 - Filter size
        int,  // 4 - Stride
        int,  // 5 - Output padding
        bool, // 6 - With bias
        int>; // 7 - Input X/Y size


using TestParamType_convolution_depthwise_gpu = ::testing::tuple<int,   // 0 - Input XY size
        int,   // 1 - Kernel sizeY
        int,   // 2 - Kernel sizeX
        int,   // 3 - Groups number
        int,   // 4 - Stride
        int,   // 5 - Output padding
        bool>; // 6 - With bias

using TestParamType_convolution_depthwise_xy_gpu = ::testing::tuple<int,   // 0 - Input Y size
        int,   // 1 - Input X size
        int,   // 2 - Kernel sizeY
        int,   // 3 - Kernel sizeX
        int,   // 4 - Groups number
        int,   // 5 - Stride
        int,   // 6 - Output padding
        bool>; // 7 - With bias

using TestParamType_grouped_convolution_gpu = ::testing::tuple<  int,    // 0 - Input X size
        int,            // 1  - Input Y size
        int,            // 2  - Input Z size
        int,            // 3  - Input features
        int,            // 4  - Output features
        int,            // 5  - Kernel sizeX
        int,            // 6  - Kernel sizeY
        int,            // 7  - Kernel sizeZ
        int,            // 8  - Groups number
        int,            // 9  - Stride
        int,            // 10 - Batch
        bool,           // 11 - Zero points for activations
        bool,           // 12 - Zero points for weights
        bool,           // 13 - Compensation
        format,         // 14 - Input data format
        std::string>;   // 15 - Implementation name

using TestParamType_general_convolution_gpu = ::testing::tuple<  int,    // 0 - Input X size
        int,            // 1  - Input Y size
        int,            // 2  - Input Z size
        int,            // 3  - Input features
        int,            // 4  - Output features
        int,            // 5  - Kernel sizeX
        int,            // 6  - Kernel sizeY
        int,            // 7  - Kernel sizeZ
        int,            // 8  - Groups number
        int,            // 9  - Stride
        int,            // 10 - Batch
        format,         // 11 - Input data format
        std::string,    // 12 - Implementation name
        bool>;          // 13 - With bias

struct convolution_gpu : public ::testing::TestWithParam<TestParamType_convolution_gpu>
{
    static std::string
    PrintToStringParamName(testing::TestParamInfo<TestParamType_convolution_gpu> param_info)
    {
        // construct a readable name
        return std::to_string(testing::get<0>(param_info.param))
            + 'x' + std::to_string(testing::get<0>(param_info.param))
            + "_f" + std::to_string(testing::get<1>(param_info.param))
            + "_stride" + std::to_string(testing::get<2>(param_info.param))
            + "_pad" + std::to_string(testing::get<3>(param_info.param))
            + (testing::get<4>(param_info.param) ? "_bias" : "");
    }
};

struct convolution_gpu_fs_byx_fsv32 : public convolution_gpu {};

struct convolution_gpu_block_layout : public ::testing::TestWithParam<TestParamType_convolution_gpu_block_layout>
{
    static std::string
    PrintToStringParamName(testing::TestParamInfo<TestParamType_convolution_gpu_block_layout> param_info)
    {
        // construct a readable name
        return std::to_string(testing::get<0>(param_info.param))
               + "x_in" + std::to_string(testing::get<1>(param_info.param))
               + "x_out" + std::to_string(testing::get<2>(param_info.param))
               + "_k" + std::to_string(testing::get<3>(param_info.param))
               + "x" + std::to_string(testing::get<3>(param_info.param))
               + "_stride" + std::to_string(testing::get<4>(param_info.param))
               + "_pad" + std::to_string(testing::get<5>(param_info.param))
               + (testing::get<6>(param_info.param) ? "_bias" : "");
    }
};

struct convolution_depthwise_gpu : public ::testing::TestWithParam<TestParamType_convolution_depthwise_gpu>
{
    static std::string
    PrintToStringParamName(testing::TestParamInfo<TestParamType_convolution_depthwise_gpu> param_info)
    {
        // construct a readable name
        return "in" + std::to_string(testing::get<0>(param_info.param))
               + "x" + std::to_string(testing::get<0>(param_info.param))
               + "_k" + std::to_string(testing::get<1>(param_info.param))
               + 'x' + std::to_string(testing::get<2>(param_info.param))
               + "_f" + std::to_string(testing::get<3>(param_info.param))
               + "_stride" + std::to_string(testing::get<4>(param_info.param))
               + "_pad" + std::to_string(testing::get<5>(param_info.param))
               + (testing::get<6>(param_info.param) ? "_bias" : "");
    }
};

struct convolution_depthwise_gpu_fsv16 : public ::testing::TestWithParam<TestParamType_convolution_depthwise_gpu>
{
    static std::string
    PrintToStringParamName(testing::TestParamInfo<TestParamType_convolution_depthwise_gpu> param_info)
    {
        // construct a readable name
        return "in" + std::to_string(testing::get<0>(param_info.param))
            + "x" + std::to_string(testing::get<0>(param_info.param))
            + "_k" + std::to_string(testing::get<1>(param_info.param))
            + 'x' + std::to_string(testing::get<2>(param_info.param))
            + "_f" + std::to_string(testing::get<3>(param_info.param))
            + "_stride" + std::to_string(testing::get<4>(param_info.param))
            + "_pad" + std::to_string(testing::get<5>(param_info.param))
            + (testing::get<6>(param_info.param) ? "_bias" : "");
    }
};

struct convolution_depthwise_gpu_fsv16_xy : public ::testing::TestWithParam<TestParamType_convolution_depthwise_xy_gpu>
{
    static std::string
    PrintToStringParamName(testing::TestParamInfo<TestParamType_convolution_depthwise_xy_gpu> param_info)
    {
        // construct a readable name
        return "in" + std::to_string(testing::get<0>(param_info.param))
            + "x" + std::to_string(testing::get<1>(param_info.param))
            + "_k" + std::to_string(testing::get<2>(param_info.param))
            + 'x' + std::to_string(testing::get<3>(param_info.param))
            + "_f" + std::to_string(testing::get<4>(param_info.param))
            + "_stride" + std::to_string(testing::get<5>(param_info.param))
            + "_pad" + std::to_string(testing::get<6>(param_info.param))
            + (testing::get<7>(param_info.param) ? "_bias" : "");
    }
};

struct convolution_grouped_gpu : public ::testing::TestWithParam<TestParamType_grouped_convolution_gpu> {
    static std::string PrintToStringParamName(
        testing::TestParamInfo<TestParamType_grouped_convolution_gpu> param_info) {
        // construct a readable name
        std::string res = "in" + std::to_string(testing::get<0>(param_info.param)) + "x" +
            std::to_string(testing::get<1>(param_info.param)) + "y" +
            std::to_string(testing::get<2>(param_info.param)) + "z" +
            std::to_string(testing::get<3>(param_info.param)) + "f" +
            "_output" + std::to_string(testing::get<4>(param_info.param)) + "f" +
            "_filter" + std::to_string(testing::get<5>(param_info.param)) + "x" +
            std::to_string(testing::get<6>(param_info.param)) + "y" +
            std::to_string(testing::get<7>(param_info.param)) + "z" +
            "_groups" + std::to_string(testing::get<8>(param_info.param)) +
            "_stride" + std::to_string(testing::get<9>(param_info.param)) +
            "_batch" + std::to_string(testing::get<10>(param_info.param)) +
            "_data_zp" + std::to_string(testing::get<11>(param_info.param)) +
            "_weights_zp" + std::to_string(testing::get<12>(param_info.param)) +
            "_comp" + std::to_string(testing::get<13>(param_info.param)) +
            "_format" + std::to_string(testing::get<14>(param_info.param));

        if (testing::get<15>(param_info.param) != "") {
            res += "_impl_" + testing::get<15>(param_info.param);
        }

        return res;
    }
};

struct convolution_general_gpu : public ::testing::TestWithParam<TestParamType_general_convolution_gpu> {
    static std::string PrintToStringParamName(
        testing::TestParamInfo<TestParamType_general_convolution_gpu> param_info) {
        // construct a readable name
        std::string res = "in" + std::to_string(testing::get<0>(param_info.param)) + "x" +
                          std::to_string(testing::get<1>(param_info.param)) + "y" +
                          std::to_string(testing::get<2>(param_info.param)) + "z" +
                          std::to_string(testing::get<3>(param_info.param)) + "f" + "_output" +
                          std::to_string(testing::get<4>(param_info.param)) + "f" + "_filter" +
                          std::to_string(testing::get<5>(param_info.param)) + "x" +
                          std::to_string(testing::get<6>(param_info.param)) + "y" +
                          std::to_string(testing::get<7>(param_info.param)) + "z" + "_groups" +
                          std::to_string(testing::get<8>(param_info.param)) + "_stride" +
                          std::to_string(testing::get<9>(param_info.param)) + "_batch" +
                          std::to_string(testing::get<10>(param_info.param)) + "_format" +
                          std::to_string(testing::get<11>(param_info.param)) + "_wih_bias_" +
                          std::to_string(testing::get<13>(param_info.param));

        if (testing::get<12>(param_info.param) != "") {
            res += "_impl_" + testing::get<12>(param_info.param);
        }

        return res;
    }
};

INSTANTIATE_TEST_SUITE_P(convolution_gpu_test,
                        convolution_gpu_fs_byx_fsv32,
                        ::testing::Values(
                                // Filter size, Input features, Stride, Output padding, With bias
                                TestParamType_convolution_gpu(1, 20, 1, 0, false),
                                TestParamType_convolution_gpu(3, 80, 1, 0, false),
                                TestParamType_convolution_gpu(1, 32, 1, 0, true),
                                TestParamType_convolution_gpu(3, 32, 1, 0, true),
                                TestParamType_convolution_gpu(1, 32, 1, 1, false),
                                TestParamType_convolution_gpu(3, 32, 1, 1, false),
                                TestParamType_convolution_gpu(1, 32, 2, 0, false),
                                TestParamType_convolution_gpu(3, 32, 2, 0, false),
                                TestParamType_convolution_gpu(3, 64, 2, 1, true)),
                        convolution_gpu::PrintToStringParamName);

TEST_P(convolution_gpu_fs_byx_fsv32, fs_byx_fsv32)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int batch_num = 2;
    const int input_xy = 5;
    const int input_f = testing::get<1>(GetParam());
    const int output_f = 64;
    const int filter_xy = testing::get<0>(GetParam());
    const uint64_t stride = testing::get<2>(GetParam());
    const int output_padding = testing::get<3>(GetParam());
    const bool with_bias = testing::get<4>(GetParam());
    const int pad = filter_xy / 2;

    const int output_xy = 1 + (input_xy + 2 * pad - filter_xy) / stride + 2 * output_padding;

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_xy, input_xy, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, filter_xy, filter_xy, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::bfyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<ov::float16>(batch_num, VVVF<ov::float16>(output_f));

    topology topology(
        input_layout("input", input_mem->get_layout()),
        data("weights_fsv", weights_mem));

    // Reorder input to fs_byx_fsv32
    topology.add(reorder("input_fsv", input_info("input"), { data_types::f16, format::fs_b_yx_fsv32, input_size }));

    if (with_bias)
    {
        // Generate bias data
        auto biases_size = tensor(1, output_f, 1, 1);
        auto biases_data = rg.generate_random_1d<ov::float16>(output_f, -1, 1);
        auto biases_mem = engine.allocate_memory({ data_types::f16, format::bfyx, biases_size });
        set_values(biases_mem, biases_data);

        // Calculate reference values with bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                    input_data[bi], weights_data[ofi],
                    stride, stride,
                    biases_data[ofi],
                    1, 1,                               // dilation
                    pad, pad,                           // input padding
                    output_padding, output_padding);
            }
        }

        topology.add(data("biases_fsv", biases_mem));

        auto conv_fsv = convolution("conv_fsv",
                                    input_info("input_fsv"),
                                     "weights_fsv",
                                     "biases_fsv",
                                     1,
                                     { stride, stride },
                                     {1, 1},
                                     { pad, pad },
                                     { pad, pad },
                                     false);
        conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

        topology.add(conv_fsv);
    }
    else
    {
        // Calculate reference values without bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                    input_data[bi], weights_data[ofi],
                    stride, stride,
                    0,                                  // bias
                    1, 1,                               // dilation
                    pad, pad,                           // input padding
                    output_padding, output_padding);
            }
        }

        auto conv_fsv = convolution("conv_fsv",
                                    input_info("input_fsv"),
                                    "weights_fsv",
                                    no_bias,
                                    1,
                                    { stride, stride },
                                    {1, 1},
                                    { pad, pad },
                                    { pad, pad },
                                    false);
        conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

        topology.add(conv_fsv);
    }


    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl = { format::fs_b_yx_fsv32, "" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_fsv", conv_impl } }));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv_fsv").get_memory();
    cldnn::mem_lock<ov::float16> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(out_mem->get_layout().format, format::fs_b_yx_fsv32);

    for (int bi = 0; bi < batch_num; ++bi)
        for (int fi = 0; fi < output_f; ++fi)
            for (int yi = 0; yi < output_xy; ++yi)
                for (int xi = 0; xi < output_xy; ++xi)
                {
                    auto val_ref = reference_result[bi][fi][yi][xi];
                    auto val = out_ptr[(fi / 32) * batch_num * output_xy * output_xy * 32 +
                                        bi * output_xy * output_xy * 32 +
                                        yi * output_xy * 32 +
                                        xi * 32 +
                                        fi % 32];
                    auto equal = are_equal(val_ref, val, 1e-2f);
                    ASSERT_TRUE(equal);
                    if (!equal)
                    {
                        std::cout << "At b = " << bi << ", fi = " << fi << ", xi = " << xi << ", yi = " << yi << std::endl;
                    }
                }
}

TEST(convolution_f16_fsv_gpu, convolution_f16_fsv_gpu_padding) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int batch_num = 2;
    const int input_xy = 32;
    const int input_f = 96;
    const int output_f = 192;
    const int filter_xy = 1;
    const uint64_t stride = 1;
    const int output_xy = 1 + (input_xy - filter_xy) / stride;

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_xy, input_xy, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, filter_xy, filter_xy, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::bfyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<ov::float16>(batch_num, VVVF<ov::float16>(output_f));

    topology topology(
        input_layout("input", input_mem->get_layout()),
        data("weights_fsv", weights_mem));

    // add input padding by X and Y
    layout w_pad(data_types::f16, format::bfyx, input_size, padding({ 0, 0, 1, 1 }, { 0, 0, 0, 0 }));
    topology.add(reorder("input_fsv", input_info("input"), w_pad));

    // Generate bias data
    auto biases_size = tensor(1, output_f, 1, 1);
    auto biases_data = rg.generate_random_1d<ov::float16>(output_f, -1, 1);
    auto biases_mem = engine.allocate_memory({ data_types::f16, format::bfyx, biases_size });
    set_values(biases_mem, biases_data);

    // Calculate reference values
    for (auto bi = 0; bi < batch_num; ++bi)
    {
        for (auto ofi = 0; ofi < output_f; ++ofi)
        {
            reference_result[bi][ofi] = reference_convolve(
                input_data[bi], weights_data[ofi],
                stride, stride,
                biases_data[ofi],
                1, 1);
        }
    }

    topology.add(data("biases_fsv", biases_mem));

    auto conv_fsv = convolution("conv_fsv",
                                input_info("input_fsv"),
                                "weights_fsv",
                                "biases_fsv",
                                1,
                                { stride, stride },
                                {1, 1},
                                { 0, 0 },
                                { 0, 0 },
                                false);

    topology.add(conv_fsv);

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl = { format::fs_b_yx_fsv32, "convolution_gpu_bfyx_to_fs_byx_fsv32" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_fsv", conv_impl } }));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv_fsv").get_memory();
    cldnn::mem_lock<ov::float16> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(out_mem->get_layout().format, format::fs_b_yx_fsv32);

    for (int bi = 0; bi < batch_num; ++bi)
        for (int fi = 0; fi < output_f; ++fi)
            for (int yi = 0; yi < output_xy; ++yi)
                for (int xi = 0; xi < output_xy; ++xi)
                {
                    auto val_ref = reference_result[bi][fi][yi][xi];
                    auto val = out_ptr[(fi / 32) * batch_num * output_xy * output_xy * 32 +
                        bi * output_xy * output_xy * 32 +
                        yi * output_xy * 32 +
                        xi * 32 +
                        fi % 32];
                    auto equal = are_equal(val_ref, val, 1e-2f);
                    ASSERT_TRUE(equal);
                    if (!equal)
                    {
                        std::cout << "At b = " << bi << ", fi = " << fi << ", xi = " << xi << ", yi = " << yi << std::endl;
                    }
                }
}

using TestParamType_convolution_gpu_with_crop = ::testing::tuple<int,   // 0 - Filter size
    int,   // 1 - Input size
    int,   // 2 - Input/output features
    int,   // 3 - Stride
    int,   // 4 - Output padding
    bool>; // 5 - With bias

struct convolution_gpu_fs_byx_fsv32_crop : public ::testing::TestWithParam<TestParamType_convolution_gpu_with_crop>
{
    static std::string
        PrintToStringParamName(testing::TestParamInfo<TestParamType_convolution_gpu_with_crop> param_info)
    {
        // construct a readable name
        return std::to_string(testing::get<0>(param_info.param))
            + 'x' + std::to_string(testing::get<1>(param_info.param))
            + "_f" + std::to_string(testing::get<2>(param_info.param))
            + "_stride" + std::to_string(testing::get<3>(param_info.param))
            + "_pad" + std::to_string(testing::get<4>(param_info.param))
            + (testing::get<5>(param_info.param) ? "_bias" : "");
    }
};

INSTANTIATE_TEST_SUITE_P(convolution_gpu_with_crop,
    convolution_gpu_fs_byx_fsv32_crop,
    ::testing::Values(
        TestParamType_convolution_gpu_with_crop(1, 14, 24, 1, 0, true)
    ),
    convolution_gpu_fs_byx_fsv32_crop::PrintToStringParamName);

TEST_P(convolution_gpu_fs_byx_fsv32_crop, fs_byx_fsv32_crop)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int batch_num = 4;
    const int filter_xy = testing::get<0>(GetParam());
    const int input_xy = testing::get<1>(GetParam());
    const int input_f = testing::get<2>(GetParam());
    const int output_f = input_f;
    const uint64_t stride = testing::get<3>(GetParam());
    const int output_padding = testing::get<4>(GetParam());
    const bool with_bias = testing::get<5>(GetParam());

    const int pad = filter_xy / 2;
    const int output_xy = 1 + (input_xy + 2 * pad - filter_xy) / stride + 2 * output_padding;

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, filter_xy, filter_xy, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::bfyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // ref input
    auto half_input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_xy, input_xy, -1, 1);
    auto input_data = VVVVF<ov::float16>(batch_num);

    // concatenated cldnn input tensor
    for (auto bi = 0; bi < batch_num; ++bi)
    {
        for (auto ifi = 0; ifi < input_f; ++ifi)
        {
            auto fdata = half_input_data[bi][ifi];
            input_data[bi].emplace_back(std::move(fdata));
        }
        for (auto ifi = 0; ifi < input_f; ++ifi)
        {
            auto fdata = half_input_data[bi][ifi];
            input_data[bi].emplace_back(std::move(fdata));
        }
    }

    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_size = tensor(batch_num, input_f * 2, input_xy, input_xy);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    topology topology(
        input_layout("input", input_mem->get_layout()),
        data("weights_fsv", weights_mem));

    auto crop_batch_num = batch_num;
    auto crop_feature_num = input_f;
    auto crop_x_size = input_xy;
    auto crop_y_size = input_xy;

    auto left_crop = crop("left_crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 });
    auto right_crop = crop("right_crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, input_f, 0, 0 });
    topology.add(left_crop);
    topology.add(right_crop);

    // Will be used to store reference values calculated in branches depending on bias
    auto half_ref_result = VVVVF<ov::float16>(batch_num, VVVF<ov::float16>(output_f));

    // reference convolution and concat
    if (with_bias)
    {
        // Generate bias data
        auto biases_size = tensor(1, output_f, 1, 1);
        auto biases_data = rg.generate_random_1d<ov::float16>(output_f, -1, 1);
        auto biases_mem = engine.allocate_memory({ data_types::f16, format::bfyx, biases_size });
        set_values(biases_mem, biases_data);

        // Calculate reference values with bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                half_ref_result[bi][ofi] = reference_convolve(
                    half_input_data[bi], weights_data[ofi],
                    stride, stride,
                    biases_data[ofi],
                    1, 1,                               // dilation
                    pad, pad,                           // input padding
                    output_padding, output_padding);
            }
        }

        topology.add(data("biases_fsv", biases_mem));

        auto conv_fsv = convolution("conv_fsv",
                                    input_info("right_crop"),
                                    "weights_fsv",
                                    "biases_fsv",
                                    1,
                                    { stride, stride },
                                    {1, 1},
                                    { pad, pad },
                                    { pad, pad },
                                    false);
        conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};
        topology.add(conv_fsv);
    }
    else
    {
        // Calculate reference values without bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                half_ref_result[bi][ofi] = reference_convolve(
                    half_input_data[bi], weights_data[ofi],
                    stride, stride,
                    0,                                  // bias
                    1, 1,                               // dilation
                    pad, pad,                           // input padding
                    output_padding, output_padding);
            }
        }

        auto conv_fsv = convolution("conv_fsv",
                                    input_info("right_crop"),
                                    "weights_fsv",
                                    no_bias,
                                    1,
                                    { stride, stride },
                                    {1, 1},
                                    { pad, pad },
                                    { pad, pad },
                                    false);
        conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};
        topology.add(conv_fsv);
    }


    topology.add(reorder("reorder", input_info("conv_fsv"), { data_types::f16, format::bfyx, input_size }));
    topology.add(concatenation("concat", { input_info("left_crop"), input_info("reorder") }, 1));

    auto ref_result = VVVVF<ov::float16>(batch_num);
    // concatenate half ref input and ref conv output, by features
    for (auto bi = 0; bi < batch_num; ++bi)
    {
        for (auto ifi = 0; ifi < input_f; ++ifi)
        {
            auto fdata = half_input_data[bi][ifi];
            ref_result[bi].emplace_back(std::move(fdata));
        }
        for (auto ifi = 0; ifi < input_f; ++ifi)
        {
            auto fdata = half_ref_result[bi][ifi];
            ref_result[bi].emplace_back(std::move(fdata));
        }
    }

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl = { format::fs_b_yx_fsv32, "convolution_gpu_bfyx_to_fs_byx_fsv32" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_fsv", conv_impl } }));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("concat").get_memory();
    cldnn::mem_lock<ov::float16> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(out_mem->get_layout().format, format::bfyx);

    for (int bi = 0; bi < batch_num; ++bi)
        for (int fi = 0; fi < output_f * 2; ++fi)
            for (int yi = 0; yi < output_xy; ++yi)
                for (int xi = 0; xi < output_xy; ++xi)
                {
                    auto val_ref = ref_result[bi][fi][yi][xi];
                    auto val = out_ptr[bi * output_xy * output_xy * output_f * 2 +
                        fi * output_xy * output_xy +
                        yi * output_xy +
                        xi];
                    auto equal = are_equal(val_ref, val, 1e-2f);
                    ASSERT_TRUE(equal);
                    if (!equal)
                    {
                        std::cout << "At b = " << bi << ", fi = " << fi << ", xi = " << xi << ", yi = " << yi << std::endl;
                    }
                }
}

TEST(convolution_f32_fw_gpu, convolution_int8_b_fs_yx_fsv4_to_bfyx) {

    const int batch_num = 1;
    const int output_f = 12;
    const int input_f = 16;
    const int filter_xy = 5;
    const int input_padding = 2;
    const int output_padding = 2;
    const int input_size_x = 1280;
    const int input_size_y = 720;

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (engine.get_device_info().supports_immad) {
        // Currently, oneDNN does NOT support input/output padding.(And oneDNN does NOT use fsv4 for this tensor.)
        return;
    }

    auto input_size = tensor(batch_num, input_f, input_size_x, input_size_y);
    auto input_data = rg.generate_random_4d<float>(batch_num, input_f, input_size_y, input_size_x, -10, 10);

    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, input_size });
    set_values(input, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy);
    auto weights_data = rg.generate_random_4d<int8_t>(output_f, input_f, filter_xy, filter_xy, -10, 10);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, weights_size });
    set_values(weights, weights_data_bfyx);

    auto biases_size = tensor(1, output_f, 1, 1);
    auto biases_data = rg.generate_random_4d<int8_t>(1, output_f, 1, 1, -10, 10);
    auto biases_data_bfyx = flatten_4d(format::bfyx, biases_data);
    auto biases = engine.allocate_memory({ data_types::i8, format::bfyx, biases_size });
    set_values(biases, biases_data_bfyx);

    topology topology_ref(
        input_layout("input", input->get_layout()),
        reorder("to_int", input_info("input"), { data_types::i8, format::bfyx, { batch_num, input_f, input_size_x, input_size_y } }),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("to_int"), "weights", "biases", 1, { 1, 1 }, { 1, 1 }, { input_padding, input_padding }, { input_padding, input_padding }, false,
                    ov::op::PadType::EXPLICIT, padding{ { 0, 0, output_padding, output_padding }, 0 }),
        reorder("output", input_info("conv"), { data_types::f32, format::bfyx, { batch_num, input_f, input_size_x, input_size_y } }));

    ExecutionConfig config_ref = get_test_default_config(engine);

    network network_ref(engine, topology_ref, config_ref);
    network_ref.set_input_data("input", input);

    auto outputs = network_ref.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    topology topology_act(
        input_layout("input", input->get_layout()),
        reorder("to_int", input_info("input"), { data_types::i8,format::b_fs_yx_fsv4, { batch_num, input_f, input_size_x, input_size_y } }),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", input_info("to_int"), "weights", "biases", 1, { 1, 1 }, { 1, 1 }, { input_padding, input_padding }, { input_padding, input_padding }, false,
            ov::op::PadType::EXPLICIT, padding{ { 0, 0, output_padding, output_padding }, 0 }),
        reorder("output", input_info("conv"), { data_types::f32,format::bfyx, { batch_num, input_f, input_size_x, input_size_y } }));

    ExecutionConfig config_act = get_test_default_config(engine);

    config_act.set_property(ov::intel_gpu::optimize_data(true));

    network network_act(engine, topology_act, config_act);
    network_act.set_input_data("input", input);

    auto outputs_act = network_act.execute();
    ASSERT_EQ(outputs_act.size(), size_t(1));
    ASSERT_EQ(outputs_act.begin()->first, "output");

    auto output_memory_act = outputs_act.at("output").get_memory();
    cldnn::mem_lock<float> output_act_ptr(output_memory_act, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();

    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 720);
    ASSERT_EQ(x_size, 1280);
    ASSERT_EQ(f_size, output_f);
    ASSERT_EQ(b_size, 1);
    for (int o = 0; o < f_size; ++o) {
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_EQ(output_act_ptr[o * x_size * y_size + y * x_size + x], output_ptr[o * x_size * y_size + y * x_size + x]);
            }
        }
    }
}

TEST(convolution_gpu, bfyx_iyxo_5x5_fp16)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int batch_num = 1;
    const int output_f = 4;

    const int input_f = 32;
    const int filter_xy = 5;
    const uint64_t stride = 1;
    const int output_padding = 0;
    const bool with_bias = false;
    const int input_size_x = 64;
    const int input_size_y = 20;


    const int pad = filter_xy / 2;

    const int output_x = 1 + (input_size_x + 2 * pad - filter_xy) / stride + 2 * output_padding;

    const int output_y = 1 + (input_size_y + 2 * pad - filter_xy) / stride + 2 * output_padding;

    auto input_size = tensor(batch_num, input_f, input_size_x, input_size_y);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_size_y, input_size_x, -1, 1);

    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, filter_xy, filter_xy, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::bfyx, weights_size });

    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<ov::float16>(batch_num, VVVF<ov::float16>(output_f));

    topology topology(
        input_layout("input", input_mem->get_layout()),
        data("weights_fsv", weights_mem)
    );

    if (with_bias)
    {
        // Generate bias data
        auto biases_size = tensor(1, output_f, 1, 1);
        auto biases_data = rg.generate_random_1d<ov::float16>(output_f, -1, 1);
        auto biases_mem = engine.allocate_memory({ data_types::f16, format::bfyx, biases_size });
        set_values(biases_mem, biases_data);

        // Calculate reference values with bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                    input_data[bi], weights_data[ofi],
                    stride, stride, biases_data[ofi],
                    1, 1,                               // dilation
                    pad, pad,                           // input padding
                    output_padding, output_padding);
            }
        }

        topology.add(data("biases_fsv", biases_mem));

        auto conv_fsv = convolution("conv_fsv", input_info("input"), "weights_fsv", "biases_fsv",
                                    1, { stride, stride }, {1, 1}, { pad, pad }, { pad, pad }, false);
        conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

        topology.add(conv_fsv);
    }
    else
    {

        // Calculate reference values without bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                    input_data[bi], weights_data[ofi],
                    stride, stride,
                    0,                                  // bias
                    1, 1,                               // dilation
                    pad, pad,                           // input padding
                    output_padding, output_padding);
            }
        }


        auto conv_fsv = convolution("conv_fsv", input_info("input"), "weights_fsv", no_bias,
            1, { stride, stride }, {1, 1}, { pad, pad }, { pad, pad }, false);
        conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

        topology.add(conv_fsv);
        topology.add(reorder("out", input_info("conv_fsv"), format::bfyx, data_types::f16));
    }


    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl = { format::bfyx, "" };
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("out").get_memory();
    cldnn::mem_lock<ov::float16> out_ptr(out_mem, get_test_stream());

    auto output_layout = out_mem->get_layout();
    ASSERT_EQ(output_layout.format, format::bfyx);

    for (int bi = 0; bi < batch_num; ++bi) {
        for (int fi = 0; fi < output_f; ++fi) {
            for (int yi = 0; yi < output_y; ++yi) {
                for (int xi = 0; xi < output_x; ++xi) {
                    auto val_ref = reference_result[bi][fi][yi][xi];
                    auto val = out_ptr[bi * output_f * output_x * output_y +
                                        fi * output_y * output_x  +
                                        yi * output_x +
                                        xi];

                    auto equal = are_equal(val_ref, val, 1e-2f);
                    ASSERT_TRUE(equal);
                }
            }
        }
    }
}

template<typename T>
void blockedFormatZeroCheck(cldnn::memory::ptr out_mem) {
    cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

    bool batch_blocked = false;
    if (out_mem->get_layout().format == format::bs_fs_zyx_bsv16_fsv16 ||
        out_mem->get_layout().format == format::bs_fs_yx_bsv16_fsv16)
        batch_blocked = true;
    const int block_size = 16;

    auto output_tensor = out_mem->get_layout().get_buffer_size();
    const int b = output_tensor.batch[0];
    const int f = output_tensor.feature[0];
    const int spatials = std::accumulate(output_tensor.spatial.begin(), output_tensor.spatial.end(), 1, std::multiplies<int>());
    const int f_mod = output_tensor.feature[0] % block_size;
    const size_t batch_skip = batch_blocked ? b / block_size : b;
    const size_t number_of_zeroes = f_mod == 0 ? 0 : (block_size - f_mod) * spatials * b;

    size_t to_skip = (output_tensor.feature[0] / block_size) * block_size * spatials;
    to_skip *= batch_blocked ? block_size : 1;
    size_t zero_ind = to_skip + f_mod;

    size_t i = 0;
    while (i < number_of_zeroes) {
        size_t f_tmp = f_mod;
        while (f_tmp % 16 != 0) {
            auto equal = are_equal(out_ptr[zero_ind], 0, 1e-2f);
            ASSERT_TRUE(equal);
            if (!equal) {
                std::cout << "Should be zero idx: " << zero_ind << std::endl;
                return;
            }
            f_tmp++;
            zero_ind++;
            i++;
        }
        // skip on new batch
        if (i % (number_of_zeroes / batch_skip) == 0)
            zero_ind += to_skip;
        if (zero_ind >= (size_t) b * f * spatials)
            return;

        zero_ind += f_mod;
    }
}
struct convolution_gpu_block_layout3D : public convolution_gpu_block_layout {};
INSTANTIATE_TEST_SUITE_P(convolution_gpu_block3D,
                        convolution_gpu_block_layout3D,
                        ::testing::Values(
                                TestParamType_convolution_gpu_block_layout(1, 3, 10, 1, 1, 0, false, 16),
                                TestParamType_convolution_gpu_block_layout(4, 4, 17, 3, 1, 0, false, 16),
                                TestParamType_convolution_gpu_block_layout(15, 17, 15, 3, 1, 0, false, 4),
                                TestParamType_convolution_gpu_block_layout(1, 17, 16, 1, 1, 0, false, 4),
                                TestParamType_convolution_gpu_block_layout(1, 17, 20, 3, 1, 0, false, 9),
                                TestParamType_convolution_gpu_block_layout(4, 15, 17, 1, 1, 0, false, 5),
                                TestParamType_convolution_gpu_block_layout(1, 2, 15, 3, 1, 0, true, 16),
                                TestParamType_convolution_gpu_block_layout(17, 2, 16, 3, 1, 0, true, 32),
                                TestParamType_convolution_gpu_block_layout(30, 2, 5, 1, 1, 0, true, 8),
                                TestParamType_convolution_gpu_block_layout(2, 1, 7, 3, 1, 0, true, 8),
                                TestParamType_convolution_gpu_block_layout(5, 16, 1, 1, 1, 0, true, 5),

                                TestParamType_convolution_gpu_block_layout(32, 4, 15, 1, 1, 0, false, 5),
                                TestParamType_convolution_gpu_block_layout(32, 2, 16, 3, 1, 0, false, 8),
                                TestParamType_convolution_gpu_block_layout(32, 4, 17, 3, 1, 0, false, 1),
                                TestParamType_convolution_gpu_block_layout(32, 17, 15, 1, 1, 0, false, 16),
                                TestParamType_convolution_gpu_block_layout(32, 15, 15, 1, 1, 0, true, 32),
                                TestParamType_convolution_gpu_block_layout(32, 2, 1, 3, 3, 0, true, 10),
                                TestParamType_convolution_gpu_block_layout(32, 17, 1, 1, 1, 0, true, 8),

                                TestParamType_convolution_gpu_block_layout(16, 1, 17, 1, 1, 0, true, 10),
                                TestParamType_convolution_gpu_block_layout(16, 3, 15, 1, 1, 0, false, 8),
                                TestParamType_convolution_gpu_block_layout(16, 1, 17, 3, 1, 0, false, 1),
                                TestParamType_convolution_gpu_block_layout(16, 15, 17, 1, 1, 0, false, 3),
                                TestParamType_convolution_gpu_block_layout(16, 16, 17, 3, 1, 0, false, 16),
                                TestParamType_convolution_gpu_block_layout(16, 17, 15, 3, 1, 0, false, 32),
                                TestParamType_convolution_gpu_block_layout(16, 17, 16, 1, 1, 0, true, 17),
                                TestParamType_convolution_gpu_block_layout(16, 3, 7, 3, 1, 0, true, 8),
                                TestParamType_convolution_gpu_block_layout(16, 5, 10, 1, 1, 0, true, 8),
                                TestParamType_convolution_gpu_block_layout(16, 17, 6, 3, 1, 0, true, 15),
                                TestParamType_convolution_gpu_block_layout(16, 16, 1, 1, 1, 0, true, 2),

                                TestParamType_convolution_gpu_block_layout(16, 32, 16, 3, 1, 0, false, 5),
                                TestParamType_convolution_gpu_block_layout(16, 16, 32, 3, 1, 0, false, 3),
                                TestParamType_convolution_gpu_block_layout(32, 16, 32, 3, 1, 0, false, 9),
                                TestParamType_convolution_gpu_block_layout(16, 32, 32, 3, 1, 0, false, 8),
                                TestParamType_convolution_gpu_block_layout(16, 64, 16, 3, 1, 0, false, 5),
                                TestParamType_convolution_gpu_block_layout(32, 32, 16, 3, 1, 0, false, 5),
                                TestParamType_convolution_gpu_block_layout(32, 32, 32, 3, 1, 0, false, 3),
                                TestParamType_convolution_gpu_block_layout(32, 32, 32, 1, 1, 0, false, 2),
                                TestParamType_convolution_gpu_block_layout(32, 16, 16, 3, 1, 0, true, 16),
                                TestParamType_convolution_gpu_block_layout(16, 32, 16, 3, 1, 0, true, 5),
                                TestParamType_convolution_gpu_block_layout(32, 16, 32, 3, 1, 0, true, 5),
                                TestParamType_convolution_gpu_block_layout(16, 64, 16, 3, 1, 0, true, 3),
                                TestParamType_convolution_gpu_block_layout(32, 32, 16, 3, 1, 0, true, 2),
                                TestParamType_convolution_gpu_block_layout(32, 32, 32, 3, 1, 0, true, 2),
                                TestParamType_convolution_gpu_block_layout(64, 64, 64, 3, 1, 0, true, 2)),
                        convolution_gpu_block_layout::PrintToStringParamName);

TEST_P(convolution_gpu_block_layout3D, bfzyx_bsv16_fsv16_fp32)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int batch_num = testing::get<0>(GetParam());
    const int input_f = testing::get<1>(GetParam());
    const int output_f = testing::get<2>(GetParam());
    const int filter_xy = testing::get<3>(GetParam());
    const uint64_t stride = testing::get<4>(GetParam());
    const int output_padding = testing::get<5>(GetParam());
    const bool with_bias = testing::get<6>(GetParam());
    const int input_xy = testing::get<7>(GetParam());
    const int pad = filter_xy / 2;
    format input_format = format::b_fs_zyx_fsv16;
    if (batch_num % 16 == 0)
        input_format = format::bs_fs_zyx_bsv16_fsv16;

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy, 1);
    auto input_data = rg.generate_random_4d<float>(batch_num, input_f, input_xy, input_xy, 1, 10);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfzyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy, 1);
    auto weights_data = rg.generate_random_4d<float>(output_f, input_f, filter_xy, filter_xy, -1, 1);

    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    auto weights_mem = engine.allocate_memory({ data_types::f32, format::bfzyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<float>(batch_num, VVVF<float>(output_f));

    topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem));

    // Reorder input to correct format
    topology.add(reorder("input_bsv16_fsv16", input_info("input"), { data_types::f32, input_format, input_size }));

    if (with_bias)
    {
        // Generate bias data
        auto biases_size = tensor(1, output_f, 1, 1, 1);
        auto biases_data = rg.generate_random_1d<float>(output_f, -1, 1);
        auto biases_mem = engine.allocate_memory({ data_types::f32, format::bfzyx, biases_size });
        set_values(biases_mem, biases_data);

        // Calculate reference values with bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                        input_data[bi], weights_data[ofi],
                        stride, stride, biases_data[ofi],
                        1, 1,                               // dilation
                        pad, pad,                           // input padding
                        output_padding, output_padding);
            }
        }

        topology.add(data("biases", biases_mem));

        auto conv_bsv16_fsv16 = convolution("conv_bsv16_fsv16", input_info("input_bsv16_fsv16"), "weights", "biases",
                                    1, { 1, stride, stride }, {1, 1, 1}, { 0, pad, pad }, { 0, pad, pad }, false);
        conv_bsv16_fsv16.output_paddings = {padding({ 0, 0, output_padding, output_padding, 0 }, 0.f)};

        topology.add(conv_bsv16_fsv16);
    }
    else
    {
        // Calculate reference values without bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                        input_data[bi], weights_data[ofi],
                        stride, stride,
                        0,                                  // bias
                        1, 1,                               // dilation
                        pad, pad,                           // input padding
                        output_padding, output_padding);
            }
        }

        auto conv_bsv16_fsv16 = convolution("conv_bsv16_fsv16", input_info("input_bsv16_fsv16"), "weights", no_bias,
                                    1, { 1, stride, stride }, {1, 1, 1}, { 0, pad, pad }, { 0, pad, pad }, false);
        conv_bsv16_fsv16.output_paddings = {padding({ 0, 0, output_padding, output_padding, 0 }, 0.f)};

        topology.add(conv_bsv16_fsv16);
    }

    topology.add(reorder("reorder_bfzyx", input_info("conv_bsv16_fsv16"), format::bfzyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "conv_bsv16_fsv16", "reorder_bfzyx", }));
    ov::intel_gpu::ImplementationDesc conv_impl = { input_format, "", impl_types::ocl };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{ "conv_bsv16_fsv16", conv_impl }}));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv_bsv16_fsv16").get_memory();
    cldnn::mem_lock<float> out_ptr(out_mem, get_test_stream());

    auto out_mem_bfyx = network.get_output("reorder_bfzyx").get_memory();
    cldnn::mem_lock<float> out_ptr_bfyx(out_mem_bfyx, get_test_stream());

    blockedFormatZeroCheck<float>(out_mem);

    ASSERT_EQ(out_mem->get_layout().format, input_format);

    auto flatten_ref = flatten_4d(format::bfyx, reference_result);

    for (size_t i = 0; i < out_ptr_bfyx.size(); i++) {
        auto equal = are_equal(flatten_ref[i], out_ptr_bfyx[i], 1e-2f);
        ASSERT_TRUE(equal);
        if (!equal)
        {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }

}

TEST_P(convolution_gpu_block_layout3D, bfzyx_bsv16_fsv16_fp16)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int batch_num = testing::get<0>(GetParam());
    const int input_f = testing::get<1>(GetParam());
    const int output_f = testing::get<2>(GetParam());
    const int filter_xy = testing::get<3>(GetParam());
    const uint64_t stride = testing::get<4>(GetParam());
    const int output_padding = testing::get<5>(GetParam());
    const bool with_bias = testing::get<6>(GetParam());
    const int input_xy = testing::get<7>(GetParam());
    const int pad = filter_xy / 2;
    format input_format = format::b_fs_zyx_fsv16;
    if (batch_num % 32 == 0)
        input_format = format::bs_fs_zyx_bsv16_fsv16;

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy, 1);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_xy, input_xy, 0, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);

    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfzyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, filter_xy, filter_xy, 0, 1);

    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    auto weights_mem = engine.allocate_memory({ data_types::f16, format::bfzyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<ov::float16>(batch_num, VVVF<ov::float16>(output_f));

    topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem));

    // Reorder input to correct format
    topology.add(reorder("input_bsv16_fsv16", input_info("input"), { data_types::f16, input_format, input_size }));

    if (with_bias)
    {
        // Generate bias data
        auto biases_size = tensor(1, output_f, 1, 1, 1);
        auto biases_data = rg.generate_random_1d<ov::float16>(output_f, -1, 1);
        auto biases_mem = engine.allocate_memory({ data_types::f16, format::bfzyx, biases_size });
        set_values(biases_mem, biases_data);

        // Calculate reference values with bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                        input_data[bi], weights_data[ofi],
                        stride, stride, biases_data[ofi],
                        1, 1,                               // dilation
                        pad, pad,                           // input padding
                        output_padding, output_padding);
            }
        }

        topology.add(data("biases", biases_mem));

        auto conv_bsv16_fsv16 = convolution("conv_bsv16_fsv16", input_info("input_bsv16_fsv16"), "weights", "biases",
                                        1, { 1, stride, stride }, {1, 1, 1}, { 0, pad, pad }, { 0, pad, pad }, false);
        conv_bsv16_fsv16.output_paddings = {padding({ 0, 0, output_padding, output_padding, 0 }, 0.f)};

        topology.add(conv_bsv16_fsv16);
    }
    else
    {
        // Calculate reference values without bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                        input_data[bi], weights_data[ofi],
                        stride, stride,
                        0,                                  // bias
                        1, 1,                               // dilation
                        pad, pad,                           // input padding
                        output_padding, output_padding);
            }
        }

        auto conv_bsv16_fsv16 = convolution("conv_bsv16_fsv16", input_info("input_bsv16_fsv16"), "weights", no_bias,
                                        1, { 1, stride, stride }, {1, 1, 1}, { 0, pad, pad }, { 0, pad, pad }, false);
        conv_bsv16_fsv16.output_paddings = {padding({ 0, 0, output_padding, output_padding, 0 }, 0.f)};

        topology.add(conv_bsv16_fsv16);
    }

    topology.add(reorder("reorder_bfzyx", input_info("conv_bsv16_fsv16"), format::bfzyx, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "conv_bsv16_fsv16", "reorder_bfzyx" }));
    ov::intel_gpu::ImplementationDesc conv_impl = { input_format, "" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{ "conv_bsv16_fsv16", conv_impl }}));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv_bsv16_fsv16").get_memory();
    cldnn::mem_lock<ov::float16> out_ptr(out_mem, get_test_stream());

    auto out_mem_bfyx = network.get_output("reorder_bfzyx").get_memory();
    cldnn::mem_lock<ov::float16> out_ptr_bfyx(out_mem_bfyx, get_test_stream());

    blockedFormatZeroCheck<ov::float16>(out_mem);

    ASSERT_EQ(out_mem->get_layout().format, input_format);

    auto flatten_ref = flatten_4d(format::bfyx, reference_result);

    for (size_t i = 0; i < out_ptr_bfyx.size(); i++) {
        auto equal = are_equal(flatten_ref[i], out_ptr_bfyx[i], 1);
        ASSERT_TRUE(equal);
        if (!equal)
        {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }
}

TEST_P(convolution_gpu_block_layout3D, bfzyx_bsv16_fsv16_fp32_fused_ops)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int batch_num = testing::get<0>(GetParam());
    const int input_f = testing::get<1>(GetParam());
    const int output_f = testing::get<2>(GetParam());
    const int filter_xy = testing::get<3>(GetParam());
    const uint64_t stride = testing::get<4>(GetParam());
    const int output_padding = testing::get<5>(GetParam());
    const bool with_bias = testing::get<6>(GetParam());
    const int input_xy = testing::get<7>(GetParam());
    const int pad = filter_xy / 2;
    format input_format = format::b_fs_zyx_fsv16;
    if (batch_num % 16 == 0)
        input_format = format::bs_fs_zyx_bsv16_fsv16;

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy, 1);
    auto input_data = rg.generate_random_4d<float>(batch_num, input_f, input_xy, input_xy, 1, 10);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfzyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy, 1);
    auto weights_data = rg.generate_random_4d<float>(output_f, input_f, filter_xy, filter_xy, -1, 1);

    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    auto weights_mem = engine.allocate_memory({ data_types::f32, format::bfzyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<float>(batch_num, VVVF<float>(output_f));

    topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem));

    // Reorder input to correct format
    topology.add(reorder("input_bsv16_fsv16", input_info("input"), { data_types::f32, input_format, input_size }));

    if (with_bias)
    {
        // Generate bias data
        auto biases_size = tensor(1, output_f, 1, 1, 1);
        auto biases_data = rg.generate_random_1d<float>(output_f, -1, 1);
        auto biases_mem = engine.allocate_memory({ data_types::f32, format::bfzyx, biases_size });
        set_values(biases_mem, biases_data);

        // Calculate reference values with bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                        input_data[bi], weights_data[ofi],
                        stride, stride, biases_data[ofi],
                        1, 1,                               // dilation
                        pad, pad,                           // input padding
                        output_padding, output_padding);
            }
        }

        topology.add(data("biases", biases_mem));

        auto conv_bsv16_fsv16 = convolution("conv_bsv16_fsv16", input_info("input_bsv16_fsv16"), "weights", "biases",
                                       1, { 1, stride, stride }, {1, 1, 1}, { 0, pad, pad }, { 0, pad, pad }, false);
        conv_bsv16_fsv16.output_paddings = {padding({ 0, 0, output_padding, output_padding, 0 }, 0.f)};

        topology.add(conv_bsv16_fsv16);
    }
    else
    {
        // Calculate reference values without bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                        input_data[bi], weights_data[ofi],
                        stride, stride,
                        0,                                  // bias
                        1, 1,                               // dilation
                        pad, pad,                           // input padding
                        output_padding, output_padding);
            }
        }

        auto conv_bsv16_fsv16 = convolution("conv_bsv16_fsv16", input_info("input_bsv16_fsv16"), "weights", no_bias,
                                       1, { 1, stride, stride }, {1, 1, 1}, { 0, pad, pad }, { 0, pad, pad }, false);
        conv_bsv16_fsv16.output_paddings = {padding({ 0, 0, output_padding, output_padding, 0 }, 0.f)};

        topology.add(conv_bsv16_fsv16);
    }

    const float scalar = 5.5f;
    auto scale_mem = engine.allocate_memory({ data_types::f32, format::bfzyx, { 1, 1, 1, 1, 1 } });
    set_values(scale_mem, { scalar });

    topology.add(data("scalar", scale_mem));
    topology.add(eltwise("scale", { input_info("conv_bsv16_fsv16"), input_info("scalar") }, eltwise_mode::prod));

    topology.add(reorder("reorder_bfzyx", input_info("scale"), format::bfzyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "conv_bsv16_fsv16", "reorder_bfzyx" }));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv_bsv16_fsv16").get_memory();
    cldnn::mem_lock<float> out_ptr(out_mem, get_test_stream());

    auto out_mem_bfyx = network.get_output("reorder_bfzyx").get_memory();
    cldnn::mem_lock<float> out_ptr_bfyx(out_mem_bfyx, get_test_stream());

    blockedFormatZeroCheck<float>(out_mem);

    ASSERT_EQ(out_mem->get_layout().format, input_format);

    auto flatten_ref = flatten_4d(format::bfyx, reference_result);

    for (size_t i = 0; i < out_ptr_bfyx.size(); i++) {
        auto equal = are_equal(flatten_ref[i] * scalar, out_ptr_bfyx[i], 1e-2f);
        ASSERT_TRUE(equal);
        if (!equal)
        {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(convolution_gpu_block,
                        convolution_gpu_block_layout,
                        ::testing::Values(
                                TestParamType_convolution_gpu_block_layout(16, 64, 64, 1, 1, 0, false, 0),
                                TestParamType_convolution_gpu_block_layout(16, 16, 16, 3, 1, 0, false, 0),
                                TestParamType_convolution_gpu_block_layout(32, 16, 16, 3, 1, 0, false, 0),
                                TestParamType_convolution_gpu_block_layout(16, 32, 16, 3, 1, 0, false, 0),
                                TestParamType_convolution_gpu_block_layout(16, 16, 32, 3, 1, 0, false, 0),
                                TestParamType_convolution_gpu_block_layout(32, 16, 32, 3, 1, 0, false, 0),
                                TestParamType_convolution_gpu_block_layout(16, 32, 32, 3, 1, 0, false, 0),
                                TestParamType_convolution_gpu_block_layout(16, 64, 16, 3, 1, 0, false, 0),
                                TestParamType_convolution_gpu_block_layout(32, 32, 16, 3, 1, 0, false, 0),
                                TestParamType_convolution_gpu_block_layout(32, 32, 32, 3, 1, 0, false, 0),
                                TestParamType_convolution_gpu_block_layout(32, 32, 32, 1, 1, 0, false, 0),
                                TestParamType_convolution_gpu_block_layout(32, 64, 64, 1, 1, 0, false, 0),
                                TestParamType_convolution_gpu_block_layout(16, 64, 64, 1, 1, 0, true, 0),
                                TestParamType_convolution_gpu_block_layout(16, 16, 16, 3, 1, 0, true, 0),
                                TestParamType_convolution_gpu_block_layout(32, 16, 16, 3, 1, 0, true, 0),
                                TestParamType_convolution_gpu_block_layout(16, 32, 16, 3, 1, 0, true, 0),
                                TestParamType_convolution_gpu_block_layout(16, 16, 32, 3, 1, 0, true, 0),
                                TestParamType_convolution_gpu_block_layout(32, 16, 32, 3, 1, 0, true, 0),
                                TestParamType_convolution_gpu_block_layout(16, 32, 32, 3, 1, 0, true, 0),
                                TestParamType_convolution_gpu_block_layout(16, 64, 16, 3, 1, 0, true, 0),
                                TestParamType_convolution_gpu_block_layout(32, 32, 16, 3, 1, 0, true, 0),
                                TestParamType_convolution_gpu_block_layout(32, 32, 32, 3, 1, 0, true, 0),
                                TestParamType_convolution_gpu_block_layout(64, 64, 64, 3, 1, 0, true, 0)),
                        convolution_gpu_block_layout::PrintToStringParamName);

TEST_P(convolution_gpu_block_layout, bfyx_bsv16_fsv16_fp32)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int batch_num = testing::get<0>(GetParam());
    const int input_xy = 5;
    const int input_f = testing::get<1>(GetParam());
    const int output_f = testing::get<2>(GetParam());
    const int filter_xy = testing::get<3>(GetParam());
    const uint64_t stride = testing::get<4>(GetParam());
    const int output_padding = testing::get<5>(GetParam());
    const bool with_bias = testing::get<6>(GetParam());
    const int pad = filter_xy / 2;

    if (batch_num <= 16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (for bs_fs_yx_bsv16_fsv16 batch should be greater than 16)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy);
    auto input_data = rg.generate_random_4d<float>(batch_num, input_f, input_xy, input_xy, 1, 10);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy);
    auto weights_data = rg.generate_random_4d<float>(output_f, input_f, filter_xy, filter_xy, -1, 1);

    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    auto weights_mem = engine.allocate_memory({ data_types::f32, format::bfyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<float>(batch_num, VVVF<float>(output_f));

    topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem));

    // Reorder input to bs_fs_yx_bsv16_fsv16
    topology.add(reorder("input_bsv16_fsv16", input_info("input"), { data_types::f32, format::bs_fs_yx_bsv16_fsv16, input_size }));

    if (with_bias)
    {
        // Generate bias data
        auto biases_size = tensor(1, output_f, 1, 1);
        auto biases_data = rg.generate_random_1d<float>(output_f, -1, 1);
        auto biases_mem = engine.allocate_memory({ data_types::f32, format::bfyx, biases_size });
        set_values(biases_mem, biases_data);

        // Calculate reference values with bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                        input_data[bi], weights_data[ofi],
                        stride, stride, biases_data[ofi],
                        1, 1,                               // dilation
                        pad, pad,                           // input padding
                        output_padding, output_padding);
            }
        }

        topology.add(data("biases", biases_mem));

        auto conv_bsv16_fsv16 = convolution("conv_bsv16_fsv16", input_info("input_bsv16_fsv16"), "weights", "biases",
                                       1, { stride, stride }, {1, 1}, { pad, pad }, { pad, pad }, false);
        conv_bsv16_fsv16.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

        topology.add(conv_bsv16_fsv16);
    }
    else
    {
        // Calculate reference values without bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                        input_data[bi], weights_data[ofi],
                        stride, stride,
                        0,                                  // bias
                        1, 1,                               // dilation
                        pad, pad,                           // input padding
                        output_padding, output_padding);
            }
        }

        auto conv_bsv16_fsv16 = convolution("conv_bsv16_fsv16", input_info("input_bsv16_fsv16"), "weights", no_bias,
                                       1, { stride, stride }, {1, 1}, { pad, pad }, { pad, pad }, false);
        conv_bsv16_fsv16.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

        topology.add(conv_bsv16_fsv16);
    }

    topology.add(reorder("reorder_bfyx", input_info("conv_bsv16_fsv16"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "conv_bsv16_fsv16", "reorder_bfyx" }));
    ov::intel_gpu::ImplementationDesc conv_impl = { format::bs_fs_yx_bsv16_fsv16, "" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_bsv16_fsv16", conv_impl } }));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv_bsv16_fsv16").get_memory();
    cldnn::mem_lock<float> out_ptr(out_mem, get_test_stream());

    auto out_mem_bfyx = network.get_output("reorder_bfyx").get_memory();
    cldnn::mem_lock<float> out_ptr_bfyx(out_mem_bfyx, get_test_stream());

    ASSERT_EQ(out_mem->get_layout().format, format::bs_fs_yx_bsv16_fsv16);

    auto flatten_ref = flatten_4d(format::bfyx, reference_result);

    for (size_t i = 0; i < out_ptr_bfyx.size(); i++) {
        auto equal = are_equal(flatten_ref[i], out_ptr_bfyx[i], 1e-2f);
        ASSERT_TRUE(equal);
        if (!equal)
        {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }
}

TEST_P(convolution_gpu_block_layout, bfyx_bsv16_fsv16_fp16)
{
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int batch_num = testing::get<0>(GetParam());
    const int input_xy = 5;
    const int input_f = testing::get<1>(GetParam());
    const int output_f = testing::get<2>(GetParam());
    const int filter_xy = testing::get<3>(GetParam());
    const uint64_t stride = testing::get<4>(GetParam());
    const int output_padding = testing::get<5>(GetParam());
    const bool with_bias = testing::get<6>(GetParam());
    const int pad = filter_xy / 2;

    if (batch_num % 32 != 0)
    {
        std::cout << "[ SKIPPED ] The test is skipped (for fp16 batch should be multiple of 32)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    tests::random_generator rg(GET_SUITE_NAME);
    auto input_size = tensor(batch_num, input_f, input_xy, input_xy);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_xy, input_xy, 0, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);

    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, filter_xy, filter_xy, 0, 1);

    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    auto weights_mem = engine.allocate_memory({ data_types::f16, format::bfyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<ov::float16>(batch_num, VVVF<ov::float16>(output_f));

    topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem));

    // Reorder input to bs_fs_yx_bsv16_fsv16
    topology.add(reorder("input_bsv16_fsv16", input_info("input"), { data_types::f16, format::bs_fs_yx_bsv16_fsv16, input_size }));

    if (with_bias)
    {
        // Generate bias data
        auto biases_size = tensor(1, output_f, 1, 1);
        auto biases_data = rg.generate_random_1d<ov::float16>(output_f, -1, 1);
        auto biases_mem = engine.allocate_memory({ data_types::f16, format::bfyx, biases_size });
        set_values(biases_mem, biases_data);

        // Calculate reference values with bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                        input_data[bi], weights_data[ofi],
                        stride, stride, biases_data[ofi],
                        1, 1,                               // dilation
                        pad, pad,                           // input padding
                        output_padding, output_padding);
            }
        }

        topology.add(data("biases", biases_mem));

        auto conv_bsv16_fsv16 = convolution("conv_bsv16_fsv16", input_info("input_bsv16_fsv16"), "weights", "biases",
                                       1, { stride, stride }, {1, 1}, { pad, pad }, { pad, pad }, false);
        conv_bsv16_fsv16.output_paddings = {padding({ 0, 0, output_padding, output_padding, 0 }, 0.f)};

        topology.add(conv_bsv16_fsv16);
    }
    else
    {
        // Calculate reference values without bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                        input_data[bi], weights_data[ofi],
                        stride, stride,
                        0,                                  // bias
                        1, 1,                               // dilation
                        pad, pad,                           // input padding
                        output_padding, output_padding);
            }
        }

        auto conv_bsv16_fsv16 = convolution("conv_bsv16_fsv16", input_info("input_bsv16_fsv16"), "weights", no_bias,
                                       1, { stride, stride }, {1, 1}, { pad, pad }, { pad, pad }, false);
        conv_bsv16_fsv16.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

        topology.add(conv_bsv16_fsv16);
    }

    topology.add(reorder("reorder_bfyx", input_info("conv_bsv16_fsv16"), format::bfyx, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "conv_bsv16_fsv16", "reorder_bfyx" }));
    ov::intel_gpu::ImplementationDesc conv_impl = { format::bs_fs_yx_bsv16_fsv16, "" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_bsv16_fsv16", conv_impl } }));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv_bsv16_fsv16").get_memory();
    cldnn::mem_lock<ov::float16> out_ptr(out_mem, get_test_stream());

    auto out_mem_bfyx = network.get_output("reorder_bfyx").get_memory();
    cldnn::mem_lock<ov::float16> out_ptr_bfyx(out_mem_bfyx, get_test_stream());

    ASSERT_EQ(out_mem->get_layout().format, format::bs_fs_yx_bsv16_fsv16);

    auto flatten_ref = flatten_4d(format::bfyx, reference_result);

    for (size_t i = 0; i < out_ptr_bfyx.size(); i++) {
        auto equal = are_equal(flatten_ref[i], out_ptr_bfyx[i], 1);
        ASSERT_TRUE(equal);
        if (!equal) {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }
}

TEST_P(convolution_gpu_block_layout, bfyx_bsv16_fsv16_fp32_fused_ops)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int batch_num = testing::get<0>(GetParam()) * 2;
    const int input_xy = 5;
    const int input_f = testing::get<1>(GetParam());
    const int output_f = testing::get<2>(GetParam());
    const int filter_xy = testing::get<3>(GetParam());
    const uint64_t stride = testing::get<4>(GetParam());
    const int output_padding = testing::get<5>(GetParam());
    const bool with_bias = testing::get<6>(GetParam());
    const int pad = filter_xy / 2;

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy);
    auto input_data = rg.generate_random_4d<float>(batch_num, input_f, input_xy, input_xy, 1, 10);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy, 1);
    auto weights_data = rg.generate_random_4d<float>(output_f, input_f, filter_xy, filter_xy, -1, 1);

    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    auto weights_mem = engine.allocate_memory({ data_types::f32, format::bfyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<float>(batch_num, VVVF<float>(output_f));

    topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem));

    // Reorder input to bs_fs_yx_bsv16_fsv16
    topology.add(reorder("input_bsv16_fsv16", input_info("input"), { data_types::f32, format::bs_fs_yx_bsv16_fsv16, input_size }));

    if (with_bias)
    {
        // Generate bias data
        auto biases_size = tensor(1, output_f, 1, 1, 1);
        auto biases_data = rg.generate_random_1d<float>(output_f, -1, 1);
        auto biases_mem = engine.allocate_memory({ data_types::f32, format::bfyx, biases_size });
        set_values(biases_mem, biases_data);

        // Calculate reference values with bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                        input_data[bi], weights_data[ofi],
                        stride, stride, biases_data[ofi],
                        1, 1,                               // dilation
                        pad, pad,                           // input padding
                        output_padding, output_padding);
            }
        }

        topology.add(data("biases", biases_mem));

        auto conv_bsv16_fsv16 = convolution("conv_bsv16_fsv16", input_info("input_bsv16_fsv16"), "weights", "biases",
                                       1, { stride, stride }, {1, 1}, { pad, pad }, { pad, pad }, false);
        conv_bsv16_fsv16.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

        topology.add(conv_bsv16_fsv16);
    }
    else
    {
        // Calculate reference values without bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                        input_data[bi], weights_data[ofi],
                        stride, stride,
                        0,                                  // bias
                        1, 1,                               // dilation
                        pad, pad,                           // input padding
                        output_padding, output_padding);
            }
        }

        auto conv_bsv16_fsv16 = convolution("conv_bsv16_fsv16", input_info("input_bsv16_fsv16"), "weights", no_bias,
                                       1, { stride, stride }, {1, 1}, { pad, pad }, { pad, pad }, false);
        conv_bsv16_fsv16.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

        topology.add(conv_bsv16_fsv16);
    }

    const float scalar = 5.5f;
    auto scale_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    set_values(scale_mem, { scalar });

    topology.add(data("scalar", scale_mem));
    topology.add(eltwise("scale", { input_info("conv_bsv16_fsv16"), input_info("scalar") }, eltwise_mode::prod));

    topology.add(reorder("reorder_bfyx", input_info("scale"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "conv_bsv16_fsv16", "reorder_bfyx" }));
    ov::intel_gpu::ImplementationDesc conv_impl = { format::bs_fs_yx_bsv16_fsv16, "" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_bsv16_fsv16", conv_impl } }));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv_bsv16_fsv16").get_memory();
    cldnn::mem_lock<float> out_ptr(out_mem, get_test_stream());

    auto out_mem_bfyx = network.get_output("reorder_bfyx").get_memory();
    cldnn::mem_lock<float> out_ptr_bfyx(out_mem_bfyx, get_test_stream());

    ASSERT_EQ(out_mem->get_layout().format, format::bs_fs_yx_bsv16_fsv16);

    auto flatten_ref = flatten_4d(format::bfyx, reference_result);

    for (size_t i = 0; i < out_ptr_bfyx.size(); i++) {
        auto equal = are_equal(flatten_ref[i] * scalar, out_ptr_bfyx[i], 1e-2f);
        ASSERT_TRUE(equal);
        if (!equal) {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(convolution_depthwise_gpu_fs_b_yx_fsv32,
                        convolution_depthwise_gpu,
                        ::testing::Values(
                                // Input size, Filter size Y, Filter size X, groups, Stride, Output padding, With bias
                                // Stride testing
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 2, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 3, 0, false),
                                // Different Features testing
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 16, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 20, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 25, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 33, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 35, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 45, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 65, 1, 0, false),
                                // Different filter's sizes testing
                                TestParamType_convolution_depthwise_gpu(5, 3, 2, 16, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 1, 16, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 2, 3, 16, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 1, 3, 16, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 2, 16, 2, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 1, 16, 2, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 2, 3, 16, 2, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 1, 3, 16, 2, 0, false),
                                // Input FeatureMap testing
                                TestParamType_convolution_depthwise_gpu(20, 3, 3, 50, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(30, 3, 3, 50, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(55, 3, 3, 50, 1, 0, false),
                                // Output padding testing + strides
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 1, 1, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 2, 2, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 3, 3, false)
                                ),
                        convolution_depthwise_gpu::PrintToStringParamName);

TEST_P(convolution_depthwise_gpu, depthwise_conv_fs_b_yx_fsv32)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }


    const int batch_num = 2;
    const int input_xy = testing::get<0>(GetParam());
    const int groups = testing::get<3>(GetParam());
    const int input_f = groups;
    const int output_f = groups;
    const int filter_y = testing::get<1>(GetParam());
    const int filter_x = testing::get<2>(GetParam());
    const uint64_t stride = testing::get<4>(GetParam());
    const int output_padding = testing::get<5>(GetParam());
    const int pad_y = filter_y / 2;
    const int pad_x = filter_x / 2;

    const int output_y = 1 + (input_xy + 2 * pad_y - filter_y) / stride + 2 * output_padding;
    const int output_x = 1 + (input_xy + 2 * pad_x - filter_x) / stride + 2 * output_padding;

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_xy, input_xy, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(group(groups), batch(1), feature(1), spatial(filter_x, filter_y));
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, 1, filter_y, filter_x, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::goiyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<ov::float16>(batch_num, VVVF<ov::float16>(output_f));

    topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights_fsv", weights_mem));

    // Reorder input to fs_byx_fsv32
    topology.add(reorder("input_fsv", input_info("input"), { data_types::f16, format::fs_b_yx_fsv32, input_size }));

    // Calculate reference values without bias
    for (auto bi = 0; bi < batch_num; ++bi)
    {
        for (auto ofi = 0; ofi < output_f; ++ofi)
        {
            reference_result[bi][ofi] = reference_convolve(
                    input_data[bi], weights_data[ofi],  // input, weights
                    stride, stride,                     // strides
                    0,                                  // bias
                    1, 1,                               // dilation
                    pad_y, pad_x,                       // input padding
                    output_padding, output_padding,     // output_padding
                    ofi, ofi + 1,                       // f_begin, f_end
                    true);                              // depthwise
        }
    }

    auto conv_fsv = convolution("conv_fsv", input_info("input_fsv"), "weights_fsv", no_bias, groups,
                                { stride, stride }, { 1, 1 }, {pad_y, pad_x}, {pad_y, pad_x}, true);
    conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

    topology.add(conv_fsv);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc conv_impl = { format::fs_b_yx_fsv32, "" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_fsv", conv_impl } }));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv_fsv").get_memory();
    cldnn::mem_lock<ov::float16> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(out_mem->get_layout().format, format::fs_b_yx_fsv32);

    for (int bi = 0; bi < batch_num; ++bi)
        for (int fi = 0; fi < output_f; ++fi)
            for (int yi = 0; yi < output_y; ++yi)
                for (int xi = 0; xi < output_x; ++xi)
                {
                    auto val_ref = reference_result[bi][fi][yi][xi];
                    auto val = out_ptr[(fi / 32) * batch_num * output_y * output_x * 32 +
                                       bi * output_y * output_x * 32 +
                                       yi * output_x * 32 +
                                       xi * 32 +
                                       fi % 32];
                    auto equal = are_equal(val_ref, val, 1e-2f);
                    ASSERT_TRUE(equal);
                    if (!equal)
                    {
                        std::cout << "At b = " << bi << ", fi = " << fi << ", yi = " << yi << ", xi = " << xi << std::endl;
                    }
                }
}

INSTANTIATE_TEST_SUITE_P(convolution_depthwise_gpu_b_fs_yx_fsv16,
                        convolution_depthwise_gpu_fsv16,
                        ::testing::Values(
                            // Input size, Filter size Y, Filter size X, groups, Stride, Output padding, With bias
                            // Stride testing
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 2, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 3, 0, false),
                            // Different Features testing
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 16, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 20, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 25, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 33, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 35, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 45, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 65, 1, 0, false),
                            // Different filter's sizes testing
                            TestParamType_convolution_depthwise_gpu(15, 1, 1, 16, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 2, 2, 16, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 3, 3, 16, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 4, 4, 16, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 5, 5, 16, 2, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 8, 8, 16, 2, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 1, 9, 16, 2, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 17, 8, 16, 2, 0, false),
                            // Input FeatureMap testing
                            TestParamType_convolution_depthwise_gpu(20, 3, 3, 50, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(30, 3, 3, 50, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(55, 3, 3, 50, 1, 0, false),
                            // Output padding testing + strides
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 1, 1, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 2, 2, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 3, 3, false)
                        ),
                        convolution_depthwise_gpu_fsv16::PrintToStringParamName);

TEST_P(convolution_depthwise_gpu_fsv16, depthwise_conv_b_fs_yx_fsv16)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int batch_num = 2;
    const int input_xy = testing::get<0>(GetParam());
    const int groups = testing::get<3>(GetParam());
    const int input_f = groups;
    const int output_f = groups;
    const int filter_y = testing::get<1>(GetParam());
    const int filter_x = testing::get<2>(GetParam());
    const uint64_t stride = testing::get<4>(GetParam());
    const int output_padding = testing::get<5>(GetParam());
    const int pad_y = filter_y / 2;
    const int pad_x = filter_x / 2;
    const int f_group_size = 16;
    const int f_group_num_in_batch = (output_f % f_group_size) ? (output_f / f_group_size + 1) : (output_f / f_group_size);

    const int output_y = 1 + (input_xy + 2 * pad_y - filter_y) / stride + 2 * output_padding;
    const int output_x = 1 + (input_xy + 2 * pad_x - filter_x) / stride + 2 * output_padding;

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_xy, input_xy, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(group(output_f), batch(1), feature(1), spatial(filter_x, filter_y));
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, 1, filter_y, filter_x, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::goiyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<ov::float16>(batch_num, VVVF<ov::float16>(output_f));

    topology topology(
        input_layout("input", input_mem->get_layout()),
        data("weights_fsv", weights_mem));

    // Reorder input to b_fs_yx_fsv16
    topology.add(reorder("input_fsv", input_info("input"), { data_types::f16, format::b_fs_yx_fsv16, input_size }));

    // Calculate reference values without bias
    for (auto bi = 0; bi < batch_num; ++bi)
    {
        for (auto ofi = 0; ofi < output_f; ++ofi)
        {
            reference_result[bi][ofi] = reference_convolve(
                input_data[bi], weights_data[ofi],  // input, weights
                stride, stride,                     // strides
                0,                                  // bias
                1, 1,                               // dilation
                pad_y, pad_x,                       // input padding
                output_padding, output_padding,     // output_padding
                ofi, ofi + 1,                       // f_begin, f_end
                true);                              // depthwise
        }
    }

    auto conv_fsv = convolution("conv_fsv", input_info("input_fsv"), "weights_fsv", no_bias, groups,
                                { stride, stride }, {1, 1}, { pad_y, pad_x }, { pad_y, pad_x }, true);
    conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

    topology.add(conv_fsv);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_fsv", conv_impl } }));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv_fsv").get_memory();
    cldnn::mem_lock<ov::float16> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(out_mem->get_layout().format, format::b_fs_yx_fsv16);

    for (int bi = 0; bi < batch_num; ++bi)
        for (int fi = 0; fi < output_f; ++fi)
            for (int yi = 0; yi < output_y; ++yi)
                for (int xi = 0; xi < output_x; ++xi)
                {
                    auto val_ref = reference_result[bi][fi][yi][xi];

                    auto val = out_ptr[bi * f_group_num_in_batch * f_group_size * output_y * output_x  +
                        (fi / f_group_size) * output_y * output_x * f_group_size +
                        yi * output_x * f_group_size +
                        xi * f_group_size +
                        fi % f_group_size];
                    auto equal = are_equal(val_ref, val, 1e-2f);
                    ASSERT_TRUE(equal);
                    if (!equal)
                    {
                        std::cout << "At b = " << bi << ", fi = " << fi << ", yi = " << yi << ", xi = " << xi << std::endl;
                    }
                }
}

INSTANTIATE_TEST_SUITE_P(convolution_depthwise_gpu_b_fs_yx_fsv16_1d_depthwise_convolution,
                        convolution_depthwise_gpu_fsv16_xy,
                        ::testing::Values(
                            // Input Y size, Input X size, Filter size Y, Filter size X, groups, Stride, Output padding, With bias
                            // F is multiple of 16
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 16, 1, 0, false),
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 32, 1, 0, false),
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 48, 1, 0, false),
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 64, 1, 0, false),
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 80, 1, 0, false),
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 96, 1, 0, false),
                            // F is not multiple of 16
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 8, 1, 0, false),
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 17, 1, 0, false),
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 34, 1, 0, false),
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 51, 1, 0, false),
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 68, 1, 0, false),
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 85, 1, 0, false),
                            TestParamType_convolution_depthwise_xy_gpu(55, 1, 3, 1, 102, 1, 0, false)
                        ),
                        convolution_depthwise_gpu_fsv16_xy::PrintToStringParamName);

TEST_P(convolution_depthwise_gpu_fsv16_xy, depthwise_conv_b_fs_yx_fsv16)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int batch_num = 2;
    const int input_y = testing::get<0>(GetParam());
    const int input_x = testing::get<1>(GetParam());
    const int groups = testing::get<4>(GetParam());
    const int input_f = groups;
    const int output_f = groups;
    const int filter_y = testing::get<2>(GetParam());
    const int filter_x = testing::get<3>(GetParam());
    const uint64_t stride = testing::get<5>(GetParam());
    const int output_padding = testing::get<6>(GetParam());
    const int pad_y = filter_y / 2;
    const int pad_x = filter_x / 2;
    const int f_group_size = 16;
    const int f_group_num_in_batch = (output_f % f_group_size) ? (output_f / f_group_size + 1) : (output_f / f_group_size);

    const int output_y = 1 + (input_y + 2 * pad_y - filter_y) / stride + 2 * output_padding;
    const int output_x = 1 + (input_x + 2 * pad_x - filter_x) / stride + 2 * output_padding;

    auto input_size = tensor(batch_num, input_f, input_x, input_y);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_y, input_x, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(group(output_f), batch(1), feature(1), spatial(filter_x, filter_y));
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, 1, filter_y, filter_x, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::goiyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<ov::float16>(batch_num, VVVF<ov::float16>(output_f));

    topology topology(
        input_layout("input", input_mem->get_layout()),
        data("weights_fsv", weights_mem));

    // Reorder input to b_fs_yx_fsv16
    topology.add(reorder("input_fsv", input_info("input"), { data_types::f16, format::b_fs_yx_fsv16, input_size }));

    // Calculate reference values without bias
    for (auto bi = 0; bi < batch_num; ++bi)
    {
        for (auto ofi = 0; ofi < output_f; ++ofi)
        {
            reference_result[bi][ofi] = reference_convolve(
                input_data[bi], weights_data[ofi],  // input, weights
                stride, stride,                     // strides
                0,                                  // bias
                1, 1,                               // dilation
                pad_y, pad_x,                       // input padding
                output_padding, output_padding,     // output_padding
                ofi, ofi + 1,                       // f_begin, f_end
                true);                              // depthwise
        }
    }

    auto conv_fsv = convolution("conv_fsv", input_info("input_fsv"), "weights_fsv", no_bias, groups,
                                { stride, stride }, {1, 1}, { pad_y, pad_x }, { pad_y, pad_x }, true);
    conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

    topology.add(conv_fsv);
    topology.add(reorder("out", input_info("conv_fsv"), format::b_fs_yx_fsv16, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc conv_impl;
    if (engine.get_device_info().supports_immad) {
        conv_impl = { format::b_fs_yx_fsv16, "", impl_types::onednn };
    } else {
        conv_impl = { format::b_fs_yx_fsv16, "convolution_gpu_bfyx_f16_depthwise" };
    }
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_fsv", conv_impl } }));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("out").get_memory();

    cldnn::mem_lock<ov::float16> out_ptr(out_mem, get_test_stream());
    ASSERT_EQ(out_mem->get_layout().format, format::b_fs_yx_fsv16);

    for (int bi = 0; bi < batch_num; ++bi) {
        for (int fi = 0; fi < output_f; ++fi) {
            for (int yi = 0; yi < output_y; ++yi) {
                for (int xi = 0; xi < output_x; ++xi)
                {
                    auto val_ref = reference_result[bi][fi][yi][xi];
                    auto val = out_ptr[bi * f_group_num_in_batch * f_group_size * output_y * output_x  +
                                        (fi / f_group_size) * output_y * output_x * f_group_size +
                                        yi * output_x * f_group_size +
                                        xi * f_group_size +
                                        fi % f_group_size];

                    auto equal = are_equal(val_ref, val, 1e-2f);
                    ASSERT_TRUE(equal);
                }
            }
        }
    }
}

TEST(convolution_depthwise_gpu_fsv16, depthwise_conv_b_fs_yx_fsv16_in_feature_padding) {
    //  Input:                  1x32x2x1
    //  Input padding above:    0x16x0x0
    //  Input padding below:    0x64x0x0
    //  Groups:                 32
    //  Filter:                 32x1x1x1x1
    //  Output:                 1x32x2x1

    auto num_groups = 32;
    auto input_size = tensor{ 1, num_groups, 1, 2 };
    auto weights_size = tensor(group(num_groups), batch(1), feature(1), spatial(1, 1));
    auto bias_size = tensor{ 1, num_groups, 1, 1 };
    ov::Strides stride = { 1, 1 };
    ov::CoordinateDiff pad = { 0, 0 };
    ov::Strides dilation = { 1, 1 };
    auto output_size = tensor{ 1, num_groups, 1, 2 };
    auto input_lower_sizes = { 0, 16, 0, 0 };
    auto input_upper_sizes = { 0, 64, 0, 0 };

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, input_size });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, weights_size });
    auto bias = engine.allocate_memory({ data_types::f32, format::bfyx, bias_size });

    set_values<float>(input, {
         3, -1, -1, -1,  2, -2,  2,  2,  0,  1, -5,  4, -1,  4,  1,  0,
        -1,  4, -4,  3, -4,  4,  0,  1, -4,  3,  1,  0, -3, -1,  3,  0,
         4, -2,  0, -2, -1, -4,  1, -4,  3,  2,  2,  2,  3,  4,  0, -5,
         3, -5,  2, -3,  0,  3, -3, -4, -1, -4,  2,  4,  3,  4, -5, -5
    });
    set_values<float>(weights, {
        -4, -1, -4, -2,  1,  2,  3, -4, -5, -2, -2,  1,  2,  2, -5,  2,
        -5,  4, -3,  2, -1, -3,  0, -1, -5, -3, -5, -4,  1, -4,  4,  2
    });
    set_values<float>(bias, {
        -2,  3, -4, -4, -4,  4, -3, -1, -3, -5,  3,  3,  1,  4, -4,  1,
        -1, -2, -2, -1,  2, -1, -4,  2, -4, -1,  3,  0,  0,  4, -3, -4
    });
    std::vector<float> output_vec = {
        -14,   2,   4,   4, -12,   4,  -8,  -8,  -4,  -3,  -6,  12,  -6,   9,  -5,  -1,
          2, -23,   3, -11,  11,  -5,   3,   4,  -7,   7,   6,   4,  11,   1,   7,   1,
        -21,   9,  -2, -10,   1,  10,   1,  -9,  -1,   0,  -7,  -7,  -4,  -4,   2,   7,
        -19,  21,  -7,   8,   3, -12,  12,  16,  -1,  -4,  -4, -12,   9,  13, -14, -14
    };

    // reorder input to fsv16 format and introduce feature padding
    padding input_padding = padding(input_lower_sizes, input_upper_sizes);
    layout reordered_input_layout = layout(data_types::f32, format::b_fs_yx_fsv16, input_size, input_padding);

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("input_reordered", input_info("input"), reordered_input_layout),
        data("weights", weights),
        data("bias", bias),
        convolution("conv", input_info("input_reordered"), "weights", "bias", num_groups, stride, dilation, pad, pad, true),
        reorder("out", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", conv_impl } }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();

    ASSERT_EQ(output_layout.format, format::bfyx);

    ASSERT_EQ(y_size, output_size.spatial[1]);
    ASSERT_EQ(x_size, output_size.spatial[0]);
    ASSERT_EQ(f_size, output_size.feature[0]);
    ASSERT_EQ(b_size, output_size.batch[0]);

    for (int b = 0; b < b_size; ++b) {
        for (int f = 0; f < f_size; ++f) {
            for (int y = 0; y < y_size; ++y) {
                for (int x = 0; x < x_size; ++x) {
                    ASSERT_EQ(
                        output_vec[b * f_size * y_size * x_size + f * y_size * x_size + y * x_size + x],
                        output_ptr[b * f_size * y_size * x_size + f * y_size * x_size + y * x_size + x]
                    );
                }
            }
        }
    }
}

struct convolution_depthwise_gpu_bfyx : public convolution_depthwise_gpu {};

TEST_P(convolution_depthwise_gpu_bfyx, depthwise_conv_bfyx)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int batch_num = 2;
    const int input_xy = testing::get<0>(GetParam());
    const int groups = testing::get<3>(GetParam());
    const int input_f = groups;
    const int output_f = groups;
    const int filter_y = testing::get<1>(GetParam());
    const int filter_x = testing::get<2>(GetParam());
    const uint64_t stride = testing::get<4>(GetParam());
    const int output_padding = testing::get<5>(GetParam());
    const int pad_y = filter_y / 2;
    const int pad_x = filter_x / 2;
    const int f_group_size = 1;
    const int f_group_num_in_batch = (output_f % f_group_size) ? (output_f / f_group_size + 1) : (output_f / f_group_size);

    const int output_y = 1 + (input_xy + 2 * pad_y - filter_y) / stride + 2 * output_padding;
    const int output_x = 1 + (input_xy + 2 * pad_x - filter_x) / stride + 2 * output_padding;

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_xy, input_xy, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(group(output_f), batch(1), feature(1), spatial(filter_x, filter_y));
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, 1, filter_y, filter_x, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::goiyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<ov::float16>(batch_num, VVVF<ov::float16>(output_f));

    topology topology(
        input_layout("input", input_mem->get_layout()),
        data("weights", weights_mem));

    // Calculate reference values without bias
    for (auto bi = 0; bi < batch_num; ++bi)
    {
        for (auto ofi = 0; ofi < output_f; ++ofi)
        {
            reference_result[bi][ofi] = reference_convolve(
                input_data[bi], weights_data[ofi],  // input, weights
                stride, stride,                     // strides
                0,                                  // bias
                1, 1,                               // dilation
                pad_y, pad_x,                       // input padding
                output_padding, output_padding,     // output_padding
                ofi, ofi + 1,                       // f_begin, f_end
                true);                              // depthwise
        }
    }

    auto conv_fsv = convolution("conv", input_info("input"), "weights", no_bias, groups,
                                { stride, stride }, {1, 1}, { pad_y, pad_x }, { pad_y, pad_x }, true);
    conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

    topology.add(conv_fsv);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc conv_impl = { format::bfyx, "" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", conv_impl } }));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv").get_memory();
    cldnn::mem_lock<ov::float16> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(out_mem->get_layout().format, format::bfyx);

    for (int bi = 0; bi < batch_num; ++bi)
        for (int fi = 0; fi < output_f; ++fi)
            for (int yi = 0; yi < output_y; ++yi)
                for (int xi = 0; xi < output_x; ++xi)
                {
                    auto val_ref = reference_result[bi][fi][yi][xi];

                    auto val = out_ptr[bi * f_group_num_in_batch * f_group_size * output_y * output_x  +
                        (fi / f_group_size) * output_y * output_x * f_group_size +
                        yi * output_x * f_group_size +
                        xi * f_group_size +
                        fi % f_group_size];
                    auto equal = are_equal(val_ref, val, 1e-2f);
                    ASSERT_TRUE(equal) << "At b = " << bi << ", fi = " << fi << ", yi = " << yi << ", xi = " << xi << std::endl;
                }
}


INSTANTIATE_TEST_SUITE_P(convolution_depthwise_gpu_bfyx,
                        convolution_depthwise_gpu_bfyx,
                        ::testing::Values(
                            // Input size, Filter size Y, Filter size X, groups, Stride, Output padding, With bias
                            // Stride testing
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 2, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 3, 0, false),
                            // Different Features testing
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 16, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 20, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 25, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 33, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 35, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 45, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 65, 1, 0, false),
                            // Different filter's sizes testing
                            TestParamType_convolution_depthwise_gpu(15, 1, 1, 16, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 2, 2, 16, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 3, 3, 16, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 4, 4, 16, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 5, 5, 16, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 5, 5, 16, 2, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 8, 8, 16, 2, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 1, 9, 16, 2, 0, false),
                            TestParamType_convolution_depthwise_gpu(15, 17, 8, 16, 2, 0, false),
                            // Input FeatureMap testing
                            TestParamType_convolution_depthwise_gpu(20, 3, 3, 50, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(30, 3, 3, 50, 1, 0, false),
                            TestParamType_convolution_depthwise_gpu(55, 3, 3, 50, 1, 0, false),
                            // Output padding testing + strides
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 1, 1, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 2, 2, false),
                            TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 3, 3, false)
                        ),
                        convolution_depthwise_gpu::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(convolution_grouped_fsv4_fsv16,
                        convolution_grouped_gpu,
                        ::testing::Values(
                            // Input X size, Input Y size, Input Z size, Input features, Output features,
                            // Kernel size X, Kernel size Y, Kernel size Z, Groups number, Stride, Batch,
                            // Activation zero points, Weights zero points, Compensation,
                            // Input data format, Implementation name

                            // Format: b_fs_yx_fsv4
                            TestParamType_grouped_convolution_gpu(4, 4, 1, 16, 17, 3, 3, 1, 1, 1, 1, true, true, true, format::b_fs_yx_fsv4, ""),
                            TestParamType_grouped_convolution_gpu(4, 4, 1, 16, 16, 3, 3, 1, 4, 1, 1, true, true, true, format::b_fs_yx_fsv4, ""),
                            TestParamType_grouped_convolution_gpu(4, 4, 1, 8, 4, 2, 2, 1, 2, 1, 4, true, true, true, format::b_fs_yx_fsv4, ""),
                            TestParamType_grouped_convolution_gpu(8, 8, 1, 16, 16, 4, 4, 1, 4, 1, 1, true, true, true, format::b_fs_yx_fsv4, ""),
                            TestParamType_grouped_convolution_gpu(17, 17, 1, 32, 96, 3, 3, 1, 2, 2, 2, true, true, true, format::b_fs_yx_fsv4, ""),
                            TestParamType_grouped_convolution_gpu(16, 16, 1, 8, 48, 2, 2, 1, 2, 2, 1, true, true, true, format::b_fs_yx_fsv4, ""),
                            TestParamType_grouped_convolution_gpu(3, 3, 1, 48, 96, 2, 2, 1, 2, 8, 1, true, true, true, format::b_fs_yx_fsv4, ""),
                            TestParamType_grouped_convolution_gpu(6, 6, 1, 8, 26, 3, 3, 1, 2, 4, 1, true, true, true, format::b_fs_yx_fsv4, ""),
                            TestParamType_grouped_convolution_gpu(3, 1, 1, 80, 252, 3, 1, 1, 4, 1, 1, false, false, false, format::b_fs_yx_fsv4, ""),
                            TestParamType_grouped_convolution_gpu(3, 1, 1, 80, 252, 3, 1, 1, 4, 1, 1, false, true, false, format::b_fs_yx_fsv4, ""),
                            TestParamType_grouped_convolution_gpu(3, 1, 1, 80, 252, 3, 1, 1, 4, 1, 1, true, false, false, format::b_fs_yx_fsv4, ""),
                            TestParamType_grouped_convolution_gpu(3, 1, 1, 80, 252, 3, 1, 1, 4, 1, 1, true, true, false, format::b_fs_yx_fsv4, ""),
                            TestParamType_grouped_convolution_gpu(3, 1, 1, 80, 252, 3, 1, 1, 4, 1, 1, true, false, true, format::b_fs_yx_fsv4, ""),

                            // Format: b_fs_yx_fsv16
                            TestParamType_grouped_convolution_gpu(12, 12, 1, 96, 96, 3, 3, 1, 32, 1, 1, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(4, 4, 1, 8, 16, 3, 3, 1, 2, 1, 1, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(7, 7, 1, 8, 4, 3, 3, 1, 4, 1, 1, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(5, 5, 1, 34, 12, 3, 3, 1, 2, 1, 1, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(8, 8, 1, 34, 24, 3, 3, 1, 2, 1, 1, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(2, 2, 1, 12, 12, 3, 3, 1, 4, 1, 1, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(3, 3, 1, 8, 8, 3, 3, 1, 2, 1, 1, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(4, 4, 1, 8, 4, 2, 2, 1, 2, 2, 4, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(4, 4, 1, 16, 17, 3, 3, 1, 1, 1, 1, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(4, 4, 1, 8, 4, 2, 2, 1, 2, 1, 4, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(8, 8, 1, 16, 16, 4, 4, 1, 4, 1, 1, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(17, 17, 1, 32, 96, 3, 3, 1, 2, 2, 2, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(16, 16, 1, 8, 48, 2, 2, 1, 2, 2, 1, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(3, 3, 1, 48, 96, 2, 2, 1, 2, 8, 1, true, true, true, format::b_fs_yx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(6, 6, 1, 8, 26, 3, 3, 1, 2, 4, 1, true, true, true, format::b_fs_yx_fsv16, ""),

                            // Format: b_fs_zyx_fsv16
                            TestParamType_grouped_convolution_gpu(7, 5, 3, 51, 99, 3, 3, 3, 3, 1, 1, true, true, true, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(8, 6, 4, 32, 64, 2, 2, 2, 2, 1, 1, true, true, true, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(33, 6, 4, 16, 32, 4, 3, 2, 2, 1, 1, true, true, true, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(33, 1, 1, 30, 62, 1, 1, 1, 2, 1, 1, true, true, true, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(2, 1, 1, 18, 32, 3, 1, 1, 2, 2, 1, true, true, true, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(80, 1, 1, 48, 96, 33, 1, 1, 2, 8, 1, false, false, false, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(80, 1, 1, 48, 96, 33, 1, 1, 2, 8, 1, false, true, false, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(80, 1, 1, 48, 96, 33, 1, 1, 2, 8, 1, true, false, false, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(80, 1, 1, 48, 96, 33, 1, 1, 2, 8, 1, true, true, false, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(80, 1, 1, 48, 96, 33, 1, 1, 2, 8, 1, true, false, true, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(80, 1, 1, 48, 96, 33, 1, 1, 2, 8, 1, true, true, true, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(3, 1, 5, 196, 252, 3, 1, 3, 4, 1, 1, false, false, false, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(3, 1, 5, 196, 252, 3, 1, 3, 4, 1, 1, false, true, false, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(3, 1, 5, 196, 252, 3, 1, 3, 4, 1, 1, true, false, false, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(3, 1, 5, 196, 252, 3, 1, 3, 4, 1, 1, true, true, false, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(3, 1, 5, 196, 252, 3, 1, 3, 4, 1, 1, true, false, true, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(3, 1, 5, 196, 252, 3, 1, 3, 4, 1, 1, true, true, true, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(4, 1, 6, 256, 256, 2, 1, 2, 4, 1, 1, true, true, true, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(4, 1, 6, 256, 512, 2, 1, 3, 16, 1, 1, true, true, true, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(1, 3, 1, 18, 2, 1, 3, 1, 2, 1, 1, true, true, true, format::b_fs_zyx_fsv16, ""),
                            TestParamType_grouped_convolution_gpu(2, 3, 4, 3, 18, 3, 3, 3, 1, 1, 1, false, false, false, format::b_fs_zyx_fsv16, "convolution_gpu_mmad_bfyx_to_b_fs_yx_fsv32"),
                            TestParamType_grouped_convolution_gpu(79, 224, 224, 3, 64, 3, 3, 3, 1, 2, 1, false, false, false, format::b_fs_zyx_fsv16, "convolution_gpu_mmad_bfyx_to_b_fs_yx_fsv32")
                        ),
                        convolution_grouped_gpu::PrintToStringParamName);

TEST_P(convolution_grouped_gpu, base) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int input_x = testing::get<0>(GetParam()),
              input_y = testing::get<1>(GetParam()),
              input_z = testing::get<2>(GetParam()),
              input_f = testing::get<3>(GetParam()),
              output_f = testing::get<4>(GetParam()),
              filter_x = testing::get<5>(GetParam()),
              filter_y = testing::get<6>(GetParam()),
              filter_z = testing::get<7>(GetParam()),
              groups = testing::get<8>(GetParam()),
              batch_num = testing::get<10>(GetParam()),
              pad_z = (filter_z - 1) / 2,
              pad_y = (filter_y - 1) / 2,
              pad_x = (filter_x - 1) / 2;
    const uint64_t stride = testing::get<9>(GetParam());
    const auto has_input_zp = testing::get<11>(GetParam());
    const auto has_weights_zp = testing::get<12>(GetParam());
    const auto has_comp = testing::get<13>(GetParam());
    const auto input_data_format = testing::get<14>(GetParam());
    const auto impl_name = testing::get<15>(GetParam());

    // can use compensation term only if data zero points are available
    ASSERT_TRUE(has_input_zp || !has_comp);

    auto num_in_spatial_dims = input_data_format.spatial_num();

    auto input_size = tensor(batch(batch_num), feature(input_f), spatial(input_x, input_y, input_z));
    auto input_rnd = rg.generate_random_5d<int8_t>(batch_num, input_f, input_z, input_y, input_x, -127, 127);

    auto input_lay = layout(data_types::i8, format::bfzyx, input_size);
    if (num_in_spatial_dims == 2) {
        input_lay = layout(data_types::i8, format::bfyx, input_size);
    }
    std::vector<int8_t> input_flat(input_lay.get_linear_size());
    for (int b = 0; b < batch_num; b++)
        for (int f = 0; f < input_f; f++)
            for (int z = 0; z < input_z; z++)
                for (int y = 0; y < input_y; y++)
                    for (int x = 0; x < input_x; x++) {
                        tensor coords = tensor(batch(b), feature(f), spatial(x, y, z, 0));
                        size_t offset = input_lay.get_linear_offset(coords);
                        input_flat[offset] = input_rnd[b][f][z][y][x];
                    }
    auto input = engine.allocate_memory(input_lay);
    set_values(input, input_flat);

    auto input_zp_rnd = std::vector<int8_t>(input_f);
    auto input_zp_prim_name = "";
    if (has_input_zp) {
        input_zp_rnd = rg.generate_random_1d<int8_t>(input_f, -127, 127);
        input_zp_prim_name = "input_zp";
    }
    auto input_zp_lay = layout(data_types::i8, format::bfyx, tensor(feature(input_f)));
    auto input_zp = engine.allocate_memory(input_zp_lay);
    set_values(input_zp, input_zp_rnd);

    auto weights_size = tensor(group(groups), batch(output_f / groups), feature(input_f / groups), spatial(filter_x, filter_y, filter_z));

    VVVVVVF<int8_t> weights_rnd = rg.generate_random_6d<int8_t>(groups, output_f / groups, input_f / groups, filter_z, filter_y, filter_x, -127, 127);
    auto weights_lay = layout(data_types::i8, format::goizyx, weights_size);
    if (num_in_spatial_dims == 2) {
        weights_lay = layout(data_types::i8, format::goiyx, weights_size);
    }
    std::vector<int8_t> weights_flat(weights_lay.get_linear_size());
    for (int gi = 0; gi < groups; ++gi)
        for (int ofi = 0; ofi < output_f / groups; ++ofi)
            for (int ifi = 0; ifi < input_f / groups; ++ifi)
                for (int kzi = 0; kzi < filter_z; ++kzi)
                    for (int kyi = 0; kyi < filter_y; ++kyi)
                        for (int kxi = 0; kxi < filter_x; ++kxi) {
                            tensor coords = tensor(group(gi), batch(ofi), feature(ifi), spatial(kxi, kyi, kzi, 0));
                            size_t offset = weights_lay.get_linear_offset(coords);
                            weights_flat[offset] = weights_rnd[gi][ofi][ifi][kzi][kyi][kxi];
                        }
    auto weights = engine.allocate_memory(weights_lay);
    set_values(weights, weights_flat);

    auto weights_zp_rnd = std::vector<int8_t>(output_f);
    auto weights_zp_prim_name = "";
    if (has_weights_zp) {
        weights_zp_rnd = rg.generate_random_1d<int8_t>(output_f, -127, 127);
        weights_zp_prim_name = "weights_zp";
    }
    auto weights_zp_lay = layout(data_types::i8, format::bfyx, tensor(batch(output_f)));
    auto weights_zp = engine.allocate_memory(weights_zp_lay);
    set_values(weights_zp, weights_zp_rnd);

    VVVVVF<float> expected_result(batch_num, VVVVF<float>(output_f));

    // Calculate reference values without bias
    for (int bi = 0; bi < batch_num; ++bi)
        for (int gi = 0; gi < groups; ++gi)
            for (int ofi = 0; ofi < (int)weights_rnd[0].size(); ++ofi) {
                bool grouped = groups > 1;
                int f_begin = gi * input_f / groups;
                int f_end = gi * input_f / groups + input_f / groups;

                expected_result[bi][ofi + gi * output_f / groups] = reference_convolve<int8_t, float, int8_t>(
                    input_rnd[bi], weights_rnd[gi][ofi],                    // input, weights
                    stride, stride, stride,                                 // strides
                    0,                                                      // bias
                    1, 1, 1,                                                // dilation
                    pad_z, pad_y, pad_x,                                    // input padding
                    0, 0, 0,                                                // output_padding
                    f_begin, f_end,                                         // f_begin, f_end
                    false,                                                  // depthwise
                    grouped,                                                // grouped
                    input_zp_rnd,                                           // input zero points
                    weights_zp_rnd[gi * (int)weights_rnd[0].size() + ofi]); // weights zero points
            }

    auto ref_conv_out_size = tensor(batch(expected_result.size()),
                                    feature(expected_result[0].size()),
                                    spatial(expected_result[0][0][0][0].size(),
                                            expected_result[0][0][0].size(),
                                            expected_result[0][0].size()));

    auto comp_val = std::vector<float>(output_f);
    auto comp_prim_name = "";
    if (has_comp) {
        for (int g = 0; g < groups; g++) {
            for (int oc = 0; oc < output_f / groups; oc++) {
                float c = 0.f;
                for (int ic = 0; ic < input_f / groups; ic++) {
                    for (int zi = 0; zi < filter_z; zi++) {
                        for (int yi = 0; yi < filter_y; yi++) {
                            for (int xi = 0; xi < filter_x; xi++) {
                                int azp_idx = g*(input_f / groups) + ic;
                                int wzp_idx = g*(output_f / groups) + oc;
                                c += weights_rnd[g][oc][ic][zi][yi][xi] * input_zp_rnd[azp_idx];
                                if (has_weights_zp) {
                                    c -= input_zp_rnd[azp_idx] * weights_zp_rnd[wzp_idx];
                                }
                            }
                        }
                    }
                }

                comp_val[g*(output_f / groups) + oc] = -c;
            }
        }
        comp_prim_name = "compensation";
    }
    auto comp_lay = layout(data_types::f32, format::bfyx, tensor(batch(output_f)));
    auto comp = engine.allocate_memory(comp_lay);
    set_values(comp, comp_val);

    ov::Strides strides = {stride, stride};
    ov::Strides dilations(num_in_spatial_dims, 1);
    ov::CoordinateDiff pad = { pad_y, pad_x };
    if (num_in_spatial_dims == 3) {
        strides.insert(strides.begin(), stride);
        pad.insert(pad.begin(), pad_z);
    }

    topology topology(input_layout("input", input->get_layout()),
                      data("weights", weights),
                      reorder("input_fsv", input_info("input"), { data_types::i8, input_data_format, input_size }),
                      convolution("conv",
                                  input_info("input_fsv"),
                                  "weights",
                                  "",
                                  weights_zp_prim_name,
                                  input_zp_prim_name,
                                  comp_prim_name,
                                  groups,
                                  strides,
                                  dilations,
                                  pad,
                                  pad,
                                  true,
                                  data_types::f32),
                      reorder("out", input_info("conv"), { data_types::f32, format::bfzyx, ref_conv_out_size }));

    if (has_input_zp)
        topology.add(data(input_zp_prim_name, input_zp));

    if (has_weights_zp)
        topology.add(data(weights_zp_prim_name, weights_zp));

    if (has_comp)
        topology.add(data(comp_prim_name, comp));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "conv", "out" }));
    ov::intel_gpu::ImplementationDesc conv_impl = { input_data_format, impl_name };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", conv_impl } }));

    cldnn::network network(engine, topology, config);
    network.set_input_data("input", input);
    network.execute();

    auto out_mem = network.get_output("conv").get_memory();
    cldnn::mem_lock<float> out_ptr(out_mem, get_test_stream());
    auto out_lay = out_mem->get_layout();

    ASSERT_EQ(out_mem->get_layout().format, input_data_format);
    ASSERT_EQ(out_lay.batch(), expected_result.size());
    ASSERT_EQ(out_lay.feature(), expected_result[0].size());
    ASSERT_EQ(out_lay.spatial(2), expected_result[0][0].size());
    ASSERT_EQ(out_lay.spatial(1), expected_result[0][0][0].size());
    ASSERT_EQ(out_lay.spatial(0), expected_result[0][0][0][0].size());

    for (int bi = 0; bi < batch_num; ++bi)
        for (int ofi = 0; ofi < output_f; ++ofi)
            for (int zi = 0; zi < (int)expected_result[0][0].size(); ++zi)
                for (int yi = 0; yi < (int)expected_result[0][0][0].size(); ++yi)
                    for (int xi = 0; xi < (int)expected_result[0][0][0][0].size(); ++xi) {
                        tensor coords = tensor(batch(bi), feature(ofi), spatial(xi, yi, zi, 0));
                        auto offset = out_lay.get_linear_offset(coords);
                        auto val = out_ptr[offset];
                        auto val_ref = expected_result[bi][ofi][zi][yi][xi];
                        auto equal = are_equal(val_ref, val, 1e-2f);
                        if (!equal) {
                            std::cout << "Value at batch: " << bi << ", output_f: " << ofi << ", z: " << zi << ", y: " << yi << ", x: " << xi << " = " << val << std::endl;
                            std::cout << "Reference value at batch: " << bi << ", output_f: " << ofi << ", z: " << zi << ", y: " << yi << ", x: " << xi << " = " << val_ref << std::endl;
                        }
                        ASSERT_TRUE(equal);
                    }
}

INSTANTIATE_TEST_SUITE_P(conv_fp16_cases,
                        convolution_general_gpu,
                        ::testing::Values(
                            // Input X size, Input Y size, Input Z size, Input features, Output features,
                            // Kernel size X, Kernel size Y, Kernel size Z, Groups number, Stride, Batch,
                            // Input data format, Implementation name, WithBias
                            TestParamType_general_convolution_gpu(8, 8, 1, 8, 16, 3, 3, 1, 1, 1, 16, format::fs_b_yx_fsv32, "convolution_gpu_fs_byx_fsv32", true),
                            TestParamType_general_convolution_gpu(12, 12, 1, 4, 16, 3, 3, 1, 1, 1, 2, format::fs_b_yx_fsv32, "convolution_gpu_fs_byx_fsv32", false),
                            TestParamType_general_convolution_gpu(11, 11, 1, 96, 48, 3, 3, 1, 1, 1, 2, format::fs_b_yx_fsv32, "convolution_gpu_fs_byx_fsv32", true),
                            TestParamType_general_convolution_gpu(12, 12, 1, 32, 48, 3, 3, 1, 1, 1, 2, format::fs_b_yx_fsv32, "convolution_gpu_fs_byx_fsv32", false),
                            TestParamType_general_convolution_gpu(7, 7, 1, 16, 16, 3, 3, 1, 1, 1, 16, format::fs_b_yx_fsv32, "convolution_gpu_fs_byx_fsv32", true),
                            TestParamType_general_convolution_gpu(7, 8, 1, 20, 64, 4, 4, 1, 1, 1, 2, format::fs_b_yx_fsv32, "convolution_gpu_fs_byx_fsv32", false),
                            TestParamType_general_convolution_gpu(5, 5, 1, 80, 64, 3, 3, 1, 1, 1, 2, format::fs_b_yx_fsv32, "convolution_gpu_fs_byx_fsv32", false),
                            TestParamType_general_convolution_gpu(7, 7, 1, 32, 64, 4, 4, 1, 1, 1, 2, format::fs_b_yx_fsv32, "convolution_gpu_fs_byx_fsv32", true),
                            TestParamType_general_convolution_gpu(5, 5, 1, 32, 64, 3, 3, 1, 1, 1, 2, format::fs_b_yx_fsv32, "convolution_gpu_fs_byx_fsv32", true),
                            TestParamType_general_convolution_gpu(12, 10, 1, 32, 64, 5, 5, 1, 1, 1, 2, format::fs_b_yx_fsv32, "convolution_gpu_fs_byx_fsv32", false),
                            TestParamType_general_convolution_gpu(5, 5, 1, 32, 64, 3, 3, 1, 1, 1, 2, format::fs_b_yx_fsv32, "convolution_gpu_fs_byx_fsv32", false),
                            TestParamType_general_convolution_gpu(5, 5, 1, 64, 64, 3, 3, 1, 1, 2, 2, format::fs_b_yx_fsv32, "convolution_gpu_fs_byx_fsv32", true)
                        ),
                        convolution_general_gpu::PrintToStringParamName);

TEST_P(convolution_general_gpu, conv_fp16_cases) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int input_x = testing::get<0>(GetParam()),
              input_y = testing::get<1>(GetParam()),
              input_f = testing::get<3>(GetParam()),
              output_f = testing::get<4>(GetParam()),
              filter_x = testing::get<5>(GetParam()),
              filter_y = testing::get<6>(GetParam()),
              groups = testing::get<8>(GetParam()),
              batch_num = testing::get<10>(GetParam()),
              output_padding = 0,
              pad_y = (filter_y - 1) / 2,
              pad_x = (filter_x - 1) / 2;
    const uint64_t stride = testing::get<9>(GetParam());
    auto input_data_format = testing::get<11>(GetParam());
    auto impl_name = testing::get<12>(GetParam());
    auto with_bias = testing::get<13>(GetParam());

    auto input_size = tensor(batch_num, input_f, input_x, input_y);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_y, input_x, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_y, filter_x, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, filter_y, filter_x, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::bfyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto expected_result = VVVVF<ov::float16>(batch_num, VVVF<ov::float16>(output_f));
    topology topology;

    // Calculate reference values
    if (with_bias) {
        auto biases_size = tensor(1, output_f, 1, 1);
        auto biases_data = rg.generate_random_1d<ov::float16>(output_f, -1, 1);
        auto biases_mem = engine.allocate_memory({ data_types::f16, format::bfyx, biases_size });
        set_values(biases_mem, biases_data);

        for (auto bi = 0; bi < batch_num; ++bi) {
            for (auto ofi = 0; ofi < output_f; ++ofi) {
                expected_result[bi][ofi] = reference_convolve(input_data[bi],                    // input
                                                              weights_data[ofi],                 // weights
                                                              stride, stride,                    // strides
                                                              biases_data[ofi],                  // bias
                                                              1, 1,                              // dilation
                                                              pad_y, pad_x,                      // input padding
                                                              output_padding, output_padding);   // output_padding
            }
        }

        topology.add(input_layout("input", input_mem->get_layout()),
                     data("weights_fsv", weights_mem),
                     data("bias", biases_mem),
                     reorder("input_fsv", input_info("input"), { data_types::f16, input_data_format, input_size }));

        auto conv_fsv = convolution("conv_fsv",
                                    input_info("input_fsv"),
                                    "weights_fsv",
                                    "bias",
                                    groups,
                                    { stride, stride },
                                    {1, 1},
                                    { pad_y, pad_x },
                                    { pad_y, pad_x },
                                    false);
        conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

        topology.add(conv_fsv);
    } else {
        for (auto bi = 0; bi < batch_num; ++bi) {
            for (auto ofi = 0; ofi < output_f; ++ofi) {
                expected_result[bi][ofi] = reference_convolve(input_data[bi],                    // input
                                                              weights_data[ofi],                 // weights
                                                              stride, stride,                    // strides
                                                              0,                                 // bias
                                                              1, 1,                              // dilation
                                                              pad_y, pad_x,                      // input padding
                                                              output_padding, output_padding);   // output_padding
            }
        }

        topology.add(input_layout("input", input_mem->get_layout()),
                     data("weights_fsv", weights_mem),
                     reorder("input_fsv", input_info("input"), { data_types::f16, input_data_format, input_size }));

        auto conv_fsv = convolution("conv_fsv",
                                    input_info("input_fsv"),
                                    "weights_fsv",
                                    no_bias,
                                    groups,
                                    { stride, stride },
                                    {1, 1},
                                    { pad_y, pad_x },
                                    { pad_y, pad_x },
                                    false);
        conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};
        topology.add(conv_fsv);
    }
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc conv_impl = { input_data_format, impl_name };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_fsv", conv_impl } }));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);
    network.execute();

    auto out_mem = network.get_output("conv_fsv").get_memory();
    cldnn::mem_lock<ov::float16> out_ptr(out_mem, get_test_stream());
    auto out_lay = out_mem->get_layout();

    ASSERT_EQ(out_mem->get_layout().format, input_data_format);
    ASSERT_EQ(out_lay.batch(), expected_result.size());
    ASSERT_EQ(out_lay.feature(), expected_result[0].size());
    ASSERT_EQ(out_lay.spatial(1), expected_result[0][0].size());
    ASSERT_EQ(out_lay.spatial(0), expected_result[0][0][0].size());

    for (int bi = 0; bi < out_lay.batch(); ++bi)
        for (int ofi = 0; ofi < out_lay.feature(); ++ofi)
            for (int yi = 0; yi < out_lay.spatial(1); ++yi)
                for (int xi = 0; xi < out_lay.spatial(0); ++xi) {
                    tensor coords = tensor(batch(bi), feature(ofi), spatial(xi, yi, 0, 0));
                    auto offset = out_lay.get_linear_offset(coords);
                    auto val = out_ptr[offset];
                    auto val_ref = expected_result[bi][ofi][yi][xi];
                    auto equal = are_equal(val_ref, val, 1);
                    if (!equal) {
                        std::cout << "Value at batch: " << bi << ", output_f: " << ofi
                                    << ", y: " << yi << ", x: " << xi << " = " << static_cast<float>(val) << std::endl;
                        std::cout << "Reference value at batch: " << bi << ", output_f: " << ofi << ", y: " << yi
                                  << ", x: " << xi << " = " << static_cast<float>(val_ref) << std::endl;
                    }
                    ASSERT_TRUE(equal);
                }
}

struct convolution_gpu_fsv16_to_bfyx : public convolution_general_gpu {};

INSTANTIATE_TEST_SUITE_P(conv_b_fs_yx_fsv16_to_bfyx,
                        convolution_gpu_fsv16_to_bfyx,
                        ::testing::Values(
                            // Input X size, Input Y size, Input Z size, Input features, Output features,
                            // Kernel size X, Kernel size Y, Kernel size Z, Groups number, Stride, Batch,
                            // Input data format, Implementation name, WithBias
                            TestParamType_general_convolution_gpu(6, 6, 0, 16, 16, 3, 3, 0, 1, 1, 4, format::b_fs_yx_fsv16, "convolution_gpu_fsv16_to_bfyx", false),
                            TestParamType_general_convolution_gpu(6, 6, 0, 32, 32, 3, 3, 0, 1, 1, 1, format::b_fs_yx_fsv16, "convolution_gpu_fsv16_to_bfyx", false),
                            TestParamType_general_convolution_gpu(6, 6, 0, 16, 16, 3, 3, 0, 1, 1, 16, format::b_fs_yx_fsv16, "convolution_gpu_fsv16_to_bfyx", false),
                            TestParamType_general_convolution_gpu(16, 6, 0, 20, 16, 3, 3, 0, 1, 1, 20, format::b_fs_yx_fsv16, "convolution_gpu_fsv16_to_bfyx", false)
                        ),
                        convolution_gpu_fsv16_to_bfyx::PrintToStringParamName);

TEST_P(convolution_gpu_fsv16_to_bfyx, conv_b_fs_yx_fsv16_to_bfyx_padding)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int input_b = testing::get<10>(GetParam());
    const int input_f = testing::get<3>(GetParam());
    const int input_y = testing::get<1>(GetParam());
    const int input_x = testing::get<0>(GetParam());

    const int filter_x = testing::get<5>(GetParam());
    const int filter_y = testing::get<6>(GetParam());
    const uint64_t stride = testing::get<9>(GetParam());

    const std::ptrdiff_t pad_y = (filter_y - 1) / 2;
    const std::ptrdiff_t pad_x = (filter_x - 1) / 2;

    auto input_size = tensor(input_b, input_f, input_x, input_y);
    auto input_data = rg.generate_random_4d<ov::float16>(input_b, input_f, input_y, input_x, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(input_b, input_f, filter_x, filter_y, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(input_b, input_f, filter_x, filter_y, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::oiyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Set topology
    topology topology(
        input_layout("input_origin", input_mem->get_layout()),
        data("weights_fsv", weights_mem),
        reorder("input_fsv16", input_info("input_origin"), { data_types::f16, format::b_fs_yx_fsv16, input_size }));    // format 3 to 8

    // Add convolution
    ov::CoordinateDiff input_padding_before = { pad_y, pad_x };
    ov::CoordinateDiff input_padding_after = { pad_y, pad_x };

    auto conv_fsv = convolution("conv_fsv", input_info("input_fsv16"), "weights_fsv", no_bias, 1, {stride, stride}, {1, 1}, input_padding_before, input_padding_after, false);
    conv_fsv.output_paddings = {padding({ 0, 32, 2, 2 }, 0.f)};
    topology.add(conv_fsv);                                                                                 // format 8 to 8 -> after fusing, format 8 to 3

    // Add reorder to bfyx
    auto reorder_bfyx = reorder("reorder_bfyx", input_info("conv_fsv"), { data_types::f16, format::bfyx, input_size });
    reorder_bfyx.output_paddings = {padding({ 0, 16, 1, 1 }, 0.f)};
    topology.add(reorder_bfyx);                                                                             // format 8 to 3 -> after fusing, removed

    // Exec ref network (non-fusing)
    ExecutionConfig config_ref = get_test_default_config(engine);
    config_ref.set_property(ov::intel_gpu::optimize_data(false));
    config_ref.set_property(ov::intel_gpu::allow_static_input_reorder(true));

    network network_ref(engine, topology, config_ref);
    network_ref.set_input_data("input_origin", input_mem);
    auto ref_out = network_ref.execute();

    auto ref_out_mem = ref_out.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> ref_out_ptr(ref_out_mem, get_test_stream());

    // Exec target network (fusing: conv+reorder)
    ExecutionConfig config_target = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "convolution_gpu_bfyx_f16" };
    config_target.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_fsv", conv_impl } }));
    config_target.set_property(ov::intel_gpu::optimize_data(true));

    network network_target(engine, topology, config_target);
    network_target.set_input_data("input_origin", input_mem);
    auto target_out = network_target.execute();

    auto target_out_mem = target_out.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> target_out_ptr(target_out_mem, get_test_stream());

    // Compare ref and target result
    for (size_t i = 0; i < ref_out_ptr.size(); i++) {
        auto ref_val = static_cast<float>(ref_out_ptr[i]);
        auto target_val = static_cast<float>(target_out_ptr[i]);
        auto diff = std::fabs(ref_val - target_val);
        auto equal = (diff > 1e-5f) ? false : true;

        ASSERT_TRUE(equal);
        if (!equal)
        {
            std::cout << "i:" << i \
                << "\t ref_out = " << ref_val \
                << "\t target_out = " << target_val \
                << std::endl;

            break;
        }
    }
}

TEST_P(convolution_gpu_fsv16_to_bfyx, conv_b_fs_yx_fsv16_to_bfyx_different_type)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int input_b = testing::get<10>(GetParam());
    const int input_f = testing::get<3>(GetParam());
    const int input_y = testing::get<1>(GetParam());
    const int input_x = testing::get<0>(GetParam());

    const int filter_x = testing::get<5>(GetParam());
    const int filter_y = testing::get<6>(GetParam());
    const uint64_t stride = testing::get<9>(GetParam());

    const std::ptrdiff_t pad_y = (filter_y - 1) / 2;
    const std::ptrdiff_t pad_x = (filter_x - 1) / 2;

    auto input_size = tensor(input_b, input_f, input_x, input_y);
    auto input_data = rg.generate_random_4d<ov::float16>(input_b, input_f, input_y, input_x, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(input_b, input_f, filter_x, filter_y, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(input_b, input_f, filter_x, filter_y, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::goiyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Set topology
    topology topology(
        input_layout("input_origin", input_mem->get_layout()),
        data("weights_fsv", weights_mem),
        reorder("input_fsv16", input_info("input_origin"), { data_types::f16, format::b_fs_yx_fsv16, input_size }));    // format 3 to 8
    auto conv_fsv = convolution("conv_fsv", input_info("input_fsv16"), "weights_fsv", no_bias, 1, {stride, stride}, {1, 1}, {pad_y, pad_x}, {pad_y, pad_x}, true);
    topology.add(conv_fsv);                                                                                 // format 8 to 8 -> after fusing, format 8 to 3

    // Add reorder to bfyx
    auto reorder_bfyx = reorder("reorder_bfyx", input_info("conv_fsv"), { data_types::f32, format::bfyx, input_size });
    topology.add(reorder_bfyx);                                                                             // format 8 to 3 -> after fusing, removed

    // Exec ref network (non-fusing)
    ExecutionConfig config_ref = get_test_default_config(engine);
    config_ref.set_property(ov::intel_gpu::optimize_data(false));
    config_ref.set_property(ov::intel_gpu::allow_static_input_reorder(true));

    network network_ref(engine, topology, config_ref);
    network_ref.set_input_data("input_origin", input_mem);
    auto ref_out = network_ref.execute();

    auto ref_out_mem = ref_out.begin()->second.get_memory();
    cldnn::mem_lock<float> ref_out_ptr(ref_out_mem, get_test_stream());

    // Exec target network (fusing: conv+reorder)
    ExecutionConfig config_target = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "convolution_gpu_bfyx_f16" };
    config_target.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_fsv", conv_impl } }));
    config_target.set_property(ov::intel_gpu::optimize_data(true));

    network network_target(engine, topology, config_target);
    network_target.set_input_data("input_origin", input_mem);
    auto target_out = network_target.execute();

    auto target_out_mem = target_out.begin()->second.get_memory();
    cldnn::mem_lock<float> target_out_ptr(target_out_mem, get_test_stream());

    // Compare ref and target result
    for (size_t i = 0; i < ref_out_ptr.size(); i++) {
        auto ref_val = static_cast<float>(ref_out_ptr[i]);
        auto target_val = static_cast<float>(target_out_ptr[i]);
        auto diff = std::abs(ref_val - target_val);
        auto equal = (diff > 1e-5f) ? false : true;

        ASSERT_TRUE(equal);
        if (!equal)
        {
            std::cout << "i:" << i \
                << "\t ref_out = " << ref_val \
                << "\t target_out = " << target_val \
                << std::endl;

            break;
        }
    }
}

template <typename InputT, typename WeightsT, typename OutputT>
class convolution_test_base {
public:
    virtual topology build_topology(cldnn::engine& engine) {
        auto input_lay = layout(input_type(), format::bfyx, input_size(), padding_size());
        auto wei_lay = grouped_weights_shape() ? layout(weights_type(), format::bfzyx, grouped_weights_size()) : layout(weights_type(), format::bfyx, weights_size());

        auto wei_mem = engine.allocate_memory(wei_lay);
        auto weights_flat = grouped_weights_shape() ? flatten_5d(format::bfzyx, _grouped_weights) : flatten_4d(format::bfyx, _weights);
        set_values(wei_mem, weights_flat);
        layout reordered_layout = layout{ input_type(), input_format(), input_size(), padding_size() };
        auto topo = topology();
        topo.add(input_layout("input", input_lay));
        topo.add(reorder("input_reorder", input_info("input"), reordered_layout));
        std::string input_id = "input_reorder";
        std::string azp_id = "";
        std::string wzp_id = "";
        std::string bias_id = "";
        std::string comp_id = "";

        if (has_input_zp()) {
            auto input_zp_lay = layout(this->input_type(), format::bfyx, tensor(feature(this->input_features())));
            auto input_comp_lay = layout(ov::element::f32, format::bfyx, tensor(feature(this->output_features())));
            auto input_zp_mem = engine.allocate_memory(input_zp_lay);
            auto input_comp_mem = engine.allocate_memory(input_comp_lay);
            set_values(input_zp_mem, this->_input_zp);
            set_values(input_comp_mem, this->_compensation);
            azp_id = "input_zp";
            comp_id = "compensation";
            topo.add(data(azp_id, input_zp_mem));
            topo.add(data(comp_id, input_comp_mem));
        }
        topo.add(data("weights", wei_mem));
        std::string weights_id = "weights";
        if (has_weights_zp()) {
            auto weights_zp_lay = layout(weights_type(), format::bfyx, tensor(batch(output_features())));
            auto weights_zp_mem = engine.allocate_memory(weights_zp_lay);
            set_values(weights_zp_mem, _weights_zp);
            wzp_id = "weights_zp";
            topo.add(data(wzp_id, weights_zp_mem));
        }
        if (this->has_bias()) {
            auto bias_lay = layout(this->output_type(), format::bfyx, tensor(feature(this->output_features())));
            auto bias_mem = engine.allocate_memory(bias_lay);
            bias_id = "bias";
            set_values(bias_mem, this->_bias);
            topo.add(data(bias_id, bias_mem));
        }

        auto conv_prim = convolution(
                "conv",
                input_info(input_id),
                weights_id,
                bias_id,
                wzp_id,
                azp_id,
                comp_id,
                static_cast<uint32_t>(this->groups()),
                {static_cast<uint64_t>(this->_stride_y), static_cast<uint64_t>(this->_stride_x)},
                {static_cast<uint64_t>(this->_dilation_y), static_cast<uint64_t>(this->_dilation_x)},
                {static_cast<std::ptrdiff_t>(this->_offset_y), static_cast<std::ptrdiff_t>(this->_offset_x)},
                {static_cast<std::ptrdiff_t>(this->_offset_y), static_cast<std::ptrdiff_t>(this->_offset_x)},
                this->grouped_weights_shape(),
                this->output_type());
        topo.add(conv_prim);

        return topo;
    }

    virtual primitive_id output_primitive_id() const {
        return "conv";
    }

    virtual void run_expect(const VVVVF<OutputT>& expected) {
        auto& engine = get_test_engine();

        auto topo = build_topology(engine);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", { input_format(), "" } } }));

        auto prog = program::build_program(engine, topo, config);

        cldnn::network net(prog, 0);

        auto input_lay = layout(input_type(), format::bfyx, input_size(), padding_size());
        auto input_mem = engine.allocate_memory(input_lay);
        std::vector<InputT> input_flat(input_lay.get_linear_size(), static_cast<InputT>(0));
        for (size_t bi = 0; bi < batch_num(); ++bi)
            for (size_t fi = 0; fi < input_features(); ++fi)
                for (size_t yi = 0; yi < input_y(); ++yi)
                    for (size_t xi = 0; xi < input_x(); ++xi) {
                        tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        size_t offset = input_lay.get_linear_offset(coords);
                        input_flat[offset] = _input[bi][fi][yi][xi];
                    }
        set_values(input_mem, input_flat);

        net.set_input_data("input", input_mem);
        auto result = net.execute();
        auto out_mem = result.at(output_primitive_id()).get_memory();
        auto out_lay = out_mem->get_layout();
        cldnn::mem_lock<OutputT> out_ptr(out_mem, get_test_stream());

        std::stringstream description;
        for (auto i : net.get_primitives_info()) {
            if (i.original_id == "conv") {
                std::cout << i.kernel_id << std::endl;
                description << "  kernel: " << i.kernel_id << std::endl;
            }
        }
        description << "  executed: ";
        for (auto e : net.get_executed_primitive_ids()) {
            description << e << ", ";
        }

        ASSERT_EQ(out_lay.data_type, output_type());
        ASSERT_EQ(out_lay.batch(), expected.size());
        ASSERT_EQ(out_lay.feature(), expected[0].size());
        ASSERT_EQ(out_lay.spatial(1), expected[0][0].size());
        ASSERT_EQ(out_lay.spatial(0), expected[0][0][0].size());

        for (size_t bi = 0; bi < batch_num(); ++bi)
            for (size_t fi = 0; fi < output_features(); ++fi)
                for (size_t yi = 0; yi < expected[0][0].size(); ++yi)
                    for (size_t xi = 0; xi < expected[0][0][0].size(); ++xi) {
                        tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        size_t offset = out_lay.get_linear_offset(coords);

                        ASSERT_EQ(out_ptr[offset], expected[bi][fi][yi][xi])
                            << "at b= " << bi << ", f= " << fi << ", y= " << yi << ", x= " << xi << std::endl
                            << description.str();
                    }
    }

    void set_input(format::type fmt, VVVVF<InputT> input) {
        _input_fmt = fmt;
        _input = std::move(input);
    }

    void set_weights(VVVVF<WeightsT> weights) {
        _weights = std::move(weights);
    }

    void set_grouped_weights(VVVVVF<WeightsT> grouped_weights) {
        _grouped_weights = std::move(grouped_weights);
    }

    void set_bias(VF<OutputT> bias) {
        _bias = std::move(bias);
    }

    void set_strides(int stride_x, int stride_y) {
        _stride_x = stride_x;
        _stride_y = stride_y;
    }

    void set_offsets(int offset_x, int offset_y) {
        _offset_x = offset_x;
        _offset_y = offset_y;
    }

    void set_dilation(int dilation_x, int dilation_y) {
        _dilation_x = dilation_x;
        _dilation_y = dilation_y;
    }

    void set_input_zp(VF<InputT> input_zp) {
        _input_zp = std::move(input_zp);
    }

    void set_weights_zp(VF<WeightsT> weights_zp) {
        _weights_zp = std::move(weights_zp);
    }

    void set_compensation(VF<float> compensation) {
        _compensation = std::move(compensation);
    }

    void set_padded_input(bool padded_input) {
        _padded_input = padded_input;
    }

    void set_bigger_pad(bool bigger_pad) {
        _bigger_pad = bigger_pad;
    }

    void set_grouped_weights_shape(bool grouped_weights_shape) {
        _grouped_weights_shape = grouped_weights_shape;
    }

protected:
    VVVVF<InputT> _input;
    VVVVF<WeightsT> _weights;
    VVVVVF<WeightsT> _grouped_weights;
    VF<OutputT> _bias;
    VF<InputT> _input_zp;
    VF<WeightsT> _weights_zp;
    VF<float> _compensation;
    format::type _input_fmt;
    int _stride_x, _stride_y;
    int _offset_x, _offset_y;
    int _dilation_x, _dilation_y;
    bool _padded_input;
    bool _bigger_pad;
    bool _grouped_weights_shape;

    size_t batch_num() const { return _input.size(); }
    size_t input_features() const { return _input[0].size(); }
    size_t input_x() const { return _input[0][0][0].size(); }
    size_t input_y() const { return _input[0][0].size(); }
    size_t output_features() const { return _grouped_weights_shape ? _grouped_weights[0].size() : _weights.size(); }
    size_t weights_input_features() const { return _grouped_weights_shape ? _grouped_weights[0][0].size() : _weights[0].size(); }
    size_t filter_x() const { return _grouped_weights_shape ? _grouped_weights[0][0][0][0].size() : _weights[0][0][0].size(); }
    size_t filter_y() const { return _grouped_weights_shape ? _grouped_weights[0][0][0].size() : _weights[0][0].size(); }
    size_t groups() const { return input_features() / weights_input_features(); }

    bool has_bias() { return _bias.size() > 0; }
    bool has_input_zp() { return _input_zp.size() > 0; }
    bool has_weights_zp() { return _weights_zp.size() > 0; }
    bool need_padded_input() { return _padded_input; }
    bool bigger_pad() { return _bigger_pad; }
    bool grouped_weights_shape() { return _grouped_weights_shape; }

    data_types input_type() const { return ov::element::from<InputT>(); }
    format input_format() const { return _input_fmt; }
    tensor input_size() const {
        return tensor(TensorValue(batch_num()),
                      TensorValue(input_features()),
                      TensorValue(input_x()),
                      TensorValue(input_y()));
    }

    data_types weights_type() const { return ov::element::from<WeightsT>(); }
    tensor weights_size() const {
        return tensor(TensorValue(output_features()),
                      TensorValue(weights_input_features()),
                      TensorValue(filter_x()),
                      TensorValue(filter_y()));
    }
    tensor grouped_weights_size() const {
        return tensor(TensorValue(groups()),
                      TensorValue(output_features()),
                      TensorValue(filter_x()),
                      TensorValue(filter_y()),
                      TensorValue(weights_input_features()));
    }
    padding padding_size() const {
        if (_padded_input) {
            if (_bigger_pad) {
                return padding{
                        tensor(0, 0, TensorValue(filter_x() / 2 + 1), TensorValue(filter_y() / 2 + 1), 0, 0).sizes(),
                        tensor(0, 0, TensorValue(filter_x() / 2 + 1), TensorValue(filter_y() / 2 + 1), 0, 0).sizes()};
            } else {
                return padding{
                        tensor(0, 0, TensorValue(filter_x() / 2), TensorValue(filter_y() / 2), 0, 0).sizes(),
                        tensor(0, 0, TensorValue(filter_x() / 2), TensorValue(filter_y() / 2), 0, 0).sizes()};
            }
        } else {
            return padding{};
        }
    }

    data_types output_type() const { return ov::element::from<OutputT>(); }
};

struct convolution_random_test_all_params {
    static constexpr size_t spatials = 2;
    size_t batch;
    size_t input_features;
    size_t output_features;
    std::array<size_t, spatials> input_xy;
    std::array<size_t, spatials> filter_xy;
    std::array<int, spatials> stride_xy;
    std::array<int, spatials> offset_xy;
    std::array<int, spatials> dilation_xy;
    bool with_bias;
    size_t groups;
    format::type input_format;
    bool asymmetric_weights;
    bool asymmetric_data;
    bool need_padded_input;
    bool bigger_pad;
    bool grouped_weights_shape;
};

template <typename InputT, typename WeightsT, typename OutputT>
class convolution_random_test_base : public convolution_test_base<InputT, WeightsT, OutputT> {
public:
    virtual VVVVF<OutputT> calculate_reference() {
        VVVVF<OutputT> expected = VVVVF<OutputT>(this->batch_num(), VVVF<OutputT>(this->output_features()));
        bool depthwise = this->groups() == this->input_features();
        bool grouped = (this->groups() > 1 && !depthwise) ? true : false;
        size_t group_size = this->output_features() / this->groups();
        for (size_t bi = 0; bi < this->batch_num(); ++bi)
        for (size_t fi = 0; fi < this->output_features(); ++fi) {
            size_t f_begin = depthwise ? fi : 0;
            size_t f_end = (depthwise ? fi : 0) + this->weights_input_features();
            auto bias = this->has_bias() ? this->_bias[fi] : static_cast<OutputT>(0);
            auto weights_zp = this->has_weights_zp() ? this->_weights_zp[fi] : static_cast<WeightsT>(0);
            expected[bi][fi] = reference_convolve<InputT, OutputT, WeightsT>(
                this->_input[bi],
                (this->_grouped_weights_shape ? this->_grouped_weights[fi / group_size][fi % group_size] : this->_weights[fi]),
                this->_stride_y,
                this->_stride_x,
                static_cast<float>(bias),
                this->_dilation_y,
                this->_dilation_x,
                this->_offset_y,
                this->_offset_x,
                0,
                0,
                f_begin,
                f_end,
                depthwise,
                grouped,
                this->_input_zp,
                weights_zp);
        }
        return expected;
    }

    virtual void param_set_up(const convolution_random_test_all_params& params) {
        auto& rg = get_random_generator();
        rg.set_seed(GET_SUITE_NAME);
        auto wei_in_f = params.input_features / params.groups;

        auto input_data = rg.template generate_random_4d<InputT>(
            params.batch, params.input_features, params.input_xy[1], params.input_xy[0], -256, 256);
        if (params.grouped_weights_shape) {
            auto weights_data = rg.template generate_random_5d<WeightsT>(
                params.groups, (params.output_features / params.groups), wei_in_f, params.filter_xy[1], params.filter_xy[0], -256, 256);
            this->set_grouped_weights(std::move(weights_data));
        } else {
            auto weights_data = rg.template generate_random_4d<WeightsT>(
                params.output_features, wei_in_f, params.filter_xy[1], params.filter_xy[0], -256, 256);
            this->set_weights(std::move(weights_data));
        }
        auto bias_data = params.with_bias ? rg.template generate_random_1d<OutputT>(params.output_features, -256, 256) : VF<OutputT>();
        auto weights_zp_data = params.asymmetric_weights ? rg.template generate_random_1d<WeightsT>(params.output_features, -256, 256) : VF<WeightsT>();
        auto input_zp_data = params.asymmetric_data ? rg.template generate_random_1d<InputT>(params.input_features, -256, 256) : VF<InputT>();

        this->set_input(params.input_format, std::move(input_data));
        this->set_bias(std::move(bias_data));
        this->set_strides(params.stride_xy[0], params.stride_xy[1]);
        this->set_offsets(params.offset_xy[0], params.offset_xy[1]);
        this->set_dilation(params.dilation_xy[0], params.dilation_xy[1]);
        this->set_weights_zp(std::move(weights_zp_data));
        this->set_input_zp(std::move(input_zp_data));
        this->set_padded_input(params.need_padded_input);
        this->set_bigger_pad(params.bigger_pad);
        this->set_grouped_weights_shape(params.grouped_weights_shape);

        auto spatial_size = this->filter_x() * this->filter_y();
        auto w = params.grouped_weights_shape ? flatten_5d(format::bfzyx, this->_grouped_weights) : flatten_4d(format::bfyx, this->_weights);
        auto compensation = params.asymmetric_data ? get_compensation(w, this->_input_zp, this->_weights_zp,
                                                                      this->groups(), this->output_features() / this->groups(),
                                                                      this->input_features() / this->groups(), spatial_size)
                                                   : VF<float>();
        this->set_compensation(std::move(compensation));
    }

    void run_random(const convolution_random_test_all_params& params) {
        param_set_up(params);

        VVVVF<OutputT> expected = calculate_reference();
        ASSERT_NO_FATAL_FAILURE(this->run_expect(expected));
    }

    VF<float> get_compensation(VF<WeightsT> w_tensor, VF<InputT> azp_tensor, VF<WeightsT> wzp_tensor,
                                 int64_t groups, size_t output_channels, size_t input_channels, size_t spatial_size) {
        size_t azp_total = std::max<size_t>(azp_tensor.size(), 1);
        size_t wzp_total = std::max<size_t>(wzp_tensor.size(), 1);
        const size_t groups_count = std::max<int64_t>(groups, 1);
        VF<float> compensation(output_channels * groups_count);

        float* comp = compensation.data();
        const WeightsT* w = w_tensor.data();
        const WeightsT* wzp = !wzp_tensor.empty() ? wzp_tensor.data() : nullptr;
        const InputT* azp = !azp_tensor.empty() ? azp_tensor.data() : nullptr;

        if (!azp)
            return {};

        for (size_t g = 0; g < groups_count; g++) {
            for (size_t oc = 0; oc < output_channels; oc++) {
                float c = 0.f;
                for (size_t ic = 0; ic < input_channels; ic++) {
                    for (size_t k = 0; k < spatial_size; k++) {
                        size_t azp_offset = (g * input_channels + ic) % azp_total;
                        size_t wzp_offset = (g * output_channels + oc) % wzp_total;
                        const auto w_offset = g * output_channels * input_channels * spatial_size
                                            + oc * input_channels * spatial_size
                                            + ic * spatial_size
                                            + k;

                        c += w[w_offset] * azp[azp_offset];
                        if (wzp) {
                            c -= azp[azp_offset] * wzp[wzp_offset];
                        }
                    }
                }
                comp[(g * output_channels + oc)] = -c;
            }
        }

        return compensation;
    }

    tests::random_generator _rg;
    tests::random_generator& get_random_generator() { return _rg; }
};

// construct a readable name in format as follows:
// <out format>_i<input>_w<weights>_s<stride>_ofs<offset>_d<dilation>_g<groups>_<bias>
static std::string to_string_convolution_all_params(const testing::TestParamInfo<convolution_random_test_all_params>& param_info) {
    auto& params = param_info.param;
    int Batch = (int)params.batch;
    int iF = (int)params.input_features;
    int oF = (int)params.output_features;
    auto& iSize = params.input_xy;
    auto& fSize = params.filter_xy;
    auto& Stride = params.stride_xy;
    auto& Offset = params.offset_xy;
    auto& Dilation = params.dilation_xy;
    auto groups = params.groups;
    bool Bias = params.with_bias;
    format::type iType = params.input_format;  // input format
    bool asymm_weights = params.asymmetric_weights;
    bool asymm_input = params.asymmetric_data;
    bool padded_input = params.need_padded_input;
    bool bigger_pad = params.bigger_pad;
    bool grouped_weights_shape = params.grouped_weights_shape;
    // Wrapper for negative walues as ex. "-1" will generate invalid gtest param string
    auto to_string_neg = [](int val) {
        if (val >= 0)
            return std::to_string(val);
        else
            return "m" + std::to_string(-val);
    };

    return fmt_to_str(iType) +
        "_i" + std::to_string(Batch) + 'x' + std::to_string(iF) + 'x' + std::to_string(iSize[0]) + 'x' + std::to_string(iSize[1]) +
        "_w" + std::to_string(oF) + 'x' + std::to_string(iF) + 'x' + std::to_string(fSize[0]) + 'x' + std::to_string(fSize[1]) +
        "_s" + std::to_string(Stride[0]) + 'x' + std::to_string(Stride[1]) +
        "_ofs" + to_string_neg(Offset[0]) + 'x' + to_string_neg(Offset[1]) +
        "_d" + std::to_string(Dilation[0]) + 'x' + std::to_string(Dilation[1]) +
        "_g" + std::to_string(groups) +
        (Bias ? "_bias" : "") + (asymm_weights ? "_wzp" : "") + (asymm_input ? "_izp" : "") +
        (padded_input ? "_in_pad" : "") +
        (bigger_pad ? "_bigger_pad" : "") + (grouped_weights_shape ? "_grouped_weights" : "");
}

template <typename InputT, typename WeightsT, typename OutputT>
class convolution_random_test_fsv4_input : public convolution_random_test_base<InputT, WeightsT, OutputT> {
public:
    using parent = convolution_random_test_base<InputT, WeightsT, OutputT>;
    topology build_topology(cldnn::engine& engine) override {
        auto input_lay = layout(this->input_type(), format::b_fs_yx_fsv4, this->input_size(), this->padding_size());
        auto wei_lay = this->grouped_weights_shape() ? layout(this->weights_type(), format::bfzyx, this->grouped_weights_size()) :
                       layout(this->weights_type(), format::bfyx, this->weights_size());

        auto wei_mem = engine.allocate_memory(wei_lay);
        auto wei_flat = this->grouped_weights_shape() ? flatten_5d(format::bfzyx, this->_grouped_weights) : flatten_4d(format::bfyx, this->_weights);
        set_values(wei_mem, wei_flat);
        layout reordered_layout = layout{ this->input_type(), this->input_format(), this->input_size(), this->padding_size() };
        auto topo = topology();
        topo.add(input_layout("input", input_lay));
        topo.add(reorder("input_reorder", input_info("input"), reordered_layout));
        std::string input_id = "input_reorder";
        std::string azp_id = "";
        std::string wzp_id = "";
        std::string bias_id = "";
        std::string comp_id = "";
        if (this->has_input_zp()) {
            auto input_zp_lay = layout(this->input_type(), format::bfyx, tensor(feature(this->input_features())));
            auto input_comp_lay = layout(ov::element::f32, format::bfyx, tensor(feature(this->output_features())));
            auto input_zp_mem = engine.allocate_memory(input_zp_lay);
            auto input_comp_mem = engine.allocate_memory(input_comp_lay);
            set_values(input_zp_mem, this->_input_zp);
            set_values(input_comp_mem, this->_compensation);
            azp_id = "input_zp";
            comp_id = "compensation";
            topo.add(data(azp_id, input_zp_mem));
            topo.add(data(comp_id, input_comp_mem));
        }
        topo.add(data("weights", wei_mem));
        std::string weights_id = "weights";
        if (this->has_weights_zp()) {
            auto weights_zp_lay = layout(this->weights_type(), format::bfyx, tensor(batch(this->output_features())));
            auto weights_zp_mem = engine.allocate_memory(weights_zp_lay);
            set_values(weights_zp_mem, this->_weights_zp);
            wzp_id = "weights_zp";
            topo.add(data(wzp_id, weights_zp_mem));
        }
        if (this->has_bias()) {
            auto bias_lay = layout(this->output_type(), format::bfyx, tensor(feature(this->output_features())));
            auto bias_mem = engine.allocate_memory(bias_lay);
            bias_id = "bias";
            set_values(bias_mem, this->_bias);
            topo.add(data(bias_id, bias_mem));
        }

        auto conv_prim = convolution(
                "conv",
                input_info(input_id),
                weights_id,
                bias_id,
                wzp_id,
                azp_id,
                comp_id,
                static_cast<uint32_t>(this->groups()),
                {static_cast<uint64_t>(this->_stride_y), static_cast<uint64_t>(this->_stride_x)},
                {static_cast<uint64_t>(this->_dilation_y), static_cast<uint64_t>(this->_dilation_x)},
                {static_cast<std::ptrdiff_t>(this->_offset_y), static_cast<std::ptrdiff_t>(this->_offset_x)},
                {static_cast<std::ptrdiff_t>(this->_offset_y), static_cast<std::ptrdiff_t>(this->_offset_x)},
                this->grouped_weights_shape(),
                this->output_type());
        topo.add(conv_prim);

        return topo;
    }
    void run_expect(const VVVVF<OutputT>& expected) override {
        auto& engine = get_test_engine();

        auto topo = this->build_topology(engine);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", { this->input_format(), "" } } }));

        auto prog = program::build_program(engine, topo, config);

        cldnn::network net(prog, 0);

        auto input_lay = layout(this->input_type(), format::b_fs_yx_fsv4,  this->input_size(), this->padding_size());
        auto input_mem = engine.allocate_memory(input_lay);
        std::vector<InputT> input_flat(input_lay.get_linear_size(), static_cast<InputT>(0));
        for (size_t bi = 0; bi < this->batch_num(); ++bi)
            for (size_t fi = 0; fi < this->input_features(); ++fi)
                for (size_t yi = 0; yi < this->input_y(); ++yi)
                    for (size_t xi = 0; xi < this->input_x(); ++xi) {
                        tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        size_t offset = input_lay.get_linear_offset(coords);
                        input_flat[offset] = this->_input[bi][fi][yi][xi];
                    }
        set_values(input_mem, input_flat);

        net.set_input_data("input", input_mem);
        auto result = net.execute();
        auto out_mem = result.at(this->output_primitive_id()).get_memory();
        auto out_lay = out_mem->get_layout();
        cldnn::mem_lock<OutputT> out_ptr(out_mem, get_test_stream());

        std::stringstream description;
        for (auto i : net.get_primitives_info()) {
            if (i.original_id == "conv") {
                std::cout << i.kernel_id << std::endl;
                description << "  kernel: " << i.kernel_id << std::endl;
            }
        }
        description << "  executed: ";
        for (auto e : net.get_executed_primitive_ids()) {
            description << e << ", ";
        }

        ASSERT_EQ(out_lay.data_type, this->output_type());
        ASSERT_EQ(out_lay.batch(), expected.size());
        ASSERT_EQ(out_lay.feature(), expected[0].size());
        ASSERT_EQ(out_lay.spatial(1), expected[0][0].size());
        ASSERT_EQ(out_lay.spatial(0), expected[0][0][0].size());

        for (size_t bi = 0; bi < this->batch_num(); ++bi)
            for (size_t fi = 0; fi < this->output_features(); ++fi)
                for (size_t yi = 0; yi < expected[0][0].size(); ++yi)
                    for (size_t xi = 0; xi < expected[0][0][0].size(); ++xi) {
                        tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        size_t offset = out_lay.get_linear_offset(coords);

                        ASSERT_EQ(out_ptr[offset], expected[bi][fi][yi][xi])
                            << "at b= " << bi << ", f= " << fi << ", y= " << yi << ", x= " << xi << std::endl
                            << description.str();
                    }
    }
};

template <typename InputT, typename WeightsT, typename OutputT>
class convolution_scale_random_test : public convolution_random_test_base<InputT, WeightsT, OutputT> {
public:
    using parent = convolution_random_test_base<InputT, WeightsT, OutputT>;

    primitive_id output_primitive_id() const override {
        return "scale_wa_reorder";
    }

    topology build_topology(cldnn::engine& engine) override {
        topology topo = parent::build_topology(engine);

        auto scale_lay = layout(this->output_type(), format::bfyx, tensor(batch(1), feature(this->output_features())));
        auto shift_lay = layout(this->output_type(), format::bfyx, tensor(batch(1), feature(this->output_features())));

        auto scale_mem = engine.allocate_memory(scale_lay);
        auto shift_mem = engine.allocate_memory(shift_lay);

        set_values(scale_mem, _scale);
        set_values(shift_mem, _shift);

        topo.add(cldnn::data("scale_scale", scale_mem));
        topo.add(cldnn::data("scale_shift", shift_mem));
        topo.add(cldnn::eltwise("scale", { input_info("conv"), input_info("scale_scale") }, eltwise_mode::prod));
        topo.add(cldnn::eltwise("shift", { input_info("scale"), input_info("scale_shift") }, eltwise_mode::sum));
        // Work-around since if scale is output it will not be fused
        topo.add(cldnn::reorder("scale_wa_reorder", input_info("shift"), format::bfyx, this->output_type()));
        return topo;
    }

    VVVVF<OutputT> calculate_reference() override {
        auto expected = parent::calculate_reference();

        for (size_t bi = 0; bi < this->batch_num(); ++bi)
            for (size_t fi = 0; fi < this->output_features(); ++fi) {
                expected[bi][fi] = reference_scale_post_op<OutputT>(expected[bi][fi], _scale[fi], _shift[fi]);
            }
        return expected;
    }

    void param_set_up(const convolution_random_test_all_params& params) override {
        parent::param_set_up(params);

        auto& rg = this->get_random_generator();
        _scale = rg.template generate_random_1d<OutputT>(this->output_features(), -1, 1);
        _shift = rg.template generate_random_1d<OutputT>(this->output_features(), -128, 128);
    }
protected:
    VF<OutputT> _scale;
    VF<OutputT> _shift;
};

class convolution_random_smoke_test : public testing::TestWithParam<convolution_random_test_all_params> {};

using convolution_random_test_s8s8f32 = convolution_random_test_base<int8_t, int8_t, float>;
using convolution_random_test_u8s8f32 = convolution_random_test_base<uint8_t, int8_t, float>;

using convolution_random_test_fsv4_input_s8s8f32 = convolution_random_test_fsv4_input<int8_t, int8_t, float>;
using convolution_random_test_fsv4_input_u8s8f32 = convolution_random_test_fsv4_input<uint8_t, int8_t, float>;

using convolution_scale_random_test_s8s8f32 = convolution_scale_random_test<int8_t, int8_t, float>;
using convolution_scale_random_test_u8s8f32 = convolution_scale_random_test<uint8_t, int8_t, float>;

struct params_generator : std::vector<convolution_random_test_all_params> {
    params_generator& smoke_test_params(format::type input_format,
                                        bool asymm_weights = false,
                                        bool asymm_data = false,
                                        bool padded_input = false,
                                        bool bigger_pad = false) {
        std::vector<size_t> batches = { 1, 2 };
        for (auto b : batches) {
            // first conv
            push_back(convolution_random_test_all_params{
                b, 3, 32, { 28, 28 }, { 7, 7 }, { 2, 2 }, { 3, 3 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            push_back(convolution_random_test_all_params{
                b, 3, 64, { 1024, 10 }, { 5, 5 }, { 2, 2 }, { 2, 2 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            push_back(convolution_random_test_all_params{
                b, 3, 15, { 10, 10 }, { 5, 5 }, { 1, 1 }, { 2, 2 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            push_back(convolution_random_test_all_params{
                b, 4, 18, { 10, 10 }, { 5, 5 }, { 1, 1 }, { 2, 2 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            // 3x3
            push_back(convolution_random_test_all_params{
                b, 32, 48, { 14, 14 }, { 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            push_back(convolution_random_test_all_params{
                b, 32, 48, { 14, 14 }, { 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            // 1x1
            push_back(convolution_random_test_all_params{
                b, 32, 48, { 28, 28 }, { 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            push_back(convolution_random_test_all_params{
                b, 32, 48, { 28, 28 }, { 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            // 5x5
            push_back(convolution_random_test_all_params{
                b, 32, 48, { 28, 28 }, { 5, 5 }, { 1, 1 }, { 2, 2 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            push_back(convolution_random_test_all_params{
                b, 32, 48, { 28, 28 }, { 5, 5 }, { 2, 2 }, { 2, 2 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            // depthwise
            push_back(convolution_random_test_all_params{
                b, 64, 64, { 19, 19 }, { 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, true, 64, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            push_back(convolution_random_test_all_params{
                b, 64, 64, { 19, 19 }, { 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, true, 64, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            // dilation
            push_back(convolution_random_test_all_params{
                b, 32, 24, { 19, 19 }, { 3, 3 }, { 1, 1 }, { 1, 1 }, { 2, 2 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            push_back(convolution_random_test_all_params{
                b, 32, 24, { 19, 19 }, { 3, 3 }, { 2, 2 }, { 1, 1 }, { 2, 2 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            // depthwise + dilation
            push_back(convolution_random_test_all_params{
                b, 64, 64, { 19, 19 }, { 3, 3 }, { 1, 1 }, { 1, 1 }, { 2, 2 }, true, 64, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            push_back(convolution_random_test_all_params{
                b, 64, 64, { 19, 19 }, { 3, 3 }, { 2, 2 }, { 1, 1 }, { 2, 2 }, true, 64, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
        }
        return *this;
    }

    params_generator& extra_test_params(format::type input_format,
                                        bool asymm_weights = false,
                                        bool asymm_data = false,
                                        bool padded_input = false,
                                        bool bigger_pad = false) {
        std::vector<size_t> batches = { 1, 2 };
        for (auto b : batches) {
            // 1x1
            push_back(convolution_random_test_all_params{
                b, 23, 41, { 19, 19 }, { 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            push_back(convolution_random_test_all_params{
                b, 23, 41, { 19, 19 }, { 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            // 3x3
            push_back(convolution_random_test_all_params{
                b, 16, 28, { 14, 14 }, { 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            push_back(convolution_random_test_all_params{
                b, 23, 41, { 19, 17 }, { 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            // 5x5
            push_back(convolution_random_test_all_params{
                b, 16, 28, { 14, 14 }, { 5, 5 }, { 1, 1 }, { 2, 2 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
            push_back(convolution_random_test_all_params{
                b, 23, 41, { 19, 17 }, { 5, 5 }, { 1, 1 }, { 2, 2 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, false });
        }
        return *this;
    }

    params_generator& bs_test_params(format::type input_format,
                                     bool asymm_weights = false,
                                     bool asymm_data = false,
                                     bool padded_input = false,
                                     bool bigger_pad = false,
                                     bool grouped_weights_shape = false) {
        std::vector<int> strides = { 1, 2 };
        for (auto s : strides) {
            // 1x1
            push_back(convolution_random_test_all_params{
            //      feature   input     filter    stride    offset  dilation  bias  groups
            //batch in  out   x  y      x  y      x  y      x  y      x  y
                16, 32, 32, { 4, 4 }, { 1, 1 }, { s, s }, { 0, 0 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, grouped_weights_shape });
            push_back(convolution_random_test_all_params{
                16, 32, 32, { 9, 9 }, { 1, 1 }, { s, s }, { 0, 0 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, grouped_weights_shape });
            // 3x3
            push_back(convolution_random_test_all_params{
                16, 32, 32, { 4, 4 }, { 3, 3 }, { s, s }, { 0, 0 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, grouped_weights_shape });
            push_back(convolution_random_test_all_params{
                16, 32, 32, { 9, 9 }, { 3, 3 }, { s, s }, { 0, 0 }, { 1, 1 }, true, 1, input_format, asymm_weights, asymm_data, padded_input, bigger_pad, grouped_weights_shape });
        }
        return *this;
    }

    params_generator& all_test_params(format::type input_format,
                                      bool asymm_weights = false,
                                      bool asymm_data = false,
                                      bool padded_input = false,
                                      bool bigger_pad = false) {
        return smoke_test_params(input_format, asymm_weights, asymm_data, padded_input, bigger_pad)
            .extra_test_params(input_format, asymm_weights, asymm_data, padded_input, bigger_pad);
    }

    params_generator& add(convolution_random_test_all_params params) {
        push_back(params);
        return *this;
    }
};

TEST_P(convolution_random_smoke_test, u8s8f32) {
    convolution_random_test_u8s8f32 test;
    ASSERT_NO_FATAL_FAILURE(test.run_random(GetParam()));
}

TEST_P(convolution_random_smoke_test, u8s8f32_scale) {
    convolution_scale_random_test_u8s8f32 test;
    ASSERT_NO_FATAL_FAILURE(test.run_random(GetParam()));
}

TEST_P(convolution_random_smoke_test, s8s8f32_fsv4_input) {
    convolution_random_test_fsv4_input_s8s8f32 test;
    ASSERT_NO_FATAL_FAILURE(test.run_random(GetParam()));
}

TEST_P(convolution_random_smoke_test, u8s8f32_fsv4_input) {
    convolution_random_test_fsv4_input_u8s8f32 test;
    ASSERT_NO_FATAL_FAILURE(test.run_random(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    basic,
    convolution_random_smoke_test,
    testing::ValuesIn(
        params_generator()
        .smoke_test_params(format::b_fs_yx_fsv4)
        .smoke_test_params(format::bfyx)
        .smoke_test_params(format::b_fs_yx_fsv32)
        .smoke_test_params(format::b_fs_yx_fsv32, true, true)
        .smoke_test_params(format::b_fs_yx_fsv32, false, true)
        .smoke_test_params(format::b_fs_yx_fsv32, true, false)
        .smoke_test_params(format::b_fs_yx_fsv32, false, false, true)
        .smoke_test_params(format::b_fs_yx_fsv16)
        .smoke_test_params(format::b_fs_yx_fsv16, true, true)
        .smoke_test_params(format::b_fs_yx_fsv16, false, true)
        .smoke_test_params(format::b_fs_yx_fsv16, true, false)
        .smoke_test_params(format::b_fs_yx_fsv16, false, false, true)
        .smoke_test_params(format::b_fs_yx_fsv16, false, false, true, true)
        .bs_test_params(format::bs_fs_yx_bsv16_fsv16)
        .bs_test_params(format::b_fs_yx_fsv16, false, true, false, false, true)
    ),
    to_string_convolution_all_params
);

class convolution_random_all_test : public testing::TestWithParam<convolution_random_test_all_params> {};

TEST_P(convolution_random_all_test, u8s8f32) {
    convolution_random_test_u8s8f32 test;
    ASSERT_NO_FATAL_FAILURE(test.run_random(GetParam()));
}

TEST_P(convolution_random_all_test, s8s8f32) {
    convolution_random_test_s8s8f32 test;
    ASSERT_NO_FATAL_FAILURE(test.run_random(GetParam()));
}

TEST_P(convolution_random_all_test, u8s8f32_scale) {
    convolution_scale_random_test_u8s8f32 test;
    ASSERT_NO_FATAL_FAILURE(test.run_random(GetParam()));
}

TEST_P(convolution_random_all_test, s8s8f32_scale) {
    convolution_scale_random_test_s8s8f32 test;
    ASSERT_NO_FATAL_FAILURE(test.run_random(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    DISABLED_basic,
    convolution_random_all_test,
    testing::ValuesIn(
        params_generator()
        .all_test_params(format::bfyx)
        .all_test_params(format::bfyx, true, true)
        .all_test_params(format::bfyx, false, true)
        .all_test_params(format::bfyx, true, false)
        .all_test_params(format::b_fs_yx_fsv4)
        .all_test_params(format::b_fs_yx_fsv32)
        .all_test_params(format::b_fs_yx_fsv32, true, true)
        .all_test_params(format::b_fs_yx_fsv32, false, true)
        .all_test_params(format::b_fs_yx_fsv32, true, false)
        .all_test_params(format::b_fs_yx_fsv16)
        .add(convolution_random_test_all_params{
            1, 89, 3, { 1, 1 }, { 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, true, 1, format::b_fs_yx_fsv4, false, false, false, false })
        .add(convolution_random_test_all_params{
            1, 16, 32, { 3, 3 }, { 17, 17 }, { 1, 1 }, { 8, 8 }, { 1, 1 }, true, 1, format::b_fs_yx_fsv16, false, false, true, false })
    ),
    to_string_convolution_all_params
);

class convolution_test : public tests::generic_test {
public:
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    static void TearDownTestCase() {
        all_generic_params.clear();
        all_layer_params.clear();
        all_test_params.clear();
    }

    static std::vector<std::shared_ptr<cldnn::primitive>> generate_specific_test_params() {
        // TODO: check convolution without bias

        const primitive_id& weights = "input1";
        const primitive_id& bias = "input2";

        std::vector<ov::Strides> stride_sizes = {
            ov::Strides{1, 1},
            ov::Strides{3, 2},
            ov::Strides{1, 4},
            ov::Strides{5, 5}
            };
        std::vector<ov::Strides> dilation_sizes = {
            ov::Strides{1, 1},
            ov::Strides{4, 5},
            ov::Strides{3, 1},
            ov::Strides{2, 7}
        };
        std::vector<ov::CoordinateDiff> pad_sizes = {
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{2, 2},
            ov::CoordinateDiff{-2, -5},
            ov::CoordinateDiff{-3, 3} };

        // No padding
        all_layer_params.emplace_back(new convolution("convolution_no_relu", input_info("input0"), weights, bias, 1, stride_sizes[0], dilation_sizes[0], pad_sizes[0], pad_sizes[0], false));
        all_layer_params.emplace_back(new convolution("convolution_no_relu", input_info("input0"), weights, bias, 1, stride_sizes[1], dilation_sizes[1], pad_sizes[1], pad_sizes[1], false));
        all_layer_params.emplace_back(new convolution("convolution_no_relu", input_info("input0"), weights, bias, 1, stride_sizes[2], dilation_sizes[2], pad_sizes[2], pad_sizes[2], false));
        all_layer_params.emplace_back(new convolution("convolution_no_relu", input_info("input0"), weights, bias, 1, stride_sizes[3], dilation_sizes[3], pad_sizes[3], pad_sizes[3], false));

        // Input padding
        all_layer_params.emplace_back(new convolution("convolution_no_relu", input_info("reorder0"), weights, bias, 1, stride_sizes[1], dilation_sizes[1], pad_sizes[1], pad_sizes[1], false));
        all_layer_params.emplace_back(new convolution("convolution_no_relu", input_info("reorder0"), weights, bias, 1, stride_sizes[3], dilation_sizes[3], pad_sizes[3], pad_sizes[3], false));

        // Output padding
        all_layer_params.emplace_back(new convolution("convolution_no_relu", input_info("input0"), weights, bias, 1, stride_sizes[1], dilation_sizes[1], pad_sizes[1], pad_sizes[1], false, ov::op::PadType::EXPLICIT, { { 0, 0, 2, 4 }, { 0, 0, 0, 19 } }));
        all_layer_params.emplace_back(new convolution("convolution_no_relu", input_info("input0"), weights, bias, 1, stride_sizes[2], dilation_sizes[2], pad_sizes[2], pad_sizes[2], false, ov::op::PadType::EXPLICIT, { { 0, 0, 1, 0 }, { 0, 0, 13, 9 } }));

        // Input + Output padding
        all_layer_params.emplace_back(new convolution("convolution_no_relu", input_info("reorder0"), weights, bias, 1, stride_sizes[0], dilation_sizes[0], pad_sizes[0], pad_sizes[0], false, ov::op::PadType::EXPLICIT, { { 0, 0, 1, 5 }, { 0, 0, 19, 4 } }));
        all_layer_params.emplace_back(new convolution("convolution_no_relu", input_info("reorder0"), weights, bias, 1, stride_sizes[3], dilation_sizes[3], pad_sizes[3], pad_sizes[3], false, ov::op::PadType::EXPLICIT, { { 0, 0, 1, 2 }, { 0, 0, 3, 4 } }));

        return all_layer_params;
    }

    static std::vector<std::tuple<std::shared_ptr<tests::test_params>, std::shared_ptr<cldnn::primitive>>> generate_all_test_params() {
        generate_specific_test_params();

        std::vector<cldnn::format> input_formats = { cldnn::format::bfyx, cldnn::format::yxfb };
        std::vector<cldnn::format> weights_formats = { cldnn::format::bfyx, cldnn::format::yxfb };

        std::vector<int32_t> output_features_sizes = { 1, 3, 16 };
        std::vector<cldnn::tensor> kernel_sizes = { tensor(1, 1, 1, 1), tensor(1, 1, 4, 7), tensor(1, 1, 5, 3) };

        std::vector<tensor> input_tensor_size = { tensor(1, 5, 59, 72), tensor(8, 3, 63, 56), tensor(16, 2, 50, 50), tensor(32, 1, 44, 62) };

        auto data_types = test_data_types();

        for (cldnn::data_types data_type : data_types) {
            for (cldnn::format input_format : input_formats) {
                for (cldnn::format weights_format : weights_formats) {
                    ExecutionConfig network_build_config = get_test_default_config(get_test_engine());
                    if (input_format == cldnn::format::bfyx) {
                        network_build_config.set_property(ov::intel_gpu::optimize_data(true));
                    }
                    for (cldnn::tensor input_size : input_tensor_size) {
                        for (cldnn::tensor kernel_size : kernel_sizes) {
                            for (auto output_features : output_features_sizes) {
                                std::shared_ptr<tests::test_params> params = std::make_shared<test_params>(data_type, input_format, input_size.batch[0], input_size.feature[0], tensor(1, 1, input_size.spatial[0], input_size.spatial[1]), network_build_config);
                                int input_features = params->input_layouts[0].feature();
                                params->input_layouts.push_back(cldnn::layout(params->data_type, weights_format, cldnn::tensor(output_features, input_features, kernel_size.spatial[0], kernel_size.spatial[1]))); // weights
                                params->input_layouts.push_back(cldnn::layout(params->data_type, params->fmt, cldnn::tensor(1, 1, output_features, 1))); // biases
                                all_generic_params.push_back(params);
                            }
                        }
                    }
                }
            }
        }

        // Create all the combinations for the test.
        for (const auto& layer_param : all_layer_params) {
            for (auto test_param : all_generic_params) {
                all_test_params.push_back(std::make_tuple(test_param, layer_param));
            }
        }

        return all_test_params;
    }

    bool is_format_supported(cldnn::format format) override {
        return ((format == cldnn::format::bfyx) || (format == cldnn::format::yxfb));
    }

    cldnn::tensor get_expected_output_tensor() override {
        auto convolution = std::static_pointer_cast<const cldnn::convolution>(layer_params);
        tensor input_size = generic_params->input_layouts[0].get_tensor();
        auto dilation = convolution->dilation;
        auto stride = convolution->stride;
        auto pad = convolution->padding_begin;
        tensor weights_size = generic_params->input_layouts[1].get_tensor();

        auto kernel_extent_y = dilation[dilation.size() - 2] * (weights_size.spatial[1] - 1) + 1;
        auto kernel_extent_x = dilation[dilation.size() - 1] * (weights_size.spatial[0] - 1) + 1;

        // Calculate output size
        auto output_size_y = 1 + (input_size.spatial[1] - kernel_extent_y + 2 * pad[0]) / stride[0];
        auto output_size_x = 1 + (input_size.spatial[0] - kernel_extent_x + 2 * pad[1]) / stride[1];
        auto output_features = weights_size.batch[0];

        return cldnn::tensor(input_size.batch[0],
                             static_cast<cldnn::tensor::value_type>(output_features),
                             static_cast<cldnn::tensor::value_type>(output_size_x),
                             static_cast<cldnn::tensor::value_type>(output_size_y));
    }

    void prepare_input_for_test(std::vector<cldnn::memory::ptr>& inputs) override {
        if (generic_params->data_type == data_types::f32) {
            prepare_input_for_test_typed<float>(inputs);
        } else {
            prepare_input_for_test_typed<ov::float16>(inputs);
        }
    }

    template<typename Type>
    void prepare_input_for_test_typed(std::vector<cldnn::memory::ptr>& inputs) {
        int k = (generic_params->data_type == data_types::f32) ? 8 : 4;

        // Update inputs.
        auto input = inputs[0];
        auto input_size = inputs[0]->get_layout().get_tensor();
        VVVVF<Type> input_rnd = rg.generate_random_4d<Type>(input_size.batch[0], input_size.feature[0], input_size.spatial[1], input_size.spatial[0], -2, 2, k);
        VF<Type> input_rnd_vec = flatten_4d<Type>(input->get_layout().format, input_rnd);
        set_values(input, input_rnd_vec);

        // Update weights.
        auto weight_input = inputs[1];
        auto weight_size = inputs[1]->get_layout().get_tensor();
        VVVVF<Type> weight_rnd = rg.generate_random_4d<Type>(weight_size.batch[0], weight_size.feature[0], weight_size.spatial[1], weight_size.spatial[0], -2, 2, k);
        VF<Type> weight_rnd_vec = flatten_4d<Type>(weight_input->get_layout().format, weight_rnd);
        set_values(weight_input, weight_rnd_vec);

        // Update biases.
        auto bias_input = inputs[2];
        auto bias_size = inputs[2]->get_layout().get_tensor();
        VF<Type> bias_rnd = rg.generate_random_1d<Type>(bias_size.spatial[0], -2, 2, k);
        set_values(bias_input, bias_rnd);
    }

    template<typename Type>
    memory::ptr generate_reference_typed(const std::vector<cldnn::memory::ptr>& inputs) {
        // Output reference is always bfyx.

        auto convolution = std::static_pointer_cast<const cldnn::convolution>(layer_params);

        data_types dt = inputs[0]->get_layout().data_type;

        tensor input_size = inputs[0]->get_layout().get_tensor();
        ov::Strides dilation = convolution->dilation;
        ov::Strides stride = convolution->stride;
        ov::CoordinateDiff pad = convolution->padding_begin;
        tensor weights_size = inputs[1]->get_layout().get_tensor();
        padding output_padding = convolution->output_paddings[0];

        tensor output_size = get_expected_output_tensor();

        // Calculate output size
        int output_size_y = output_size.spatial[1];
        int output_size_x = output_size.spatial[0];
        int output_features = weights_size.batch[0];
        int input_features = weights_size.feature[0];

        auto output = engine.allocate_memory(cldnn::layout(dt, cldnn::format::bfyx, output_size, output_padding));

        cldnn::mem_lock<Type> input_mem(inputs[0], get_test_stream());
        cldnn::mem_lock<Type> weights_mem(inputs[1], get_test_stream());
        cldnn::mem_lock<Type> bias_mem(inputs[2], get_test_stream());
        cldnn::mem_lock<Type> output_mem(output, get_test_stream());

        tensor output_buffer_size = output->get_layout().get_buffer_size();

        // Initialized output with zeros.
        std::fill(output_mem.begin(), output_mem.end(), static_cast<Type>(0));

        // Add the bias
        for (int b = 0; b < input_size.batch[0]; b++) {
            for (int out_f = 0; out_f < output_features; out_f++) {
                for (int y = 0; y < output_size_y; y++) {
                    for (int x = 0; x < output_size_x; x++) {
                        int output_index = (b * output_buffer_size.feature[0] + out_f) * output_buffer_size.spatial[1] * output_buffer_size.spatial[0];
                        tensor lower_output_padding = convolution->output_paddings[0].lower_size();
                        output_index += (lower_output_padding.spatial[1] + y) * output_buffer_size.spatial[0] + lower_output_padding.spatial[0] + x;

                        output_mem[output_index] += bias_mem[out_f];
                    }
                }
            }
        }

        const auto input0_desc = get_linear_memory_desc(inputs[0]->get_layout());
        const auto input1_desc = get_linear_memory_desc(inputs[1]->get_layout());

        // Convolve with weights
        for (int b = 0; b < input_size.batch[0]; b++) {
            int input_bi = b;
            for (int out_f = 0; out_f < output_features; out_f++) {
                for (int in_f = 0; in_f < input_features; in_f++) {
                    int input_fi = in_f;
                    for (int y = 0; y < output_size_y; y++) {
                        for (int x = 0; x < output_size_x; x++) {
                            int output_bi = b;
                            int output_fi = out_f;
                            int output_yi = y;
                            int output_xi = x;
                            auto output_index = (output_bi * output_buffer_size.feature[0] + output_fi) * output_buffer_size.spatial[1] * output_buffer_size.spatial[0];
                            tensor lower_output_padding = convolution->output_paddings[0].lower_size();
                            output_index += (lower_output_padding.spatial[1] + output_yi) * output_buffer_size.spatial[0] + lower_output_padding.spatial[0] + output_xi;

                            for (int kernel_y = 0; kernel_y < weights_size.spatial[1]; kernel_y++) {
                                int input_yi = static_cast<int>(y * stride[0] - pad[0] + kernel_y * dilation[0]);
                                if ((input_yi < 0) || (input_yi >= static_cast<int>(input_size.spatial[1]))) {
                                    continue;
                                }

                                for (int kernel_x = 0; kernel_x < weights_size.spatial[0]; kernel_x++) {
                                    int input_xi = static_cast<int>(x * stride[1] - pad[1] + kernel_x * dilation[1]);
                                    if ((input_xi < 0) || (input_xi >= static_cast<int>(input_size.spatial[0]))) {
                                        continue;
                                    }

                                    size_t input_index = get_linear_index(inputs[0]->get_layout(), input_bi, input_fi, input_yi, input_xi, input0_desc);

                                    int weight_bi = out_f;
                                    int weight_fi = in_f;
                                    int weight_yi = kernel_y;
                                    int weight_xi = kernel_x;
                                    size_t weight_index = get_linear_index(inputs[1]->get_layout(), weight_bi, weight_fi, weight_yi, weight_xi, input1_desc);
                                    output_mem[output_index] += input_mem[input_index] * weights_mem[weight_index];
                                }
                            }
                        }
                    }
                }
            }
        }

        return output;
    }

    memory::ptr generate_reference(const std::vector<cldnn::memory::ptr>& inputs) override {
        if (generic_params->data_type == data_types::f32) {
            return generate_reference_typed<float>(inputs);
        } else {
            return generate_reference_typed<ov::float16>(inputs);
        }
    }

private:

    static std::vector<std::shared_ptr<tests::test_params>> all_generic_params;
    static std::vector<std::shared_ptr<cldnn::primitive>> all_layer_params;
    static std::vector<std::tuple<std::shared_ptr<tests::test_params>, std::shared_ptr<cldnn::primitive>>> all_test_params;
};

std::vector<std::shared_ptr<tests::test_params>> convolution_test::all_generic_params = {};
std::vector<std::shared_ptr<cldnn::primitive>> convolution_test::all_layer_params = {};
std::vector<std::tuple<std::shared_ptr<tests::test_params>, std::shared_ptr<cldnn::primitive>>> convolution_test::all_test_params = {};

TEST_P(convolution_test, CONVOLUTION) {
    run_single_test();
}

INSTANTIATE_TEST_SUITE_P(DISABLED_CONVOLUTION,
                         convolution_test,
                         ::testing::ValuesIn(convolution_test::generate_all_test_params()),
                         tests::generic_test::custom_param_name_functor());


#ifdef ENABLE_ONEDNN_FOR_GPU
using TestParamType_convolution_gpu_onednn = ::testing::tuple<  int,    // 0 - Input X size
        int,            // 1  - Input Y size
        int,            // 2  - Input Z size
        int,            // 3  - Input features
        int,            // 4  - Output features
        int,            // 5  - Kernel sizeX
        int,            // 6  - Kernel sizeY
        int,            // 7  - Kernel sizeZ
        int,            // 8  - Groups number
        int,            // 9  - Stride
        int,            // 10 - Batch
        format,         // 11 - Input data format
        std::string,    // 12 - Implementation name
        bool>;          // 13 - With bias

struct convolution_gpu_onednn : public ::testing::TestWithParam<TestParamType_convolution_gpu_onednn> {
    static std::string PrintToStringParamName(
        testing::TestParamInfo<TestParamType_convolution_gpu_onednn> param_info) {
        // construct a readable name
        std::string res = "in" + std::to_string(testing::get<0>(param_info.param)) + "x" +
                          std::to_string(testing::get<1>(param_info.param)) + "y" +
                          std::to_string(testing::get<2>(param_info.param)) + "z" +
                          std::to_string(testing::get<3>(param_info.param)) + "f" + "_output" +
                          std::to_string(testing::get<4>(param_info.param)) + "f" + "_filter" +
                          std::to_string(testing::get<5>(param_info.param)) + "x" +
                          std::to_string(testing::get<6>(param_info.param)) + "y" +
                          std::to_string(testing::get<7>(param_info.param)) + "z" + "_groups" +
                          std::to_string(testing::get<8>(param_info.param)) + "_stride" +
                          std::to_string(testing::get<9>(param_info.param)) + "_batch" +
                          std::to_string(testing::get<10>(param_info.param)) + "_format" +
                          std::to_string(testing::get<11>(param_info.param)) + "_with_bias_" +
                          std::to_string(testing::get<13>(param_info.param));

        if (testing::get<12>(param_info.param) != "") {
            res += "_kernel_" + testing::get<12>(param_info.param);
        }

        return res;
    }
};

INSTANTIATE_TEST_SUITE_P(conv_onednn_cases,
                        convolution_gpu_onednn,
                        ::testing::Values(
                            // Input X size, Input Y size, Input Z size, Input features, Output features,
                            // Kernel size X, Kernel size Y, Kernel size Z, Groups number, Stride, Batch,
                            // Input data format, Implementation name, WithBias
                            TestParamType_convolution_gpu_onednn(8, 8, 1, 32, 32, 3, 3, 1, 1, 1, 32, format::bfyx, "", true),
                            TestParamType_convolution_gpu_onednn(8, 8, 1, 32, 32, 3, 3, 1, 1, 1, 32, format::bfyx, "", false)
                            // TestParamType_convolution_gpu_onednn(8, 8, 1, 32, 32, 3, 3, 1, 1, 1, 32, format::bfyx, "", true),
                            // TestParamType_convolution_gpu_onednn(8, 8, 1, 32, 32, 3, 3, 1, 1, 1, 32, format::bfyx, "", false)
                        ),
                        convolution_gpu_onednn::PrintToStringParamName);


TEST_P(convolution_gpu_onednn, conv_onednn_cases) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    if (!engine.get_device_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }

    const int input_x = testing::get<0>(GetParam()),
              input_y = testing::get<1>(GetParam()),
              input_f = testing::get<3>(GetParam()),
              output_f = testing::get<4>(GetParam()),
              filter_x = testing::get<5>(GetParam()),
              filter_y = testing::get<6>(GetParam()),
              groups = testing::get<8>(GetParam()),
              batch_num = testing::get<10>(GetParam());
    const uint64_t stride = testing::get<9>(GetParam());
    auto input_data_format = testing::get<11>(GetParam());
    auto impl_name = testing::get<12>(GetParam());
    auto with_bias = testing::get<13>(GetParam());

    auto input_size = tensor(batch_num, input_f, input_x, input_y);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_y, input_x, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_y, filter_x, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, filter_y, filter_x, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::bfyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto expected_result = VVVVF<ov::float16>(batch_num, VVVF<ov::float16>(output_f));
    topology topology;

    // Calculate reference values
    if (with_bias) {
        auto biases_size = tensor(1, output_f, 1, 1);
        auto biases_data = rg.generate_random_1d<ov::float16>(output_f, -1, 1);
        auto biases_mem = engine.allocate_memory({ data_types::f16, format::bfyx, biases_size });
        set_values(biases_mem, biases_data);

        for (auto bi = 0; bi < batch_num; ++bi) {
            for (auto ofi = 0; ofi < output_f; ++ofi) {
                expected_result[bi][ofi] = reference_convolve(input_data[bi],                    // input
                                                              weights_data[ofi],                 // weights
                                                              stride, stride,                    // strides
                                                              biases_data[ofi],                  // bias
                                                              1, 1,                              // dilation
                                                              0, 0,  // input padding
                                                              0, 0);   // output_padding
            }
        }

        topology.add(input_layout("input", input_mem->get_layout()),
                     data("weights_fsv", weights_mem),
                     data("bias", biases_mem),
                     reorder("input_fsv", input_info("input"), { data_types::f16, input_data_format, input_size }));

        auto conv_fsv = convolution("conv_fsv",
                                    input_info("input_fsv"),
                                    "weights_fsv",
                                    "bias",
                                    groups,
                                    { stride, stride },
                                    { 1, 1 },
                                    { 0, 0 },
                                    { 0, 0 },
                                    false);
        conv_fsv.output_paddings = {padding({ 0, 0, 0, 0 }, 0.f)};

        topology.add(conv_fsv);
    } else {
        for (auto bi = 0; bi < batch_num; ++bi) {
            for (auto ofi = 0; ofi < output_f; ++ofi) {
                expected_result[bi][ofi] = reference_convolve(input_data[bi],                    // input
                                                              weights_data[ofi],                 // weights
                                                              stride, stride,                    // strides
                                                              0,                                 // bias
                                                              1, 1,                              // dilation
                                                              0, 0,  // input padding
                                                              0, 0);   // output_padding
            }
        }

        topology.add(input_layout("input", input_mem->get_layout()),
                     data("weights_fsv", weights_mem),
                     reorder("input_fsv", input_info("input"), { data_types::f16, input_data_format, input_size }));

        auto conv_fsv = convolution("conv_fsv",
                                    input_info("input_fsv"),
                                    "weights_fsv",
                                    no_bias,
                                    groups,
                                    { stride, stride },
                                    {1, 1},
                                    { 0, 0 },
                                    { 0, 0 },
                                    false);
        conv_fsv.output_paddings = {padding({ 0, 0, 0, 0 }, 0.f)};
        topology.add(conv_fsv);
    }
    topology.add(reorder("reorder_bfyx", input_info("conv_fsv"), format::bfyx, data_types::f32));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"conv_fsv","reorder_bfyx"}));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(2));
    ASSERT_TRUE(outputs.find("conv_fsv") != outputs.end());

    for (auto& p : network.get_primitives_info())
        std::cerr << p.original_id << " " << p.kernel_id << std::endl;

    auto out_ptr = get_output_values_to_float<ov::float16>(network, outputs.find("conv_fsv")->second);
    auto out_lay = network.get_node_output_layout("conv_fsv");
    ASSERT_EQ(out_lay.batch(), expected_result.size());
    ASSERT_EQ(out_lay.feature(), expected_result[0].size());
    ASSERT_EQ(out_lay.spatial(1), expected_result[0][0].size());
    ASSERT_EQ(out_lay.spatial(0), expected_result[0][0][0].size());

    for (int bi = 0; bi < out_lay.batch(); ++bi)
        for (int ofi = 0; ofi < out_lay.feature(); ++ofi)
            for (int yi = 0; yi < out_lay.spatial(1); ++yi)
                for (int xi = 0; xi < out_lay.spatial(0); ++xi) {
                    tensor coords = tensor(batch(bi), feature(ofi), spatial(xi, yi, 0, 0));
                    auto offset = out_lay.get_linear_offset(coords);
                    auto val = out_ptr[offset];
                    auto val_ref = expected_result[bi][ofi][yi][xi];
                    auto equal = are_equal(val_ref, val, 1);
                    if (!equal) {
                        std::cout << "Value at batch: " << bi << ", output_f: " << ofi
                                    << ", y: " << yi << ", x: " << xi << " = " << static_cast<float>(val) << std::endl;
                        std::cout << "Reference value at batch: " << bi << ", output_f: " << ofi << ", y: " << yi
                                  << ", x: " << xi << " = " << static_cast<float>(val_ref) << std::endl;
                    }
                    ASSERT_TRUE(equal);
                }
}

TEST(convolution_gpu_onednn, padding_for_cldnn_kernel_after_onednn) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    tests::random_generator rg(GET_SUITE_NAME);
    int input_b = 1, input_f = 16, input_y = 3, input_x = 3;
    int output_b = 1, output_f = 16, output_y = 6, output_x = 6;

    auto input_size = tensor(input_b, input_f, input_x, input_y);
    auto input_data = rg.generate_random_4d<ov::float16>(input_b, input_f, input_y, input_x, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(16, 16, 1, 1, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, 1, 1, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::bfyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    auto input = input_layout("input", input_mem->get_layout());
    auto weights = data("weights", weights_mem);
    auto input_reorder = reorder("input_fsv", input_info("input"), { data_types::f16, format::b_fs_yx_fsv16, input_size });
    auto conv1 = convolution("conv1", input_info("input_fsv"), "weights", no_bias, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false);
    auto conv2 = convolution("conv2", input_info("conv1"), "weights", no_bias, 1, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 2, 2 }, false);
    auto output_reorder = reorder("reorder", input_info("conv2"), { data_types::f32, format::bfyx, { output_b, output_f, output_x, output_x } });

    topology topology_test(input, weights, input_reorder, conv1, conv2, output_reorder);
    topology topology_ref(input, weights, input_reorder, conv1, conv2, output_reorder);

    ExecutionConfig config_test = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv1_impl_test = { format::byxf, "", impl_types::onednn };
    ov::intel_gpu::ImplementationDesc conv2_impl_test = { format::b_fs_yx_fsv16, "convolution_gpu_bfyx_f16", impl_types::ocl };
    config_test.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv1", conv1_impl_test }, { "conv2", conv2_impl_test } }));
    config_test.set_property(ov::intel_gpu::optimize_data(true));

    ExecutionConfig config_ref = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv1_impl_ref = { format::bfyx, "", impl_types::ocl };
    ov::intel_gpu::ImplementationDesc conv2_impl_ref = { format::bfyx, "", impl_types::ocl };
    config_ref.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv1", conv1_impl_ref }, { "conv2", conv2_impl_ref } }));
    config_ref.set_property(ov::intel_gpu::optimize_data(true));

    network network_test(engine, topology_test, config_test);
    network network_ref(engine, topology_ref, config_ref);

    network_test.set_input_data("input", input_mem);
    network_ref.set_input_data("input", input_mem);

    auto outputs_test = network_test.execute();
    auto outputs_ref = network_ref.execute();

    ASSERT_EQ(outputs_test.size(), size_t(1));
    ASSERT_EQ(outputs_test.begin()->first, "reorder");
    ASSERT_EQ(outputs_ref.size(), size_t(1));
    ASSERT_EQ(outputs_ref.begin()->first, "reorder");

    auto output_memory_test = outputs_test.at("reorder").get_memory();
    auto output_layout_test = output_memory_test->get_layout();
    cldnn::mem_lock<float> output_ptr_test(output_memory_test, get_test_stream());

    auto output_memory_ref = outputs_ref.at("reorder").get_memory();
    auto output_layout_ref = output_memory_ref->get_layout();
    cldnn::mem_lock<float> output_ptr_ref(output_memory_ref, get_test_stream());

    ASSERT_EQ(output_layout_test.spatial(0), output_x);
    ASSERT_EQ(output_layout_test.spatial(1), output_y);
    ASSERT_EQ(output_layout_test.feature(), output_f);
    ASSERT_EQ(output_layout_test.batch(), output_b);

    ASSERT_EQ(output_layout_ref.spatial(0), output_x);
    ASSERT_EQ(output_layout_ref.spatial(1), output_y);
    ASSERT_EQ(output_layout_ref.feature(), output_f);
    ASSERT_EQ(output_layout_ref.batch(), output_b);

    for (size_t i = 0; i < output_memory_ref->count(); i++) {
        ASSERT_EQ(output_ptr_ref.data()[i], output_ptr_test.data()[i]);
    }
}

TEST(convolution_gpu_onednn, spatial_1d) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    tests::random_generator rg(GET_SUITE_NAME);
    ov::PartialShape input_pshape = {1, 16, 6};
    ov::PartialShape weights_pshape = {16, 16, 3};
    layout in_layout{ input_pshape, data_types::f16, format::bfyx };
    layout weights_layout{ weights_pshape, data_types::f16, format::bfyx };
    auto input_data = rg.generate_random_1d<ov::float16>(in_layout.count(), -1, 1);
    auto input_mem = engine.allocate_memory(in_layout);
    set_values(input_mem, input_data);

    auto weights_data = rg.generate_random_1d<ov::float16>(weights_layout.count(), -1, 1);
    auto weights_mem = engine.allocate_memory(weights_layout);
    set_values(weights_mem, weights_data);

    auto input = input_layout("input", input_mem->get_layout());
    auto weights = data("weights", weights_mem);
    auto conv = convolution("conv",
                            input_info("input"),
                            "weights",
                            no_bias,
                            1,
                            ov::Strides{1},
                            ov::Strides{1},
                            ov::CoordinateDiff{0},
                            ov::CoordinateDiff{0},
                            false);
    auto output_reorder = reorder("reorder", input_info("conv"), format::bfyx, data_types::f32 );

    topology t(input, weights, conv, output_reorder);

    ExecutionConfig config_test = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl_test = { format::b_fs_yx_fsv16, "", impl_types::onednn };
    config_test.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", conv_impl_test } }));
    config_test.set_property(ov::intel_gpu::optimize_data(true));
    config_test.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    ExecutionConfig config_ref = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl_ref = { format::bfyx, "", impl_types::ocl };
    config_ref.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{ "conv", conv_impl_ref } }));
    config_ref.set_property(ov::intel_gpu::optimize_data(true));
    config_ref.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network_test(engine, t, config_test);
    network network_ref(engine, t, config_ref);

    network_test.set_input_data("input", input_mem);
    network_ref.set_input_data("input", input_mem);

    auto outputs_test = network_test.execute();
    auto outputs_ref = network_ref.execute();

    ASSERT_EQ(outputs_test.size(), size_t(1));
    ASSERT_EQ(outputs_test.begin()->first, "reorder");
    ASSERT_EQ(outputs_ref.size(), size_t(1));
    ASSERT_EQ(outputs_ref.begin()->first, "reorder");

    auto output_memory_test = outputs_test.at("reorder").get_memory();
    auto output_layout_test = output_memory_test->get_layout();
    cldnn::mem_lock<float> output_ptr_test(output_memory_test, get_test_stream());

    auto output_memory_ref = outputs_ref.at("reorder").get_memory();
    auto output_layout_ref = output_memory_ref->get_layout();
    cldnn::mem_lock<float> output_ptr_ref(output_memory_ref, get_test_stream());

    ov::PartialShape expected_shape = {1, 16, 4};
    ASSERT_EQ(output_layout_test.get_partial_shape(), expected_shape);
    ASSERT_EQ(output_layout_ref.get_partial_shape(), expected_shape);

    for (size_t i = 0; i < output_memory_ref->count(); i++) {
        ASSERT_EQ(output_ptr_ref.data()[i], output_ptr_test.data()[i]);
    }
}

TEST(convolution_gpu_onednn, quantized_onednn_convolution_u8s8f32_asymmetric_activations_per_tensor) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    auto input = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 1, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, { 2, 1, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });
    auto a_zp = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 1, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });
    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5 });
    set_values<uint8_t>(a_zp, { 2 });
    set_values(biases, { 1.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { 16.0f, 42.0f, -13.0f },
            { 2.0f, 8.0f, 2.0f }
        },
        {
            { -12.0f, 3.0f, 15.0f },
            { -7.0f, 1.0f, -16.0f }
        } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("a_zp", a_zp),
        convolution("conv", input_info("input"), "weights", "biases", "", "a_zp", "", 1,
                    { 2, 2 }, { 1, 1 }, { 0, 0 }, { 1, 2 }, false, data_types::f32),
        reorder("out", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl = { format::byxf, "", impl_types::onednn };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", conv_impl }}));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    auto output_layout = output_memory->get_layout();
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 2);
    ASSERT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_NEAR(output_vec[f][y][x], ((float)output_ptr[f * y_size * x_size + y * x_size + x]), 1e-5f) <<
                " x="<< x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_gpu_onednn, quantized_onednn_convolution_u8s8f32_asymmetric_activations_per_channel) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    auto input = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 2, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, { 3, 2, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 1, 1 } });
    auto a_zp = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 3, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1,

                                 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });

    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5,

                                   1, 2, -1,
                                   -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                   -1, 3,  2,
                                   0, 2,  5,

                                   1, 2, -1,
                                   -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                   -1, 3,  2,
                                   0, 2,  5 });
    set_values<uint8_t>(a_zp, { 2, 5, 5 });
    set_values(biases, { 1.0f, -8.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { -36.0f, 5.0f, -14.0f },
            { -24.0f, -10.0f, -30.0f }
        },
        {
            { -45.0f, -4.0f, -23.0f },
            { -33.0f, -19.0f, -39.0f }
        },
        {
            { -45.0f, -4.0f, -23.0f },
            { -33.0f, -19.0f, -39.0f }
        } };

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("a_zp", a_zp),
        convolution("conv", input_info("input"), "weights", "biases", "", "a_zp", "", 1,
                    { 2, 2 }, { 1, 1 }, { 0, 0 }, { 1, 2 }, false, data_types::f32),
        reorder("out", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl = { format::byxf, "", impl_types::onednn };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", conv_impl }}));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    auto output_layout = output_memory->get_layout();
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 3);
    ASSERT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_NEAR(output_vec[f][y][x], ((float)output_ptr[f * y_size * x_size + y * x_size + x]), 1e-5f) <<
                " x="<< x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_gpu_onednn, has_proper_synchronization) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    tests::random_generator rg(GET_SUITE_NAME);
    layout in_layout{{1, 16, 2, 4}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory(in_layout);
    auto weights_mem = engine.allocate_memory({{16, 16, 1, 1}, data_types::f32, format::bfyx});

    auto in_data = rg.generate_random_4d<float>(1, 16, 2, 4, -1, 1);
    auto weights_data = rg.generate_random_4d<float>(16, 16, 1, 1, -1, 1);

    set_values(input_mem, flatten_4d(format::bfyx, in_data));
    set_values(weights_mem, flatten_4d(format::bfyx, weights_data));

    auto create_topology =[&]() {
        topology topology;
        topology.add(input_layout("input", in_layout));
        topology.add(data("weights", weights_mem));
        topology.add(convolution("conv", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        topology.add(activation("activation", input_info("conv"), activation_func::relu));
        topology.add(reorder("reorder", input_info("conv"), in_layout));
        return topology;
    };

    auto topology_ref = create_topology();
    auto topology_test = create_topology();

    auto impl_desc_cpu = ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::cpu};
    auto impl_desc_onednn = ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::onednn};
    auto impl_forcing_map = ov::intel_gpu::ImplForcingMap{{"conv", impl_desc_onednn}, {"activation", impl_desc_cpu}};

    auto config_ref = get_test_default_config(engine);
    config_ref.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    config_ref.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto config_test = config_ref;
    config_test.set_property(ov::intel_gpu::force_implementations(impl_forcing_map));

    network net_test(engine, topology_test, config_test);
    net_test.set_input_data("input", input_mem);
    auto outputs_test = net_test.execute();
    auto res_test = outputs_test.at("activation").get_memory();

    network net_ref(engine, topology_ref, config_ref);
    net_ref.set_input_data("input", input_mem);
    auto outputs_ref = net_ref.execute();
    auto res_ref = outputs_ref.at("activation").get_memory();

    ASSERT_EQ(res_test->get_layout().get_linear_size(), res_ref->get_layout().get_linear_size());

    cldnn::mem_lock<float> test_mem(res_test, get_test_stream());
    cldnn::mem_lock<float> ref_mem(res_ref, get_test_stream());

    for (size_t i = 0; i < res_ref->get_layout().get_linear_size(); ++i) {
        ASSERT_EQ(test_mem[i], ref_mem[i]);
    }
}

#endif   // ENABLE_ONEDNN_FOR_GPU

template <typename T>
void test_convolution_f32_gpu_convolution_gpu_bfyx_f16_depthwise_x_block_size_1(bool is_caching_test) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int batch_num = 1;
    const int input_xy = 1;
    const int groups = 128;
    const int input_f = groups;
    const int output_f = groups;
    const int filter_y = 3;
    const int filter_x = 3;
    const uint64_t stride = 1;
    const int output_padding = 1;
    const int pad_y = filter_y / 2;
    const int pad_x = filter_x / 2;
    const int f_group_size = 16;
    const int f_group_num_in_batch = (output_f % f_group_size) ? (output_f / f_group_size + 1) : (output_f / f_group_size);

    const int output_y = 1 + (input_xy + 2 * pad_y - filter_y) / stride + 2 * output_padding;
    const int output_x = 1 + (input_xy + 2 * pad_x - filter_x) / stride + 2 * output_padding;

    if (engine.get_device_info().supports_immad) {
        // Currently, oneDNN does NOT support padded input or output
        return;
    }

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy);
    auto input_data = rg.generate_random_4d<T>(batch_num, input_f, input_xy, input_xy, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(group(output_f), batch(1), feature(1), spatial(filter_x, filter_y));
    auto weights_data = rg.generate_random_4d<T>(output_f, 1, filter_y, filter_x, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::goiyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<T>(batch_num, VVVF<T>(output_f));

    topology topology(
        input_layout("input", input_mem->get_layout()),
        data("weights_fsv", weights_mem));

    // Reorder input to b_fs_yx_fsv16
    topology.add(reorder("input_fsv", input_info("input"), { data_types::f16, format::b_fs_yx_fsv16, input_size }));

    // Calculate reference values without bias
    for (auto bi = 0; bi < batch_num; ++bi)
    {
        for (auto ofi = 0; ofi < output_f; ++ofi)
        {
            reference_result[bi][ofi] = reference_convolve(
                input_data[bi], weights_data[ofi],  // input, weights
                stride, stride,                     // strides
                0,                                  // bias
                1, 1,                               // dilation
                pad_y, pad_x,                       // input padding
                output_padding, output_padding,     // output_padding
                ofi, ofi + 1,                       // f_begin, f_end
                true);                              // depthwise
        }
    }

    auto conv_fsv = convolution("conv_fsv", input_info("input_fsv"), "weights_fsv", no_bias, groups,
                                { stride, stride }, {1, 1}, { pad_y, pad_x }, { pad_y, pad_x }, true);
    conv_fsv.output_paddings = {padding({ 0, 0, output_padding, output_padding }, 0.f)};

    topology.add(conv_fsv);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "convolution_gpu_bfyx_f16_depthwise" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_fsv", conv_impl } }));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input_mem);

    network->execute();

    auto out_mem = network->get_output("conv_fsv").get_memory();
    cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(out_mem->get_layout().format, format::b_fs_yx_fsv16);

    for (int bi = 0; bi < batch_num; ++bi)
        for (int fi = 0; fi < output_f; ++fi)
            for (int yi = 0; yi < output_y; ++yi)
                for (int xi = 0; xi < output_x; ++xi)
                {
                    auto val_ref = reference_result[bi][fi][yi][xi];

                    auto val = out_ptr[bi * f_group_num_in_batch * f_group_size * output_y * output_x  +
                        (fi / f_group_size) * output_y * output_x * f_group_size +
                        yi * output_x * f_group_size +
                        xi * f_group_size +
                        fi % f_group_size];
                    auto equal = are_equal(val_ref, val, 1e-2f);
                    ASSERT_TRUE(equal);
                    if (!equal)
                    {
                        std::cout << "At b = " << bi << ", fi = " << fi << ", yi = " << yi << ", xi = " << xi << std::endl;
                    }
                }
}

TEST(convolution_f32_gpu, convolution_gpu_bfyx_f16_depthwise_x_block_size_1) {
    test_convolution_f32_gpu_convolution_gpu_bfyx_f16_depthwise_x_block_size_1<ov::float16>(false);
}

TEST(export_import_convolution_f32_gpu, convolution_gpu_bfyx_f16_depthwise_x_block_size_1) {
    test_convolution_f32_gpu_convolution_gpu_bfyx_f16_depthwise_x_block_size_1<ov::float16>(true);
}


TEST(convolution_f32_fw_gpu, basic_convolution_no_bias_swap_xy) {
    //  Filter : 2x2x1x3
    //  Stride : 1x1
    //  Input  : 1x2x1x5
    //  Output : 1x2x1x3
    //
    //  Input:
    //  1 1
    //  2 2
    //  3 3
    //  4 4
    //  5 5
    //
    //  Filter:
    //  1 1 1 1
    //  2 2 2 2
    //  1 1 1 1
    //
    //  Output:
    //  16 16
    //  24 24
    //  32 32
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 5 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 3 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f});
    VVVF<float> output_vec = {{{ 16.0f }, { 24.0f }, { 32.0f }}, {{ 16.0f }, { 24.0f }, { 32.0f }}};

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        convolution("conv", input_info("input"), "weights", no_bias, 1, { 1, 1 }, {1, 1}, {0, 0}, {0, 0}, false));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    const auto& const_net = network;
    const auto conv_inst = const_net.get_primitive("conv");
    ASSERT_TRUE(conv_inst != nullptr);

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();

    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 3);
    ASSERT_EQ(x_size, 1);
    ASSERT_EQ(f_size, 2);
    ASSERT_EQ(b_size, 1);

    for (int f = 0; f < f_size; ++f) {
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_EQ(output_vec[f][y][x], output_ptr[f * y_size + y * x_size + x]);
            }
        }
    }

    auto inst = network.get_primitive("conv");
    const auto& node = inst->get_node();
    auto selected_impl = node.type()->choose_impl(node);
    bool found_define = false;
    for (auto& s : selected_impl->get_kernels_source()) {
        if (s != nullptr && !s->get_str().empty()
            && s->get_str().find("#define INPUT0_SIZE_X 5") != std::string::npos)
            found_define = true;
    }
    EXPECT_TRUE(found_define);
}

struct conv_dyn_test_params {
    ov::Shape in_shape;
    ov::Shape wei_shape;
    ov::Strides stride;
    ov::Strides dilation;
    ov::CoordinateDiff pad_begin;
    ov::CoordinateDiff pad_end;
};

class conv_dyn_test : public testing::TestWithParam<conv_dyn_test_params> {};
TEST_P(conv_dyn_test, convolution_gpu_bfyx_os_iyx_osv16_no_bias) {
    auto& engine = get_test_engine();
    auto p = GetParam();

    auto is_grouped = p.wei_shape.size() == 5;
    auto groups_num = is_grouped ? static_cast<uint32_t>(p.wei_shape[0]) : 1;

    auto calculate_ref = [&](memory::ptr input, memory::ptr weights, ExecutionConfig config) {
        auto in_layout = input->get_layout();

        topology topology_ref(
            input_layout("input", in_layout),
            data("weights", weights),
            convolution("conv", input_info("input"), "weights", no_bias, groups_num, p.stride, p.dilation, p.pad_begin, p.pad_end, is_grouped));

        network network_ref(engine, topology_ref, config);
        network_ref.set_input_data("input", input);

        auto outputs_ref = network_ref.execute();

        return outputs_ref.at("conv").get_memory();
    };

    auto in_layout = layout{ov::PartialShape{ov::Dimension(), ov::Dimension(p.in_shape[1]), ov::Dimension(), ov::Dimension()}, data_types::f32, format::bfyx};
    auto input = engine.allocate_memory({ p.in_shape, data_types::f32, format::bfyx });
    auto weights = engine.allocate_memory({p.wei_shape, data_types::f32, is_grouped ? format::bfzyx : format::bfyx});

    tests::random_generator rg(GET_SUITE_NAME);
    VF<float> input_rnd = rg.generate_random_1d<float>(ov::shape_size(p.in_shape), -10, 10);
    VF<float> weights_rnd = rg.generate_random_1d<float>(ov::shape_size(p.wei_shape), -10, 10);

    set_values(input, input_rnd);
    set_values(weights, weights_rnd);

    topology topology(
        input_layout("input", in_layout),
        data("weights", weights),
        convolution("conv", input_info("input"), "weights", no_bias, groups_num, p.stride, p.dilation, p.pad_begin, p.pad_end, is_grouped));

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl = { format::bfyx, "convolution_gpu_bfyx_os_iyx_osv16", impl_types::ocl };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", conv_impl } }));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::enable_profiling(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("conv");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output_memory = outputs.at("conv").get_memory();
    auto output_memory_ref = calculate_ref(input, weights, config);

    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
    cldnn::mem_lock<float> output_ptr_ref(output_memory_ref, get_test_stream());

    ASSERT_EQ(outputs.at("conv").get_layout(), output_memory_ref->get_layout());
    for (size_t i = 0; i < output_ptr.size(); i++) {
        ASSERT_EQ(output_ptr[i], output_ptr_ref[i]);
    }

    {
        // Change original shape for the second run
        auto new_shape = p.in_shape;
        new_shape[2] += 4;
        new_shape[3] += 8;

        auto input = engine.allocate_memory({ new_shape, data_types::f32, format::bfyx });

        VF<float> input_rnd = rg.generate_random_1d<float>(ov::shape_size(p.in_shape), -10, 10);
        set_values(input, input_rnd);

        network.set_input_data("input", input);
        auto outputs = network.execute();

        auto output_memory = outputs.at("conv").get_memory();
        auto output_memory_ref = calculate_ref(input, weights, config);

        cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
        cldnn::mem_lock<float> output_ptr_ref(output_memory_ref, get_test_stream());

        ASSERT_EQ(outputs.at("conv").get_layout(), output_memory_ref->get_layout());
        for (size_t i = 0; i < output_ptr.size(); i++) {
            ASSERT_EQ(output_ptr[i], output_ptr_ref[i]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, conv_dyn_test,
    testing::ValuesIn(std::vector<conv_dyn_test_params>{
    { ov::Shape{1, 8, 14, 14}, ov::Shape{16, 8, 3, 3}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0} },
    { ov::Shape{1, 8, 32, 32}, ov::Shape{16, 8, 3, 3}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0} },
    { ov::Shape{1, 8, 60, 60}, ov::Shape{16, 8, 3, 3}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0} },
    { ov::Shape{1, 8, 64, 64}, ov::Shape{16, 8, 3, 3}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0} },
    { ov::Shape{1, 8, 110, 111}, ov::Shape{16, 8, 3, 3}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0} },
    { ov::Shape{1, 8, 110, 111}, ov::Shape{16, 8, 3, 3}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1} },
    { ov::Shape{1, 8, 110, 111}, ov::Shape{16, 8, 5, 5}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1} },
    { ov::Shape{2, 640, 32, 32}, ov::Shape{640, 640, 3, 3}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1} },
    { ov::Shape{1, 32, 16, 16}, ov::Shape{32, 32, 3, 3}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1} },
    { ov::Shape{2, 32, 16, 16}, ov::Shape{32, 32, 3, 3}, ov::Strides{2, 2}, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1} },
    { ov::Shape{1, 32, 32, 32}, ov::Shape{64, 32, 1, 1}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0} },
    { ov::Shape{1, 4, 32, 32}, ov::Shape{32, 4, 3, 3}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1} },
    { ov::Shape{1, 4, 64, 64}, ov::Shape{4, 4, 1, 1}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0} },
    { ov::Shape{1, 32, 28, 28}, ov::Shape{32, 1, 1, 3, 3}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1} },
    { ov::Shape{1, 48, 16, 16}, ov::Shape{48, 48, 4, 4}, ov::Strides{4, 4}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0} },
    { ov::Shape{1, 16, 28, 28}, ov::Shape{32, 16, 2, 2}, ov::Strides{2, 2}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0} },
    { ov::Shape{1, 3, 32, 32}, ov::Shape{96, 3, 4, 4}, ov::Strides{4, 4}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0} },
    { ov::Shape{1, 768, 7, 7}, ov::Shape{768, 1, 1, 3, 3}, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1} },
    { ov::Shape{1, 48, 56, 56}, ov::Shape{48, 48, 8, 8}, ov::Strides{8, 8}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0} },
}));

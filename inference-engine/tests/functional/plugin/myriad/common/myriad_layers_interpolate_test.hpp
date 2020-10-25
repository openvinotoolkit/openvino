// // Copyright (C) 2020 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

// #include <cmath>
// #include "myriad_layers_tests.hpp"

// using namespace InferenceEngine;

// #define ERROR_BOUND 1e-3

// PRETTY_PARAM(Factor, float)
// PRETTY_PARAM(Antialias, int)
// PRETTY_PARAM(HwOptimization, bool);
// PRETTY_PARAM(CustomConfig, std::string);

// typedef myriadLayerTestBaseWithParam<std::tuple<SizeVector, Factor, Antialias, HwOptimization, CustomConfig>>
// 	myriadInterpolateLayerTests_nightly;

// float getOriginalCoordinate(float x_resized, float x_scale, int length_resized, int length_original, int mode) {
//     switch(mode) {
//         case static_cast<int>(InterpolateCoordTransMode::half_pixel) : {
//             return (x_scale != 0)? ((x_resized + 0.5) / x_scale) - 0.5 : 0.0f;
//             break;
//         }
//         case static_cast<int>(InterpolateCoordTransMode::pytorch_half_pixel) : {
//             if (length_resized > 1) {
//                 return (x_scale != 0)? ((x_resized + 0.5) / x_scale) - 0.5 : 0.0f;
//             } else {
//                 return 0.0f;
//             }
//             break;
//         }
//         case static_cast<int>(InterpolateCoordTransMode::asymmetric) : {
//             return (x_scale != 0)? (x_resized / x_scale) : 0.0f;
//             break;
//         }
//         case static_cast<int>(InterpolateCoordTransMode::tf_half_pixel_for_nn) : {
//             return (x_scale != 0)? ((x_resized + 0.5) / x_scale) : 0.0f;
//             break;
//         }
//         case static_cast<int>(InterpolateCoordTransMode::align_corners) : {
//             if (length_resized - 1 == 0) {
//                 return 0.0f;
//             } else {
//                 return x_resized * static_cast<float>(length_original - 1) / (length_resized - 1);
//             }
//             break;
//         }
//         default: {
//             std::cout << "Interpolate does not support this coordinate transformation mode";
//             return 0.0f;
//             break;
//         }
//     }
// }

// int getNearestPixel(float originalValue, bool isDownsample, int mode) {
//     switch (mode) {
//         case static_cast<int>(InterpolateNearestMode::round_prefer_floor): {
//             if (originalValue == (static_cast<int>(originalValue) + 0.5f)) {
//                 return static_cast<int>(std::floor(originalValue));
//             } else {
//                 return static_cast<int>(std::round(originalValue));
//             }
//             break;
//         }
//         case static_cast<int>(InterpolateNearestMode::round_prefer_ceil): {
//             return static_cast<int>(std::round(originalValue));
//             break;
//         }
//         case static_cast<int>(InterpolateNearestMode::floor): {
//             return static_cast<int>(std::floor(originalValue));
//             break;
//         }
//         case static_cast<int>(InterpolateNearestMode::ceil): {
//             return static_cast<int>(std::ceil(originalValue));
//             break;
//         }
//         case static_cast<int>(InterpolateNearestMode::simple): {
//             if (isDownsample) {
//                 return static_cast<int>(std::ceil(originalValue));
//             } else {
//                 return static_cast<int>(originalValue);
//             }
//         }
//         default: {
//             std::cout << "Interpolate does not support this nearest round mode";
//             return 0;
//             break;
//         }
//     }
// }

// int changeCoord(int length, int pos) {
//     return std::max(static_cast<int>(0), std::min(pos, length - 1));
// }

// static inline float triangleCoeff(float x)
// {
//     return (1.0f - fabsf(x));
// }

// void refNearestInterpolate(const Blob::Ptr src, Blob::Ptr dst, int antialias) {
//     ie_fp16 *src_data = static_cast<ie_fp16*>(src->buffer());
//     ie_fp16 *output_sequences = static_cast<ie_fp16*>(dst->buffer());
//     ASSERT_NE(src_data, nullptr);
//     ASSERT_NE(output_sequences, nullptr);

//     const auto& src_dims = src->getTensorDesc().getDims();
//     const auto& dst_dims = dst->getTensorDesc().getDims();
//     int OH = dst_dims[2];
//     int OW = dst_dims[3];

//     int C  = src_dims[1];
//     int IH = src_dims[2];
//     int IW = src_dims[3];

//     if (IH == OH && IW == OW)
//     {
//     	std::copy(src_data, src_data + C*IH*IW, output_sequences);
//         return;
//     }

//     const float fy = static_cast<float>(IH) / static_cast<float>(OH);
//     const float fx = static_cast<float>(IW) / static_cast<float>(OW);
//     const float fz = 1;
    
//     std::vector<int> ind(OD + OH + OW, 1);
//     bool isDDownsample = (fz < 1) ? true : false;
//     bool isHDownsample = (fy < 1) ? true : false;
//     bool isWDownsample = (fx < 1) ? true : false;

//     for (int oz = 0; oz < OD; oz++) {
//         float iz = getOriginalCoordinate(float(oz), fz, OD, ID, 0);
//         ind[oz] = getNearestPixel(iz, isDDownsample, 0);
//         ind[oz] = changeCoord(ind[oz], ID);
//     }
//     for (int oy = 0; oy < OH; oy++) {
//         float iy = getOriginalCoordinate(float(oy), fy, OH, IH, 0);
//         ind[OD + oy] = getNearestPixel(iy, isHDownsample, 0);
//         ind[OD + oy] = changeCoord(ind[OD + oy], IH);
//     }
//     for (int ox = 0; ox < OW; ox++) {
//         float ix = getOriginalCoordinate(float(ox), fx, OW, IW, 0);
//         ind[OD + OH + ox] = getNearestPixel(ix, isWDownsample, 0);
//         ind[OD + OH + ox] = changeCoord(ind[OD + OH + ox], IW);
//     }
//     int *index_d = static_cast<int*>(&ind[0]);
//     int *index_h = static_cast<int*>(&ind[OD]);
//     int *index_w = static_cast<int*>(&ind[OD + OH]);

//     for (int c = 0; c < C; c++) {
//         const ie_fp16* in_ptr = src_data + IW * IH * c;
//         ie_fp16* out_ptr = output_sequences + OW * OH * c;
//         for (int od = 0; od < OD; od++) {
//             for (int oh = 0; oh < OH; oh++) {
//                 for (int ow = 0; ow < OW; ow++) {
//                     out_ptr[oh * OW + ow] = in_ptr[index_h[oh] * IW + index_w[ow]];
//                 }
//             }
//         }
//     }
// }

// TEST_P(myriadInterpolateLayerTests_nightly, Interpolate) {
//     const SizeVector inputDims = std::get<0>(GetParam());
//     const float factor = std::get<1>(GetParam());
//     const bool antialias = std::get<2>(GetParam());
//     const bool hwOptimization = std::get<3>(GetParam());
//     const std::string customConfig = std::get<4>(GetParam());

//     ASSERT_GT(factor, 0);

//     if (customConfig.empty() && antialias) {
//         GTEST_SKIP() << "Native Interpolate with antialiasing is not supported";
//     }

//     if (!customConfig.empty() && !CheckMyriadX()) {
//         GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
//     }

//     _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

//     const auto outputDims = SizeVector{inputDims[0],
//                                        inputDims[1],
//                                        (size_t)(inputDims[2] * factor),
//                                        (size_t)(inputDims[3] * factor)};

//     SetInputTensors({inputDims});
//     SetOutputTensors({outputDims});

//     std::map<std::string, std::string> params;
//     params["antialias"] = std::to_string((int)antialias);
//     params["factor"] = std::to_string(factor);

//     ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Interpolate").params(params),
//                                                    NetworkInitParams()
//                                                         .useHWOpt(hwOptimization)
//                                                         .lockLayout(true)));

//     ASSERT_TRUE(Infer());
//     ASSERT_NO_FATAL_FAILURE(refNearestInterpolate(_inputMap.begin()->second, _refBlob, antialias));

//     CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
// }

// static std::vector<SizeVector> s_InterpolateInput = {
//         {1, 128, 26, 26},
//         {1, 64, 52, 52},
//         {1, 23, 14, 14}
// };

// static std::vector<CustomConfig> s_CustomConfig = {
//     {""},
// #ifdef VPU_HAS_CUSTOM_KERNELS
//    getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
// #endif
// };


// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocessing.hpp"

namespace ov {
namespace intel_gna {
namespace pre_post_processing {

int16_t ConvertFloatToInt16(float src) {
    float rounding_value = (src > 0) ? 0.5f : -0.5f;
    float value = src + rounding_value;
    if (value > 32767.0) {
        return 32767;
    } else if (value < -32768.0) {
        return -32768;
    }
    return (int16_t)value;
}

int8_t ConvertFloatToInt8(float src) {
    float rounding_value = (src > 0) ? 0.5f : -0.5f;
    float value = src + rounding_value;
    if (value > 127.0) {
        return 127;
    } else if (value < -128.0) {
        return -128;
    }
    return (int8_t)value;
}

void ConvertToInt16(int16_t* ptr_dst,
                    const float* ptr_src,
                    const uint32_t num_rows,
                    const uint32_t num_columns,
                    const float scale_factor) {
    if (!ptr_dst || !ptr_src) {
        return;
    }
    for (uint32_t i = 0; i < num_rows * num_columns; i++) {
        ptr_dst[i] = ConvertFloatToInt16(ptr_src[i] * scale_factor);
    }
}

using InferenceEngine::Precision;

void export_scores(void* ptr_dst,
                   const void* ptr_src,
                   intel_dnn_orientation_t orientation,
                   uint32_t num_frames,
                   uint32_t num_group,
                   uint32_t num_vector_elements,
                   uint32_t num_active_elements,
                   uint32_t num_vector_stride,
                   const InferenceEngine::Precision& precision_in,
                   const InferenceEngine::Precision& precision_out,
                   const float scale_factor,
                   bool isAvx2Supported) {
    if (ptr_src == nullptr || ptr_dst == nullptr) {
        THROW_GNA_EXCEPTION << "Received null pointer arguments";
    }
    if (precision_out != Precision::I32 && precision_out != Precision::FP32) {
        THROW_GNA_EXCEPTION << "Unsupported target precision for infer : " << precision_out.name();
    }

    bool transpose = orientation == kDnnInterleavedOrientation && num_group > 1 && num_vector_stride > 1;
    // TODO: AVX2 support for zero-padding isn't implemented yet;
    // fall back to the non-vectorized version when it's necessary
    bool needsZeroPadding = (num_active_elements != num_vector_stride) || (num_frames != num_group);

    switch (precision_out) {
    case Precision::FP32:
        switch (precision_in) {
        case Precision::I8:
            if (isAvx2Supported && !needsZeroPadding) {
                convert_matrix_int8_to_fp32_avx(reinterpret_cast<float*>(ptr_dst),
                                                reinterpret_cast<const int8_t*>(ptr_src),
                                                num_vector_stride,
                                                num_frames,
                                                scale_factor,
                                                transpose);
                break;
            }
            unscale_transpose_and_cast(reinterpret_cast<float*>(ptr_dst),
                                       reinterpret_cast<const int8_t*>(ptr_src),
                                       orientation,
                                       num_frames,
                                       num_group,
                                       num_vector_elements,
                                       num_active_elements,
                                       num_vector_stride,
                                       scale_factor);
            break;
        case Precision::I16:
            if (isAvx2Supported && !needsZeroPadding) {
                convert_matrix_int16_to_fp32_avx(reinterpret_cast<float*>(ptr_dst),
                                                 reinterpret_cast<const int16_t*>(ptr_src),
                                                 num_vector_stride,
                                                 num_frames,
                                                 scale_factor,
                                                 transpose);
                break;
            }
            unscale_transpose_and_cast(reinterpret_cast<float*>(ptr_dst),
                                       reinterpret_cast<const int16_t*>(ptr_src),
                                       orientation,
                                       num_frames,
                                       num_group,
                                       num_vector_elements,
                                       num_active_elements,
                                       num_vector_stride,
                                       scale_factor);
            break;
        case Precision::I32:
            if (isAvx2Supported && !needsZeroPadding) {
                convert_matrix_int32_to_fp32_avx(reinterpret_cast<float*>(ptr_dst),
                                                 reinterpret_cast<const int32_t*>(ptr_src),
                                                 num_vector_stride,
                                                 num_frames,
                                                 scale_factor,
                                                 transpose);
                break;
            }
            unscale_transpose_and_cast(reinterpret_cast<float*>(ptr_dst),
                                       reinterpret_cast<const int32_t*>(ptr_src),
                                       orientation,
                                       num_frames,
                                       num_group,
                                       num_vector_elements,
                                       num_active_elements,
                                       num_vector_stride,
                                       scale_factor);
            break;
        default:
            THROW_GNA_EXCEPTION << "Unsupported data type";
        }
        break;
    case Precision::I32:
        switch (precision_in) {
        case Precision::I8:
            unscale_transpose_and_cast(reinterpret_cast<int32_t*>(ptr_dst),
                                       reinterpret_cast<const int8_t*>(ptr_src),
                                       orientation,
                                       num_frames,
                                       num_group,
                                       num_vector_elements,
                                       num_active_elements,
                                       num_vector_stride,
                                       scale_factor);
            break;
        case Precision::I16:
            unscale_transpose_and_cast(reinterpret_cast<int32_t*>(ptr_dst),
                                       reinterpret_cast<const int16_t*>(ptr_src),
                                       orientation,
                                       num_frames,
                                       num_group,
                                       num_vector_elements,
                                       num_active_elements,
                                       num_vector_stride,
                                       scale_factor);
            break;
        case Precision::I32:
            unscale_transpose_and_cast(reinterpret_cast<int32_t*>(ptr_dst),
                                       reinterpret_cast<const int32_t*>(ptr_src),
                                       orientation,
                                       num_frames,
                                       num_group,
                                       num_vector_elements,
                                       num_active_elements,
                                       num_vector_stride,
                                       scale_factor);
            break;
        default:
            THROW_GNA_EXCEPTION << "Unsupported data type";
        }
        break;

    default:
        THROW_GNA_EXCEPTION << "Unsupported data type";
    }
}

void import_frames(void* ptr_dst,
                   const void* ptr_src,
                   const InferenceEngine::Precision& input_precision,
                   float scaleFactor,
                   intel_dnn_orientation_t orientation,
                   uint32_t num_frames,
                   uint32_t num_group,
                   uint32_t num_vector_elements,
                   uint32_t num_vector_stride,
                   bool input_low_precision,
                   bool isGnaDevice,
                   bool isAvx2Supported) {
    switch (input_precision) {
    case Precision::U8:
    case Precision::I8: {
        auto src = reinterpret_cast<const uint8_t*>(ptr_src);
        if (!input_low_precision) {
            auto dst = reinterpret_cast<int16_t*>(ptr_dst);
            copy_input_data(dst,
                            src,
                            num_frames,
                            num_group,
                            num_vector_elements,
                            num_vector_stride,
                            orientation,
                            scaleFactor,
                            input_low_precision);
        } else {
            auto dst = reinterpret_cast<int8_t*>(ptr_dst);
            copy_input_data(dst,
                            src,
                            num_frames,
                            num_group,
                            num_vector_elements,
                            num_vector_stride,
                            orientation,
                            scaleFactor,
                            input_low_precision);
        }
        break;
    }
    case Precision::I16: {
        auto src = reinterpret_cast<const int16_t*>(ptr_src);
        if (!input_low_precision) {
            auto dst = reinterpret_cast<int16_t*>(ptr_dst);
            copy_input_data(dst,
                            src,
                            num_frames,
                            num_group,
                            num_vector_elements,
                            num_vector_stride,
                            orientation,
                            scaleFactor,
                            input_low_precision);
        } else {
            auto dst = reinterpret_cast<int8_t*>(ptr_dst);
            copy_input_data(dst,
                            src,
                            num_frames,
                            num_group,
                            num_vector_elements,
                            num_vector_stride,
                            orientation,
                            scaleFactor,
                            input_low_precision);
        }
        break;
    }
    case Precision::FP32: {
        auto src = reinterpret_cast<const float*>(ptr_src);
        if (!isGnaDevice) {
            auto dst = reinterpret_cast<float*>(ptr_dst);
            copy_input_data(dst,
                            src,
                            num_frames,
                            num_group,
                            num_vector_elements,
                            num_vector_stride,
                            orientation,
                            scaleFactor,
                            input_low_precision);
        } else {
            bool transpose = orientation == kDnnInterleavedOrientation && num_group > 1 && num_vector_stride > 1;
            // TODO: AVX2 support for zero-padding isn't implemented yet;
            // fall back to the non-vectorized version when it's necessary
            bool needsZeroPadding = (num_vector_elements != num_vector_stride) || (num_frames != num_group);

            if (!input_low_precision) {
                auto dst = reinterpret_cast<int16_t*>(ptr_dst);

                if (isAvx2Supported && !needsZeroPadding) {
                    convert_matrix_fp32_to_int16_avx(dst, src, num_group, num_vector_stride, scaleFactor, transpose);
                    break;
                }

                copy_input_data(dst,
                                src,
                                num_frames,
                                num_group,
                                num_vector_elements,
                                num_vector_stride,
                                orientation,
                                scaleFactor,
                                input_low_precision);
            } else {
                auto dst = reinterpret_cast<int8_t*>(ptr_dst);

                if (isAvx2Supported && !needsZeroPadding) {
                    convert_matrix_fp32_to_int8_avx(dst, src, num_group, num_vector_stride, scaleFactor, transpose);
                    break;
                }

                copy_input_data(dst,
                                src,
                                num_frames,
                                num_group,
                                num_vector_elements,
                                num_vector_stride,
                                orientation,
                                scaleFactor,
                                input_low_precision);
            }
        }
        break;
    }
    case Precision::I32: {
        auto src = reinterpret_cast<const float*>(ptr_src);
        if (!isGnaDevice) {
            auto dst = reinterpret_cast<float*>(ptr_dst);
            copy_input_data(dst,
                            src,
                            num_frames,
                            num_group,
                            num_vector_elements,
                            num_vector_stride,
                            orientation,
                            scaleFactor,
                            input_low_precision);
        } else {
            if (!input_low_precision) {
                auto dst = reinterpret_cast<int16_t*>(ptr_dst);
                copy_input_data(dst,
                                src,
                                num_frames,
                                num_group,
                                num_vector_elements,
                                num_vector_stride,
                                orientation,
                                scaleFactor,
                                input_low_precision);
            } else {
                auto dst = reinterpret_cast<int8_t*>(ptr_dst);
                copy_input_data(dst,
                                src,
                                num_frames,
                                num_group,
                                num_vector_elements,
                                num_vector_stride,
                                orientation,
                                scaleFactor,
                                input_low_precision);
            }
        }
        break;
    }
    default:
        break;
    }
}
}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov

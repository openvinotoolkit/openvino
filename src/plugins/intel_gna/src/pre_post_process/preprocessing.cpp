// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocessing.hpp"

namespace ov {
namespace intel_gna {
namespace pre_post_processing {

void ConvertToInt16(int16_t* ptr_dst,
                    const float* ptr_src,
                    const size_t num_rows,
                    const size_t num_columns,
                    const float scale_factor) {
    if (!ptr_dst || !ptr_src) {
        return;
    }
    for (size_t i = 0; i < num_rows * num_columns; i++) {
        ptr_dst[i] = FloatToInt<int16_t>(ptr_src[i] * scale_factor);
    }
}

using InferenceEngine::Precision;

void export_scores(void* ptr_dst,
                   const void* ptr_src,
                   intel_dnn_orientation_t orientation,
                   size_t num_frames,
                   size_t num_group,
                   size_t num_vector_elements,
                   size_t num_active_elements,
                   size_t num_vector_stride,
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
                   size_t num_frames,
                   size_t num_group,
                   size_t num_vector_elements,
                   size_t num_vector_stride,
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

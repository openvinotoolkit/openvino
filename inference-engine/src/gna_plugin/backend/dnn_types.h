// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <type_traits>

#include "gna_types.h"
#include "gna_plugin_log.hpp"

enum DnnActivationType : uint8_t {
    kActNone,
    kActSigmoid,
    kActTanh,
    kActRelu,
    kActLeakyRelu,
    kActIdentity,
    kActKaldiLstmClipping,
    kActCustom,
    kActExp,
    kActLog,
    kActSign,
    kActAbs,
    kActNegLog,
    kActNegHalfLog,
    kActSoftSign,
    kActPow,
    kActFakeQuantize,
    kActNumType
};

struct FakeQuantizeParams {
    int8_t set;
    int32_t levels;
    // if input is per-channel quantization - input pointers contains per-channel ranges
    int8_t  inputPerChannel;
    float* input_low;
    float* input_high;
    // if output is per-channel quantization - output pointers contains per-channel ranges
    int8_t  outputPerChannel;
    float* output_low;
    float* output_high;
};

struct DnnActivation {
    // for prelu
    DnnActivationType type;
    FakeQuantizeParams fqParams;
    FakeQuantizeParams srcFQParams;

    union {
        struct {
            float negative_slope;
        } lrelu;
        struct {
            float exponent;
            float scale;
            float offset;
        } pow;
        struct {
            float low;
            float high;
        } clamp;
    } args;
    operator DnnActivationType () const noexcept {
        return type;
    }
    static DnnActivation fromType(DnnActivationType type) {
        DnnActivation activation;
        activation.type = type;
        activation.args = {};
        return activation;
    }
};

static_assert(std::is_trivial<DnnActivation>::value, "DnnActivation is not trival type");

extern const char *intel_dnn_activation_name[kActNumType];

typedef enum DnnSoftmaxType {
    kSoftmaxNone,
    kSoftmaxKaldiSumgroup,
    kSoftmaxEesen,
    kSoftmaxGoogle,
    kSoftmaxNumType
} intel_dnn_softmax_type_t;

extern const char *intel_dnn_softmax_name[kSoftmaxNumType];

typedef enum {
    kDnnUnknownOrientation = 100,
    kDnnInterleavedOrientation,
    kDnnNonInterleavedOrientation,
    kDnnNumOrientation
} intel_dnn_orientation_t;

typedef enum {
    kDnnNullOp,
    kDnnAffineOp,
    kDnnDiagonalOp,
    kDnnConvolutional1dOp,
    kDnnConvolutional2dOp,
    kDnnPiecewiselinearOp,
    kDnnMaxPoolOp,
    kDnnRecurrentOp,
    kDnnInterleaveOp,
    kDnnDeinterleaveOp,
    kDnnCopyOp,
    kDnnNumOp
} intel_dnn_operation_t;

extern const char* intel_dnn_operation_name[kDnnNumOp];

typedef enum {
    kDnnMacroOpNone,
    kDnnMacroOpLstm,
    kDnnMacroOpBiLstm,
    kDnnNumMacroOp
} intel_dnn_macro_operation_t;

extern const char *intel_dnn_macro_operation_name[kDnnNumMacroOp];

typedef enum {
    kDnnFloat,
    kDnnInt,
    kDnnNumNumberType
} intel_dnn_number_type_t;

extern const char *intel_dnn_number_type_name[kDnnNumNumberType];

typedef struct {
    uint32_t num_bytes_per_weight;
    uint32_t num_bytes_per_bias;
    float weight_scale_factor;
    void *ptr_weights;
    void *ptr_biases;
} intel_affine_t;

typedef struct {
    uint32_t num_bytes_per_weight;
    uint32_t num_bytes_per_bias;
    uint32_t num_filters;
    uint32_t num_filter_rows;
    uint32_t num_filter_coefficients;
    uint32_t num_feature_maps;
    uint32_t num_feature_map_rows;
    uint32_t num_feature_map_columns;
    float weight_scale_factor;
    void *ptr_filters;     // filters stored one after the other
    void *ptr_biases;
} intel_convolutionalD_t;

typedef struct {
    std::array<uint32_t, 2> convStride;
    std::array<uint32_t, 2> zeroPadding;
    float weight_scale_factor;
    void* ptr_filters;     // filters stored one after the other
    void* ptr_biases;
} intel_convolutional2D_t;

typedef struct {
    std::array<uint32_t, 2> poolingWindowXY;
    std::array<uint32_t, 2> poolingStrideXY;
    std::array<uint32_t, 3> inCHW;
    std::array<uint32_t, 3> outCHW;
} intel_maxpool_t;

typedef struct {
    DnnActivation func_id;       // identifies function being approximated
    uint32_t num_segments;
    gna_pwl_segment_t *ptr_segments;
} intel_piecewiselinear_t;

typedef struct {
    uint32_t num_vector_delay;
    uint32_t num_bytes_per_weight;
    uint32_t num_bytes_per_bias;
    float weight_scale_factor;
    void *ptr_feedbacks;
    void *ptr_weights;
    void *ptr_biases;
} intel_recurrent_t;

typedef struct {
} intel_interleave_t;

typedef struct {
} intel_deinterleave_t;

typedef struct {
    uint32_t num_copy_columns;        // number of columns to copy
    uint32_t num_copy_rows;            // number of rows to copy
} intel_copy_t;

#if GNA_LIB_VER == 2
enum OvGnaType {
    OvGnaTypeInt8 = 1,
    OvGnaTypeInt16 = 2,
    OvGnaTypeInt32 = 4,
    OvGnaTypePwl = 8,
};

enum OvGnaMode {
    OvGnaModeDefault = 0,
    OvGnaModeDisabled = -1
};

struct OvGnaTensor {
    std::vector<uint32_t> dimensions;
    OvGnaType type;
    OvGnaMode mode;
};

template <class T>
OvGnaType OvGnaTypeIntFromBytes(T bytesPerElement) {
    static const std::map<T, OvGnaType> m = {
        {1, OvGnaTypeInt8},
        {2, OvGnaTypeInt16},
        {4, OvGnaTypeInt32}
    };
    const auto r = m.find(bytesPerElement);
    if (r == m.end()) {
        THROW_GNA_EXCEPTION << "OvGnaTypeIntFromBytes: unknown bytesPerElement == " << bytesPerElement;
    }
    return r->second;
}

static std::string OvGnaTypeToString(OvGnaType type) {
    static const std::map<OvGnaType, std::string> typeToString = {
        {OvGnaTypeInt8, "OvGnaTypeInt8"},
        {OvGnaTypeInt16, "OvGnaTypeInt16"},
        {OvGnaTypeInt32, "OvGnaTypeInt32"},
        {OvGnaTypePwl, "OvGnaTypePwl"},
    };
    const auto r = typeToString.find(type);
    if (r == typeToString.end()) {
        THROW_GNA_EXCEPTION << "OvGnaTypeToString: unknown type == " << type;
    }
    return r->second;
}

static std::string OvGnaModeToString(OvGnaMode mode) {
    static const std::map<OvGnaMode, std::string> modeToString = {
        {OvGnaModeDefault, "OvGnaModeDefault"},
        {OvGnaModeDisabled, "OvGnaModeDisabled"},
    };
    const auto r = modeToString.find(mode);
    if (r == modeToString.end()) {
        THROW_GNA_EXCEPTION << "OvGnaModeToString: unknown mode == " << mode;
    }
    return r->second;
}
#endif

typedef struct {
#if GNA_LIB_VER == 2
    std::vector < OvGnaTensor > tensors;
#endif
    uint32_t num_rows_in;
    uint32_t num_columns_in;
    uint32_t num_rows_out;
    uint32_t num_columns_out;
    uint32_t num_bytes_per_input;
    uint32_t num_bytes_per_output;
    intel_dnn_operation_t operation;
    intel_dnn_macro_operation_t macro_operation;
    intel_dnn_orientation_t orientation_in;
    intel_dnn_orientation_t orientation_out;
    union operation_struct_t {
        intel_affine_t affine;
        intel_convolutionalD_t conv1D;
        intel_convolutional2D_t conv2D;
        intel_maxpool_t maxpool;
        intel_piecewiselinear_t pwl;
        intel_recurrent_t recurrent;
        intel_interleave_t interleave;
        intel_deinterleave_t deinterleave;
        intel_copy_t copy;
    } op;
    void *ptr_inputs;
    void *ptr_outputs;
    float output_scale_factor;
    float input_scale_factor;
    const char * original_layer_name = nullptr;
} intel_dnn_component_t;

typedef struct {
    uint32_t num_scores;
    uint32_t num_errors;
    float threshold;
    float max_error;
    float rms_error;
    float sum_error;
    float sum_rms_error;
    float sum_squared_error;
    float max_rel_error;
    float sum_rel_error;
    float sum_squared_rel_error;
} intel_score_error_t;

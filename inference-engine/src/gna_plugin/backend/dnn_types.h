// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <type_traits>
#include <gna-api-types-xnn.h>

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
    kActDivByN,
    kActNumType
};

struct DnnActivation {
    // for prelu
    DnnActivationType type;
    float negative_slope;
    operator DnnActivationType () const noexcept {
        return type;
    }
    static DnnActivation fromType(DnnActivationType type) {
        DnnActivation activation;
        activation.type = type;
        activation.negative_slope = 0.0f;
        return activation;
    }
};

static_assert(std::is_trivial<DnnActivation>::value, "DnnActivation is not trival type");

static const char *intel_dnn_activation_name[kActNumType] = {
        "kActNone",
        "kActSigmoid",
        "kActTanh",
        "kActRelu",
        "kActLeakyRelu",
        "kActIdentity",
        "kActKaldiLstmClipping",
        "kActCustom",
        "kActExp",
        "kActLog",
        "kActDivByN"
};

typedef enum DnnSoftmaxType {
    kSoftmaxNone,
    kSoftmaxKaldiSumgroup,
    kSoftmaxEesen,
    kSoftmaxGoogle,
    kSoftmaxNumType
} intel_dnn_softmax_type_t;

static const char *intel_dnn_softmax_name[kSoftmaxNumType] = {
        "kSoftmaxNone",
        "kSoftmaxKaldiSumGroup",
        "kSoftmaxKaldiApplyLog",
        "kSoftmaxGoogle"
};

typedef enum {
    kDnnUnknownOrientation,
    kDnnInterleavedOrientation,
    kDnnNonInterleavedOrientation,
    kDnnNumOrientation
} intel_dnn_orientation_t;

typedef enum {
    kDnnNullOp,
    kDnnAffineOp,
    kDnnDiagonalOp,
    kDnnConvolutional1dOp,
    kDnnPiecewiselinearOp,
    kDnnMaxPoolOp,
    kDnnRecurrentOp,
    kDnnInterleaveOp,
    kDnnDeinterleaveOp,
    kDnnCopyOp,
    kDnnNumOp
} intel_dnn_operation_t;

static const char *intel_dnn_operation_name[kDnnNumOp] = {
        "kDnnNullOp",
        "kDnnAffineOp",
        "kDnnDiagonalOp",
        "kDnnConvolutional1dOp",
        "kDnnPiecewiselinearOp",
        "kDnnMaxPoolOp",
        "kDnnRecurrentOp",
        "kDnnInterleaveOp",
        "kDnnDeinterleaveOp",
        "kDnnCopyOp"
};

typedef enum {
    kDnnMacroOpNone,
    kDnnMacroOpLstm,
    kDnnMacroOpBiLstm,
    kDnnNumMacroOp
} intel_dnn_macro_operation_t;

static const char *intel_dnn_macro_operation_name[kDnnNumMacroOp] = {
        "kDnnMacroOpNone",
        "kDnnMacroOpLstm",
        "kDnnMacroOpBiLstm"
};

typedef enum {
    kDnnFloat,
    kDnnInt,
    kDnnNumNumberType
} intel_dnn_number_type_t;

static const char *intel_dnn_number_type_name[kDnnNumNumberType] = {
        "kDnnFloat",
        "kDnnInt"
};

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
    uint32_t num_inputs;         // pool size
    uint32_t num_inputs_step;     // pool step
    uint32_t num_inputs_stride;  // pool stride (number of convolution filters)
    bool do_sum_not_max;
} intel_maxpool_t;

typedef struct {
    DnnActivation func_id;       // identifies function being approximated
    uint32_t num_segments;
    intel_pwl_segment_t *ptr_segments;
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

typedef struct {
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
#ifdef PLOT
    const char * orignal_layer_name = nullptr;
#endif
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

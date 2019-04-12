// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <memory.h>
#include <xmmintrin.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <type_traits>
#include <vector>
#include "gna-api.h"

#define DNN_MAX_BATCH_SIZE 8
#define DNN_MAX_INPUTS 3072
#define DNN_MAX_OUTPUTS 8192
#define DNN_MAX_ERROR 1.0e-4f
#define DNN_NUM_BYTES_INT_BIAS 4
#define DNN_NUM_BYTES_INT_AFFINE_OUT 4
#define DNN_RAND_INT8_AMPLITUDE 127.0f
#define DNN_RAND_INT16_AMPLITUDE 16384.0f
#define DNN_RAND_INT32_AMPLITUDE 1048576.0f
#define DNN_RAND_FLOAT32_AMPLITUDE 8.0f

enum DnnActivationType {
    kActNone,
    kActSigmoid,
    kActTanh,
    kActRelu,
    kActLeakyRelu,
    kActIdentity,
    kActKaldiLstmClipping,
    kActCustom,
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
    "kActCustom"
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

class AmIntelDnn {
 public:
    AmIntelDnn()
        : ptr_active_outputs_(NULL),
          num_active_outputs_(0),
          input_scale_factor_(1.0),
          num_left_context(0),
          num_right_context(0),
          do_rotate_input(false),
          num_rotate_rows(0),
          num_rotate_columns(0),
          softmax_type(kSoftmaxNone),
          ptr_sumgroup_sizes(NULL),
          num_sumgroup_sizes(0),
          ptr_priors(NULL),
          ptr_dnn_memory_(NULL) {
    }

    ~AmIntelDnn() {
        component.clear();
        if (ptr_sumgroup_sizes != NULL) {
            _mm_free(ptr_sumgroup_sizes);
        }
        if (ptr_priors != NULL) {
            _mm_free(ptr_priors);
        }
    }

    uint32_t num_components() { return (uint32_t) component.size(); }

    void Init(void *ptr_memory, uint32_t num_memory_bytes, intel_dnn_number_type_t number_type, float scale_factor);
    void InitActiveList(uint32_t *ptr_active_list);

    template<class A, class B, class C, class D>
    static void InitAffineComponent(intel_dnn_component_t &comp,
                             uint32_t num_rows_in,
                             uint32_t num_columns,
                             uint32_t num_rows_out,
                             uint32_t num_bytes_per_input,
                             uint32_t num_bytes_per_output,
                             uint32_t num_bytes_per_weight,
                             uint32_t num_bytes_per_bias,
                             float weight_scale_factor,
                             float output_scale_factor,
                             A *&ptr_inputs,
                             B *&ptr_outputs,
                             C *&ptr_weights,
                             D *&ptr_biases,
                             bool isDiag = false) {
        InitAffineComponentPrivate(comp,
                                   num_rows_in,
                                   num_columns,
                                   num_rows_out,
                                   num_bytes_per_input,
                                   num_bytes_per_output,
                                   num_bytes_per_weight,
                                   num_bytes_per_bias,
                                   weight_scale_factor,
                                   output_scale_factor,
                                   (void *&) ptr_inputs,
                                   (void *&) ptr_outputs,
                                   (void *&) ptr_weights,
                                   (void *&) ptr_biases,
                                   isDiag,
                                   true);
    }

    template<class A, class B, class C, class D>
    void InitAffineComponent(uint32_t component_index,
                             uint32_t num_rows_in,
                             uint32_t num_columns,
                             uint32_t num_rows_out,
                             uint32_t num_bytes_per_input,
                             uint32_t num_bytes_per_output,
                             uint32_t num_bytes_per_weight,
                             uint32_t num_bytes_per_bias,
                             float weight_scale_factor,
                             float output_scale_factor,
                             A *&ptr_inputs,
                             B *&ptr_outputs,
                             C *&ptr_weights,
                             D *&ptr_biases,
                             bool isDiag = false) {
        InitAffineComponentPrivate(component[component_index],
                                   num_rows_in,
                                   num_columns,
                                   num_rows_out,
                                   num_bytes_per_input,
                                   num_bytes_per_output,
                                   num_bytes_per_weight,
                                   num_bytes_per_bias,
                                   weight_scale_factor,
                                   output_scale_factor,
                                   (void *&) ptr_inputs,
                                   (void *&) ptr_outputs,
                                   (void *&) ptr_weights,
                                   (void *&) ptr_biases,
                                   isDiag,
                                   false);
    }

    void InitDiagonalComponent(uint32_t component_index,
                               uint32_t num_rows_in,
                               uint32_t num_columns,
                               uint32_t num_rows_out,
                               uint32_t num_bytes_per_input,
                               uint32_t num_bytes_per_output,
                               uint32_t num_bytes_per_weight,
                               uint32_t num_bytes_per_bias,
                               float weight_scale_factor,
                               float output_scale_factor,
                               void *ptr_inputs,
                               void *ptr_outputs,
                               void *ptr_weights,
                               void *ptr_biases);

    template<class A, class B, class C, class D>
    void InitConvolutional1DComponent(uint32_t component_index,
                                      uint32_t num_rows_in,
                                      uint32_t num_columns_in,
                                      uint32_t num_rows_out,
                                      uint32_t num_columns_out,
                                      uint32_t num_bytes_per_input,
                                      uint32_t num_bytes_per_output,
                                      uint32_t num_bytes_per_weight,
                                      uint32_t num_bytes_per_bias,
                                      uint32_t num_filters,
                                      uint32_t num_filter_rows,
                                      uint32_t num_filter_coefficients,
                                      uint32_t num_feature_maps,
                                      uint32_t num_feature_map_rows,
                                      uint32_t num_feature_map_columns,
                                      float weight_scale_factor,
                                      float output_scale_factor,
                                      A *& ptr_inputs,
                                      B *& ptr_outputs,
                                      C *& ptr_filters,
                                      D *& ptr_biases) {
        InitConvolutional1DComponentPrivate(component[component_index],
                                            num_rows_in,
                                            num_columns_in,
                                            num_rows_out,
                                            num_columns_out,
                                            num_bytes_per_input,
                                            num_bytes_per_output,
                                            num_bytes_per_weight,
                                            num_bytes_per_bias,
                                            num_filters,
                                            num_filter_rows,
                                            num_filter_coefficients,
                                            num_feature_maps,
                                            num_feature_map_rows,
                                            num_feature_map_columns,
                                            weight_scale_factor,
                                            output_scale_factor,
                                            (void *&) ptr_inputs,
                                            (void *&) ptr_outputs,
                                            (void *&) ptr_filters,
                                            (void *&) ptr_biases,
                                            false);
    }

    template<class A, class B, class C, class D>
    static void InitConvolutional1DComponent(intel_dnn_component_t &comp,
                                      uint32_t num_rows_in,
                                      uint32_t num_columns_in,
                                      uint32_t num_rows_out,
                                      uint32_t num_columns_out,
                                      uint32_t num_bytes_per_input,
                                      uint32_t num_bytes_per_output,
                                      uint32_t num_bytes_per_weight,
                                      uint32_t num_bytes_per_bias,
                                      uint32_t num_filters,
                                      uint32_t num_filter_rows,
                                      uint32_t num_filter_coefficients,
                                      uint32_t num_feature_maps,
                                      uint32_t num_feature_map_rows,
                                      uint32_t num_feature_map_columns,
                                      float weight_scale_factor,
                                      float output_scale_factor,
                                      A *& ptr_inputs,
                                      B *& ptr_outputs,
                                      C *& ptr_filters,
                                      D *& ptr_biases) {
        InitConvolutional1DComponentPrivate(comp,
                                            num_rows_in,
                                            num_columns_in,
                                            num_rows_out,
                                            num_columns_out,
                                            num_bytes_per_input,
                                            num_bytes_per_output,
                                            num_bytes_per_weight,
                                            num_bytes_per_bias,
                                            num_filters,
                                            num_filter_rows,
                                            num_filter_coefficients,
                                            num_feature_maps,
                                            num_feature_map_rows,
                                            num_feature_map_columns,
                                            weight_scale_factor,
                                            output_scale_factor,
                                            (void *&) ptr_inputs,
                                            (void *&) ptr_outputs,
                                            (void *&) ptr_filters,
                                            (void *&) ptr_biases,
                                            true);
    }



    // TODO: this functions accepted component_index only used in legacy code
    void InitMaxpoolComponent(uint32_t component_index,
                              uint32_t num_rows_in,
                              uint32_t num_columns_in,
                              uint32_t num_rows_out,
                              uint32_t num_columns_out,
                              uint32_t num_bytes_per_input,
                              uint32_t num_bytes_per_output,
                              uint32_t num_pool_size,
                              uint32_t num_pool_step,
                              uint32_t num_pool_stride,
                              bool do_sum_not_max,
                              float output_scale_factor,
                              void * ptr_inputs,
                              void * ptr_outputs) {
        InitMaxpoolComponentPrivate(component[component_index],
            num_rows_in,
            num_columns_in,
            num_rows_out,
            num_columns_out,
            num_bytes_per_input,
            num_bytes_per_output,
            num_pool_size,
            num_pool_step,
            num_pool_stride,
            do_sum_not_max,
            output_scale_factor,
            (void *&) ptr_inputs,
            (void *&) ptr_outputs,
            false);
    }

    template<class A, class B>
    static void InitMaxpoolComponent(intel_dnn_component_t &cmp,
                              uint32_t num_rows_in,
                              uint32_t num_columns_in,
                              uint32_t num_rows_out,
                              uint32_t num_columns_out,
                              uint32_t num_bytes_per_input,
                              uint32_t num_bytes_per_output,
                              uint32_t num_pool_size,
                              uint32_t num_pool_step,
                              uint32_t num_pool_stride,
                              bool do_sum_not_max,
                              float output_scale_factor,
                              A *&ptr_inputs,
                              B *&ptr_outputs) {
        InitMaxpoolComponentPrivate(cmp,
                                    num_rows_in,
                                    num_columns_in,
                                    num_rows_out,
                                    num_columns_out,
                                    num_bytes_per_input,
                                    num_bytes_per_output,
                                    num_pool_size,
                                    num_pool_step,
                                    num_pool_stride,
                                    do_sum_not_max,
                                    output_scale_factor,
                                    (void *&) ptr_inputs,
                                    (void *&) ptr_outputs,
                                    true);
    }




    void InitPiecewiseLinearComponent(uint32_t component_index,
                                      DnnActivation function_id,
                                      intel_dnn_orientation_t orientation,
                                      uint32_t num_rows,
                                      uint32_t num_columns,
                                      uint32_t num_bytes_per_input,
                                      uint32_t num_bytes_per_output,
                                      uint32_t num_segments,
                                      float output_scale_factor,
                                      void * ptr_inputs,
                                      void * ptr_outputs,
                                      intel_pwl_segment_t *ptr_segments) {
        InitPiecewiseLinearComponentPrivate(component[component_index],
                                            function_id,
                                            orientation,
                                            num_rows,
                                            num_columns,
                                            num_bytes_per_input,
                                            num_bytes_per_output,
                                            num_segments,
                                            output_scale_factor,
                                            ptr_inputs,
                                            ptr_outputs,
                                            ptr_segments,
                                            false);
    }
    template<class A, class B>
    static void InitPiecewiseLinearComponent(intel_dnn_component_t &cmp,
                                      DnnActivation function_id,
                                      intel_dnn_orientation_t orientation,
                                      uint32_t num_rows,
                                      uint32_t num_columns,
                                      uint32_t num_bytes_per_input,
                                      uint32_t num_bytes_per_output,
                                      uint32_t num_segments,
                                      float output_scale_factor,
                                      A *&ptr_inputs,
                                      B *&ptr_outputs,
                                      intel_pwl_segment_t *ptr_segments) {
        InitPiecewiseLinearComponentPrivate(cmp,
                                            function_id,
                                            orientation,
                                            num_rows,
                                            num_columns,
                                            num_bytes_per_input,
                                            num_bytes_per_output,
                                            num_segments,
                                            output_scale_factor,
                                            (void *&) ptr_inputs,
                                            (void *&) ptr_outputs,
                                            ptr_segments,
                                            true);
    }


    void InitRecurrentComponent(uint32_t component_index,
                                uint32_t num_rows,
                                uint32_t num_columns_in,
                                uint32_t num_columns_out,
                                uint32_t num_bytes_per_input,
                                uint32_t num_bytes_per_output,
                                uint32_t num_vector_delay,
                                uint32_t num_bytes_per_weight,
                                uint32_t num_bytes_per_bias,
                                float weight_scale_factor,
                                float output_scale_factor,
                                void *ptr_inputs,
                                void *ptr_feedbacks,
                                void *ptr_outputs,
                                void *ptr_weights,
                                void *ptr_biases);
    void InitInterleaveComponent(uint32_t component_index,
                                 uint32_t num_rows,
                                 uint32_t num_columns,
                                 uint32_t num_bytes_per_input,
                                 uint32_t num_bytes_per_output,
                                 float output_scale_factor,
                                 void *ptr_inputs,
                                 void *ptr_outputs);
    void InitDeinterleaveComponent(uint32_t component_index,
                                   uint32_t num_rows,
                                   uint32_t num_columns,
                                   uint32_t num_bytes_per_input,
                                   uint32_t num_bytes_per_output,
                                   float output_scale_factor,
                                   void *ptr_inputs,
                                   void *ptr_outputs);
    void InitCopyComponent(uint32_t component_index,
                           intel_dnn_orientation_t orientation,
                           uint32_t num_rows_in,
                           uint32_t num_columns_in,
                           uint32_t num_rows_out,
                           uint32_t num_columns_out,
                           uint32_t num_bytes_per_input,
                           uint32_t num_bytes_per_output,
                           float output_scale_factor,
                           uint32_t num_copy_rows,
                           uint32_t num_copy_columns,
                           void *ptr_inputs,
                           void *ptr_outputs) {
        InitCopyComponentPrivate(component[component_index],
                                 orientation,
                                 num_rows_in,
                                 num_columns_in,
                                 num_rows_out,
                                 num_columns_out,
                                 num_bytes_per_input,
                                 num_bytes_per_output,
                                 output_scale_factor,
                                 num_copy_rows,
                                 num_copy_columns,
                                 ptr_inputs,
                                 ptr_outputs,
                                 false);
    }

    template<class A, class B>
    static  void InitCopyComponent(intel_dnn_component_t &cmp,
                                   intel_dnn_orientation_t orientation,
                                   uint32_t num_rows_in,
                                   uint32_t num_columns_in,
                                   uint32_t num_rows_out,
                                   uint32_t num_columns_out,
                                   uint32_t num_bytes_per_input,
                                   uint32_t num_bytes_per_output,
                                   float output_scale_factor,
                                   uint32_t num_copy_rows,
                                   uint32_t num_copy_columns,
                                   A *&ptr_inputs,
                                   B *&ptr_outputs) {
        InitCopyComponentPrivate(cmp,
                                 orientation,
                                 num_rows_in,
                                 num_columns_in,
                                 num_rows_out,
                                 num_columns_out,
                                 num_bytes_per_input,
                                 num_bytes_per_output,
                                 output_scale_factor,
                                 num_copy_rows,
                                 num_copy_columns,
                                 (void *&) ptr_inputs,
                                 (void *&) ptr_outputs,
                                 true);
    }
    void AddComponents(uint32_t num_components_to_add);
    void ClearComponent(uint32_t component_index);
    void ClearState();
    uint32_t CopyActiveList(std::vector<std::vector<uint32_t> > &active_list, uint32_t list_index);
    void Propagate();
    intel_dnn_macro_operation_t MacroOperation(uint32_t component_index);
    void SetMacroOperation(uint32_t component_index, intel_dnn_macro_operation_t macro_operation);
    float InputScaleFactor(uint32_t component_index);
    float WeightScaleFactor(uint32_t component_index);
    float OutputScaleFactor(uint32_t component_index) {
        return OutputScaleFactor(component[component_index]);
    }
    float OutputScaleFactor(intel_dnn_component_t &comp);
    void SetInputScaleFactor(float scale_factor) { input_scale_factor_ = scale_factor; }
    void SetOutputScaleFactor(uint32_t component_index, float scale_factor);
    void PrintOutputs(uint32_t component_index);
    uint32_t CompareScores(void *ptr_scores, intel_score_error_t *score_error, uint32_t num_frames);
    void WriteGraphWizModel(const char *filename);
    void WriteDnnText(const char *filename, intel_dnn_number_type_t number_type);
    uint32_t MemoryRequiredToReadDnnText(const char *filename);
    void ReadDnnText(const char *filename, void *ptr_memory, uint32_t num_memory_bytes, float *ptr_scale_in);

    void InitGNAStruct(intel_nnet_type_t *ptr_nnet);
    void DestroyGNAStruct(intel_nnet_type_t *ptr_nnet);
    void GetScaledOutput(float *ptr_output, uint32_t component_index);
    uint32_t *ptr_active_outputs() { return (ptr_active_outputs_); }
    uint32_t num_active_outputs() { return (num_active_outputs_); }
    uint32_t num_gna_layers() {
        uint32_t num_layers = 0;
        for (uint32_t i = 0; i < component.size(); i++) {
            if ((component[i].operation == kDnnAffineOp) || (component[i].operation == kDnnDiagonalOp)
                || (component[i].operation == kDnnConvolutional1dOp) || (component[i].operation == kDnnCopyOp)
                || (component[i].operation == kDnnDeinterleaveOp) || (component[i].operation == kDnnInterleaveOp)
                || (component[i].operation == kDnnRecurrentOp)) {
                num_layers++;
            }
        }
        return (num_layers);
    }
    uint32_t num_group_in() {
        return ((component.size() > 0) ? ((component[0].orientation_in == kDnnInterleavedOrientation)
                                          ? component[0].num_columns_in : component[0].num_rows_in) : 0);
    }
    uint32_t num_group_out() {
        return ((component.size() > 0) ? ((component[component.size() - 1].orientation_out
            == kDnnInterleavedOrientation) ? component[component.size() - 1].num_columns_out : component[
                                              component.size() - 1].num_rows_out) : 0);
    }

    std::vector<intel_dnn_component_t> component;
    uint32_t num_left_context;
    uint32_t num_right_context;
    bool do_rotate_input;
    uint32_t num_rotate_rows = 0;
    uint32_t num_rotate_columns = 0;
    DnnSoftmaxType softmax_type;
    uint32_t *ptr_sumgroup_sizes;
    uint32_t num_sumgroup_sizes;
    float *ptr_priors;

    void WriteInputAndOutputText();
    static void WriteInputAndOutputTextGNA(intel_nnet_type_t * nnet);
    void BeginNewWrite();

 private:
    void *ptr_dnn_memory_;
    uint32_t num_bytes_dnn_memory_;
    uint32_t *ptr_active_outputs_;
    uint32_t num_active_outputs_;
    intel_dnn_number_type_t number_type_;
    float input_scale_factor_;

    static void InitCopyComponentPrivate(intel_dnn_component_t &cmp,
                                         intel_dnn_orientation_t orientation,
                                         uint32_t num_rows_in,
                                         uint32_t num_columns_in,
                                         uint32_t num_rows_out,
                                         uint32_t num_columns_out,
                                         uint32_t num_bytes_per_input,
                                         uint32_t num_bytes_per_output,
                                         float output_scale_factor,
                                         uint32_t num_copy_rows,
                                         uint32_t num_copy_columns,
                                         void *&ptr_inputs,
                                         void *&ptr_outputs,
                                         bool postInitMem);

    static void InitMaxpoolComponentPrivate(intel_dnn_component_t &cmp,
                                     uint32_t num_rows_in,
                                     uint32_t num_columns_in,
                                     uint32_t num_rows_out,
                                     uint32_t num_columns_out,
                                     uint32_t num_bytes_per_input,
                                     uint32_t num_bytes_per_output,
                                     uint32_t num_pool_size,
                                     uint32_t num_pool_step,
                                     uint32_t num_pool_stride,
                                     bool do_sum_not_max,
                                     float output_scale_factor,
                                     void *&ptr_inputs,
                                     void *&ptr_outputs,
                                     bool   postInitMem);

    static void InitPiecewiseLinearComponentPrivate(intel_dnn_component_t &cmp,
                                             DnnActivation function_id,
                                             intel_dnn_orientation_t orientation,
                                             uint32_t num_rows,
                                             uint32_t num_columns,
                                             uint32_t num_bytes_per_input,
                                             uint32_t num_bytes_per_output,
                                             uint32_t num_segments,
                                             float   output_scale_factor,
                                             void *& ptr_inputs,
                                             void *& ptr_outputs,
                                             intel_pwl_segment_t *ptr_segments,
                                             bool    postInitMem);

    static void InitConvolutional1DComponentPrivate(intel_dnn_component_t &comp,
                                             uint32_t num_rows_in,
                                             uint32_t num_columns_in,
                                             uint32_t num_rows_out,
                                             uint32_t num_columns_out,
                                             uint32_t num_bytes_per_input,
                                             uint32_t num_bytes_per_output,
                                             uint32_t num_bytes_per_weight,
                                             uint32_t num_bytes_per_bias,
                                             uint32_t num_filters,
                                             uint32_t num_filter_rows,
                                             uint32_t num_filter_coefficients,
                                             uint32_t num_feature_maps,
                                             uint32_t num_feature_map_rows,
                                             uint32_t num_feature_map_columns,
                                             float   weight_scale_factor,
                                             float   output_scale_factor,
                                             void *& ptr_inputs,
                                             void *& ptr_outputs,
                                             void *& ptr_filters,
                                             void *& ptr_biases,
                                             bool    postInitMem);

    static void InitAffineComponentPrivate(intel_dnn_component_t &comp,
                                           uint32_t num_rows_in,
                                           uint32_t num_columns,
                                           uint32_t num_rows_out,
                                           uint32_t num_bytes_per_input,
                                           uint32_t num_bytes_per_output,
                                           uint32_t num_bytes_per_weight,
                                           uint32_t num_bytes_per_bias,
                                           float  weight_scale_factor,
                                           float  output_scale_factor,
                                           void *&ptr_inputs,
                                           void *&ptr_outputs,
                                           void *&ptr_weights,
                                           void *&ptr_biases,
                                           bool   isDiag,
                                           bool   postInitMem);
};

void PlotFloatIntDnn(AmIntelDnn *dnn, AmIntelDnn *dnn_int);
bool isCompatibleDnn(AmIntelDnn dnn1, AmIntelDnn dnn2);
void ClearScoreError(intel_score_error_t *error);
void UpdateScoreError(intel_score_error_t *error, intel_score_error_t *total_error);
void SoftmaxGoogle(float *ptr_output, float *ptr_input, const uint32_t num_outputs, const uint32_t num_inputs);

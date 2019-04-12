// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// dnn.cpp : component based neural network class for ease of use
//
extern bool global_debug;

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <set>
#include <details/ie_exception.hpp>
#include <algorithm>
#include <gna-api-types-xnn.h>

#ifndef _NO_MKL_
#include <mkl_dnn.h>
#endif
#include "dnn.h"
#ifdef INTEGER_REF
#include "convnet.h"
#include "igemv16.h"
#include "igemv8.h"
#include "sgemm.h"
#else
#include "floatmath.h"
#endif
#include "pwl.h"
#include "util.h"
#include "gna_plugin_log.hpp"

#ifdef WIN32
# define rand_r(X) rand()
#endif

/**
 * whether to dump weights and biases
 */
#define DUMP_WB
/**
 * in light mode only layer names are dumped
 * @param filename
 * @param number_type
 * @return
 */
#define LIGHT_DUMP

static int & getDumpFolderId() {
    static int N = 0;
    return N;
}

static std::string getDumpFolderNameGNA() {
    return std::string("./gna_layers/")+std::to_string(getDumpFolderId() - 1)+"/";
}

static std::string getDumpFolderName() {
    return std::string("./layers/")+std::to_string(getDumpFolderId() - 1)+"/";
}

static std::string getRefFolderName() {
    return std::string("./ref_layers/")+std::to_string(getDumpFolderId() - 1)+"/";
}

void AmIntelDnn::BeginNewWrite() {
    getDumpFolderId()++;
}


void AmIntelDnn::Init(void *ptr_memory,
                      uint32_t num_memory_bytes,
                      intel_dnn_number_type_t number_type,
                      float scale_factor) {
    ptr_dnn_memory_ = ptr_memory;
    num_bytes_dnn_memory_ = num_memory_bytes;
    number_type_ = number_type;
    input_scale_factor_ = scale_factor;

    ptr_active_outputs_ = nullptr;
    num_active_outputs_ = 0;
    num_left_context = 0;
    num_right_context = 0;
    do_rotate_input = false;
    softmax_type = kSoftmaxNone;
    ptr_sumgroup_sizes = nullptr;
    num_sumgroup_sizes = 0;
    ptr_priors = nullptr;


    //  component.clear();
}

void AmIntelDnn::InitActiveList(uint32_t *ptr_active_list) {
    ptr_active_outputs_ = ptr_active_list;
    if (ptr_active_list == nullptr) {
        if (component[component.size() - 1].orientation_out == kDnnInterleavedOrientation) {
            num_active_outputs_ = component[component.size() - 1].num_rows_out;
        } else {
            num_active_outputs_ = component[component.size() - 1].num_columns_out;
        }
    } else {
        num_active_outputs_ = 0;
    }
}

void AmIntelDnn::AddComponents(uint32_t num_components_to_add) {
    component.resize(component.size() + num_components_to_add);
    for (uint32_t i = 0; i < num_components_to_add; i++) {
        ClearComponent(component.size() - i - 1);
    }
}

void AmIntelDnn::ClearComponent(uint32_t component_index) {
    if (component_index > component.size() - 1) {
        fprintf(stderr, "Error:  attempt to clear non-existent component!\n");
        throw -1;
    }
    component[component_index].num_rows_in = 0;
    component[component_index].num_columns_in = 0;
    component[component_index].num_rows_out = 0;
    component[component_index].num_columns_out = 0;
    component[component_index].num_bytes_per_input = 0;
    component[component_index].num_bytes_per_output = 0;
    component[component_index].operation = kDnnNullOp;
    component[component_index].macro_operation = kDnnMacroOpNone;
    component[component_index].orientation_in = kDnnUnknownOrientation;
    component[component_index].orientation_out = kDnnUnknownOrientation;
    component[component_index].ptr_inputs = nullptr;
    component[component_index].ptr_outputs = nullptr;
    memset(&component[component_index].op, 0, sizeof(component[component_index].op));
}

void AmIntelDnn::ClearState() {
    // To support recurrent networks, provide mechanism to clear persistent state
    // (e.g., between utterances for speech recognition).  For recurrent component,
    // this means clearing the feedback buffer.  For other components, just clear the
    // output buffer since any feedback will come from some component's output.
    for (uint32_t i = 0; i < component.size(); i++) {
        if (component[i].operation == kDnnRecurrentOp) {
            memset(component[i].op.recurrent.ptr_feedbacks,
                   0,
                   component[i].op.recurrent.num_vector_delay * component[i].num_columns_out
                       * component[i].num_bytes_per_input);
        } else {
            memset(component[i].ptr_outputs,
                   0,
                   component[i].num_bytes_per_output * component[i].num_rows_out * component[i].num_columns_out);
        }
    }
}

void AmIntelDnn::InitAffineComponentPrivate(intel_dnn_component_t &comp,
                                            uint32_t num_rows_in,
                                            uint32_t num_columns,
                                            uint32_t num_rows_out,
                                            uint32_t num_bytes_per_input,
                                            uint32_t num_bytes_per_output,
                                            uint32_t num_bytes_per_weight,
                                            uint32_t num_bytes_per_bias,
                                            float weight_scale_factor,
                                            float output_scale_factor,
                                            void *&ptr_inputs,
                                            void *&ptr_outputs,
                                            void *&ptr_weights,
                                            void *&ptr_biases,
                                            bool isDiag,
                                            bool postInitMem) {
    comp.num_rows_in = num_rows_in;
    comp.num_columns_in = num_columns;
    comp.num_rows_out = num_rows_out;
    comp.num_columns_out = num_columns;
    comp.num_bytes_per_input = num_bytes_per_input;
    comp.num_bytes_per_output = num_bytes_per_output;
    comp.operation = isDiag ? kDnnDiagonalOp : kDnnAffineOp;
    comp.macro_operation = kDnnMacroOpNone;
    comp.orientation_in = kDnnInterleavedOrientation;
    comp.orientation_out = kDnnInterleavedOrientation;
    comp.op.affine.num_bytes_per_weight = num_bytes_per_weight;
    comp.op.affine.num_bytes_per_bias = num_bytes_per_bias;
    comp.op.affine.weight_scale_factor = weight_scale_factor;
    comp.output_scale_factor = output_scale_factor;
    if (!postInitMem) {
        comp.op.affine.ptr_weights = ptr_weights;
        comp.op.affine.ptr_biases = ptr_biases;
        comp.ptr_inputs = ptr_inputs;
        comp.ptr_outputs = ptr_outputs;
    } else {
        ptr_weights = &comp.op.affine.ptr_weights;
        ptr_biases = &comp.op.affine.ptr_biases;
        ptr_inputs = &comp.ptr_inputs;
        ptr_outputs = &comp.ptr_outputs;
    }
}

void AmIntelDnn::InitDiagonalComponent(uint32_t component_index,
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
                                       void *ptr_biases) {
    component[component_index].num_rows_in = num_rows_in;
    component[component_index].num_columns_in = num_columns;
    component[component_index].num_rows_out = num_rows_out;
    component[component_index].num_columns_out = num_columns;
    component[component_index].num_bytes_per_input = num_bytes_per_input;
    component[component_index].num_bytes_per_output = num_bytes_per_output;
    component[component_index].operation = kDnnDiagonalOp;
    component[component_index].macro_operation = kDnnMacroOpNone;
    component[component_index].orientation_in = kDnnInterleavedOrientation;
    component[component_index].orientation_out = kDnnInterleavedOrientation;
    component[component_index].ptr_inputs = ptr_inputs;
    component[component_index].ptr_outputs = ptr_outputs;
    component[component_index].op.affine.num_bytes_per_weight = num_bytes_per_weight;
    component[component_index].op.affine.num_bytes_per_bias = num_bytes_per_bias;
    component[component_index].op.affine.weight_scale_factor = weight_scale_factor;
    component[component_index].output_scale_factor = output_scale_factor;
    component[component_index].op.affine.ptr_weights = ptr_weights;
    component[component_index].op.affine.ptr_biases = ptr_biases;
}

void AmIntelDnn::InitConvolutional1DComponentPrivate(intel_dnn_component_t &comp,
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
                                              void *&ptr_inputs,
                                              void *&ptr_outputs,
                                              void *&ptr_filters,
                                              void *&ptr_biases,
                                              bool postInitMem) {
    comp.num_rows_in = num_rows_in;
    comp.num_columns_in = num_columns_in;
    comp.num_rows_out = num_rows_out;
    comp.num_columns_out = num_columns_out;
    comp.num_bytes_per_input = num_bytes_per_input;
    comp.num_bytes_per_output = num_bytes_per_output;
    comp.operation = kDnnConvolutional1dOp;
    comp.macro_operation = kDnnMacroOpNone;
    comp.orientation_in = kDnnNonInterleavedOrientation;
    comp.orientation_out = kDnnNonInterleavedOrientation;
    comp.ptr_inputs = ptr_inputs;
    comp.ptr_outputs = ptr_outputs;
    comp.op.conv1D.num_bytes_per_weight = num_bytes_per_weight;
    comp.op.conv1D.num_bytes_per_bias = num_bytes_per_bias;
    comp.op.conv1D.num_filters = num_filters;
    comp.op.conv1D.num_filter_rows = num_filter_rows;
    comp.op.conv1D.num_filter_coefficients = num_filter_coefficients;
    comp.op.conv1D.num_feature_maps = num_feature_maps;
    comp.op.conv1D.num_feature_map_rows = num_feature_map_rows;
    comp.op.conv1D.num_feature_map_columns = num_feature_map_columns;
    comp.op.conv1D.weight_scale_factor = weight_scale_factor;
    comp.output_scale_factor = output_scale_factor;

    if (!postInitMem) {
        comp.op.conv1D.ptr_filters = ptr_filters;
        comp.op.conv1D.ptr_biases  = ptr_biases;
        comp.ptr_inputs = ptr_inputs;
        comp.ptr_outputs = ptr_outputs;
    } else {
        ptr_filters = &comp.op.conv1D.ptr_filters;
        ptr_biases  = &comp.op.conv1D.ptr_biases;
        ptr_inputs  = &comp.ptr_inputs;
        ptr_outputs = &comp.ptr_outputs;
    }
}

void AmIntelDnn::InitMaxpoolComponentPrivate(intel_dnn_component_t &comp,
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
                                      bool postInitMem) {
    comp.num_rows_in = num_rows_in;
    comp.num_columns_in = num_columns_in;
    comp.num_rows_out = num_rows_out;
    comp.num_columns_out = num_columns_out;
    comp.num_bytes_per_input = num_bytes_per_input;
    comp.num_bytes_per_output = num_bytes_per_output;
    comp.operation = kDnnMaxPoolOp;
    comp.macro_operation = kDnnMacroOpNone;
    comp.orientation_in = kDnnNonInterleavedOrientation;
    comp.orientation_out = kDnnNonInterleavedOrientation;
    comp.op.maxpool.num_inputs = num_pool_size;
    comp.op.maxpool.num_inputs_step = num_pool_step;
    comp.op.maxpool.num_inputs_stride = num_pool_stride;
    comp.op.maxpool.do_sum_not_max = do_sum_not_max;
    comp.output_scale_factor = output_scale_factor;

    if (!postInitMem) {
        comp.ptr_inputs = ptr_inputs;
        comp.ptr_outputs = ptr_outputs;
    } else {
        ptr_inputs  = &comp.ptr_inputs;
        ptr_outputs = &comp.ptr_outputs;
    }
}

void AmIntelDnn::InitCopyComponentPrivate(intel_dnn_component_t &comp,
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
                                          bool postInitMem) {
    comp.num_rows_in = num_rows_in;
    comp.num_columns_in = num_columns_in;
    comp.num_rows_out = num_rows_out;
    comp.num_columns_out = num_columns_out;
    comp.num_bytes_per_input = num_bytes_per_input;
    comp.num_bytes_per_output = num_bytes_per_output;
    comp.operation = kDnnCopyOp;
    comp.macro_operation = kDnnMacroOpNone;
    comp.orientation_in = orientation;
    comp.orientation_out = orientation;
    comp.ptr_inputs = ptr_inputs;
    comp.ptr_outputs = ptr_outputs;
    comp.output_scale_factor = output_scale_factor;
    comp.op.copy.num_copy_rows = num_copy_rows;
    comp.op.copy.num_copy_columns = num_copy_columns;

    if (!postInitMem) {
        comp.ptr_inputs = ptr_inputs;
        comp.ptr_outputs = ptr_outputs;
    } else {
        ptr_inputs  = &comp.ptr_inputs;
        ptr_outputs = &comp.ptr_outputs;
    }
}

void AmIntelDnn::InitPiecewiseLinearComponentPrivate(intel_dnn_component_t &comp,
                                                     DnnActivation function_id,
                                                     intel_dnn_orientation_t orientation,
                                                     uint32_t num_rows,
                                                     uint32_t num_columns,
                                                     uint32_t num_bytes_per_input,
                                                     uint32_t num_bytes_per_output,
                                                     uint32_t num_segments,
                                                     float output_scale_factor,
                                                     void *&ptr_inputs,
                                                     void *&ptr_outputs,
                                                     intel_pwl_segment_t *ptr_segments,
                                                     bool postInitMem) {
    comp.num_rows_in = num_rows;
    comp.num_columns_in = num_columns;
    comp.num_rows_out = num_rows;
    comp.num_columns_out = num_columns;
    comp.num_bytes_per_input = num_bytes_per_input;
    comp.num_bytes_per_output = num_bytes_per_output;
    comp.operation = kDnnPiecewiselinearOp;
    comp.macro_operation = kDnnMacroOpNone;
    comp.orientation_in = orientation;
    comp.orientation_out = orientation;
    comp.op.pwl.func_id = function_id;
    comp.op.pwl.num_segments = num_segments;
    comp.output_scale_factor = output_scale_factor;

    if (!postInitMem) {
        comp.ptr_inputs = ptr_inputs;
        comp.ptr_outputs = ptr_outputs;
        comp.op.pwl.ptr_segments = ptr_segments;
    } else {
        ptr_inputs = &comp.ptr_inputs;
        ptr_outputs = &comp.ptr_outputs;
        if (ptr_segments != nullptr) {
            *reinterpret_cast<intel_pwl_segment_t **>(ptr_segments) =
                reinterpret_cast<intel_pwl_segment_t *>(& comp.op.pwl.ptr_segments);
        }
    }
}

void AmIntelDnn::InitRecurrentComponent(uint32_t component_index,
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
                                        void *ptr_biases) {
    component[component_index].num_rows_in = num_rows;
    component[component_index].num_columns_in = num_columns_in;
    component[component_index].num_rows_out = num_rows;
    component[component_index].num_columns_out = num_columns_out;
    component[component_index].num_bytes_per_input = num_bytes_per_input;
    component[component_index].num_bytes_per_output = num_bytes_per_output;
    component[component_index].operation = kDnnRecurrentOp;
    component[component_index].macro_operation = kDnnMacroOpNone;
    component[component_index].orientation_in = kDnnNonInterleavedOrientation;
    component[component_index].orientation_out = kDnnNonInterleavedOrientation;
    component[component_index].ptr_inputs = ptr_inputs;
    component[component_index].ptr_outputs = ptr_outputs;
    component[component_index].op.recurrent.num_vector_delay = num_vector_delay;
    component[component_index].op.recurrent.num_bytes_per_weight = num_bytes_per_weight;
    component[component_index].op.recurrent.num_bytes_per_bias = num_bytes_per_bias;
    component[component_index].op.recurrent.weight_scale_factor = weight_scale_factor;
    component[component_index].output_scale_factor = output_scale_factor;
    component[component_index].op.recurrent.ptr_feedbacks = ptr_feedbacks;
    component[component_index].op.recurrent.ptr_weights = ptr_weights;
    component[component_index].op.recurrent.ptr_biases = ptr_biases;
}

void AmIntelDnn::InitInterleaveComponent(uint32_t component_index, uint32_t num_rows, uint32_t num_columns,
                                         uint32_t num_bytes_per_input, uint32_t num_bytes_per_output,
                                         float output_scale_factor, void *ptr_inputs, void *ptr_outputs) {
    component[component_index].num_rows_in = num_rows;
    component[component_index].num_columns_in = num_columns;
    component[component_index].num_rows_out = num_columns;
    component[component_index].num_columns_out = num_rows;
    component[component_index].num_bytes_per_input = num_bytes_per_input;
    component[component_index].num_bytes_per_output = num_bytes_per_output;
    component[component_index].operation = kDnnInterleaveOp;
    component[component_index].macro_operation = kDnnMacroOpNone;
    component[component_index].orientation_in = kDnnNonInterleavedOrientation;
    component[component_index].orientation_out = kDnnInterleavedOrientation;
    component[component_index].ptr_inputs = ptr_inputs;
    component[component_index].ptr_outputs = ptr_outputs;
    component[component_index].output_scale_factor = output_scale_factor;
}

void AmIntelDnn::InitDeinterleaveComponent(uint32_t component_index, uint32_t num_rows, uint32_t num_columns,
                                           uint32_t num_bytes_per_input, uint32_t num_bytes_per_output,
                                           float output_scale_factor, void *ptr_inputs, void *ptr_outputs) {
    component[component_index].num_rows_in = num_rows;
    component[component_index].num_columns_in = num_columns;
    component[component_index].num_rows_out = num_columns;
    component[component_index].num_columns_out = num_rows;
    component[component_index].num_bytes_per_input = num_bytes_per_input;
    component[component_index].num_bytes_per_output = num_bytes_per_output;
    component[component_index].operation = kDnnDeinterleaveOp;
    component[component_index].macro_operation = kDnnMacroOpNone;
    component[component_index].orientation_in = kDnnInterleavedOrientation;
    component[component_index].orientation_out = kDnnNonInterleavedOrientation;
    component[component_index].ptr_inputs = ptr_inputs;
    component[component_index].ptr_outputs = ptr_outputs;
    component[component_index].output_scale_factor = output_scale_factor;
}

__inline void ApplyAffineTransform(intel_dnn_component_t *component, uint32_t *list, uint32_t listsize) {
    auto transform = &component->op.affine;
    int m = component->num_rows_out;
    int n = component->num_columns_in;
    int k = component->num_rows_in;
    int lda = component->num_rows_in;
    int ldb = component->num_columns_in;
    int ldc = component->num_columns_out;

    switch (component->num_bytes_per_input) {
#ifdef INTEGER_REF
        case 2:
            if (component->op.affine.num_bytes_per_weight == 1) {
                int8_t *A = reinterpret_cast<int8_t*>(transform->ptr_weights);
                int16_t *B = reinterpret_cast<int16_t*>(component->ptr_inputs);
                int32_t *C = reinterpret_cast<int32_t*>(component->ptr_outputs);
                intel_compound_bias_t *bias = reinterpret_cast<intel_compound_bias_t*>(transform->ptr_biases);
                if (list == nullptr) {
                    //  PrintMatrixInt8("W int8", W, k, m, ldw, component->op.affine.weight_scale_factor);
                    //  PrintMatrixInt16("X int16", X, k, n, ldx, component->op.affine.weight_scale_factor);
                    //  PrintMatrixInt32("Y int32", Y, m, n, ldy, component->output_scale_factor);
                    igemm8_gna(m, n, k, A, lda, B, ldb, bias, C, ldc);
                } else {
                    //  PrintMatrixInt8("W int8", W, k, m, ldw, component->op.affine.weight_scale_factor);
                    //  PrintMatrixInt16("X int16", X, k, n, ldx, component->op.affine.weight_scale_factor);
                    //  PrintMatrixInt32("Y int32", Y, m, n, ldy, component->output_scale_factor);
                    igemm8_gna_subset(m, n, k, A, lda, B, ldb, bias, C, ldc, list, listsize);
                }
                //  PrintMatrixInt32("C int32", C, m, n, ldc, component->output_scale_factor);
            } else if (component->op.affine.num_bytes_per_weight == 2) {
                int16_t *A = reinterpret_cast<int16_t*>(transform->ptr_weights);
                int16_t *B = reinterpret_cast<int16_t*>(component->ptr_inputs);
                int32_t *C = reinterpret_cast<int32_t*>(component->ptr_outputs);
                int32_t *bias = reinterpret_cast<int32_t*>(transform->ptr_biases);
                if (list == nullptr) {
                    for (uint32_t i = 0; i < m; i++) {
                        for (uint32_t j = 0; j < n; j++) {
                            C[i*ldc+j] = bias[i];
                        }
                    }
                    //  PrintMatrixInt16("A int16", A, k, m, lda, component->op.affine.weight_scale_factor);
                    //  PrintMatrixInt16("trans(B) int16", B, k, n, ldb, component->op.affine.weight_scale_factor);
                    //  PrintMatrixInt32("C int32", C, m, n, ldc, component->output_scale_factor);
                    cblas_igemm16(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, lda, B, ldb, 1.0, C, ldc);
                } else {
                    for (int l = 0; l < listsize; l++) {
                        int i = list[l];
                        for (uint32_t j = 0; j < n; j++) {
                            C[l*ldc+j] = bias[i];
                        }
                    }
                    //  PrintMatrixInt16("A int16", A, k, m, lda, component->op.affine.scale_factor);
                    //  PrintMatrixInt16("trans(B) int16", B, k, n, ldb, component->op.affine.scale_factor);
                    //  PrintMatrixInt32("C int32", C, m, n, ldc, component->op.affine.scale_factor * component->op.affine.scale_factor);
                    cblas_igemm16_subset(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, lda, B, ldb, 1.0, C, ldc, list, listsize);
                }
                //  PrintMatrixInt32("C int32", C, m, n, ldc, component->output_scale_factor);
            } else {
                fprintf(stderr, "Bad weight width in ApplyAffineTransform!\n");
                throw -1;
            }
            break;
#endif  // #ifdef INTEGER_REF
        case 4: {
            auto A = reinterpret_cast<float *>(transform->ptr_weights);
            auto B = reinterpret_cast<float *>(component->ptr_inputs);
            auto C = reinterpret_cast<float *>(component->ptr_outputs);
            auto bias = reinterpret_cast<float *>(transform->ptr_biases);
            if (list == nullptr) {
                for (uint32_t i = 0; i < m; i++) {
                    for (uint32_t j = 0; j < n; j++) {
                        C[i * ldc + j] = bias[i];
                    }
                }
                //  if (global_debug) PrintMatrixFloat32("A float", A, m, k, lda);
                //  if (global_debug) PrintMatrixFloat32("B float", B, k, n, ldb);
                //  if (global_debug) PrintMatrixFloat32("C float before", C, m, n, ldc);
                cblas_sgemm1(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, lda, B, ldb, 1.0, C, ldc);
                //  if (global_debug) PrintMatrixFloat32("C float after", C, m, n, ldc);
            } else {
                for (int l = 0; l < listsize; l++) {
                    int i = list[l];
                    for (uint32_t j = 0; j < n; j++) {
                        C[l * ldc + j] = bias[i];
                    }
                }
                //  PrintMatrixFloat32("A float", A, k, m, lda);
                //  PrintMatrixFloat32("trans(B) float", B, k, n, ldb);
                //  PrintMatrixFloat32("C float before", C, listsize, n, ldc);
                cblas_sgemm_subset(CblasRowMajor,
                                   CblasNoTrans,
                                   CblasNoTrans,
                                   m,
                                   n,
                                   k,
                                   1.0,
                                   A,
                                   lda,
                                   B,
                                   ldb,
                                   1.0,
                                   C,
                                   ldc,
                                   list,
                                   listsize);
                //  PrintMatrixFloat32("C float after", C, listsize, n, ldc);
            }
        }
            break;
        default:fprintf(stderr, "Bad data width in ApplyAffineTransform!\n");
            throw -1;
    }
}

__inline void ApplyDiagonalTransform(intel_dnn_component_t *component) {
    auto transform = &component->op.affine;
    int m = component->num_rows_out;
    int n = component->num_columns_in;
    int ldb = component->num_columns_in;
    int ldc = component->num_columns_out;

    switch (component->num_bytes_per_input) {
#ifdef INTEGER_REF
        case 2:
            if (component->op.affine.num_bytes_per_weight == 1) {
                int8_t *A = reinterpret_cast<int8_t*>(transform->ptr_weights);
                int16_t *B = reinterpret_cast<int16_t*>(component->ptr_inputs);
                int32_t *C = reinterpret_cast<int32_t*>(component->ptr_outputs);
                intel_compound_bias_t *bias = reinterpret_cast<intel_compound_bias_t*>(transform->ptr_biases);
                //  PrintMatrixInt8("W int8", W, k, m, ldw, component->op.affine.weight_scale_factor);
                //  PrintMatrixInt16("X int16", X, k, n, ldx, component->op.affine.weight_scale_factor);
                //  PrintMatrixInt32("Y int32", Y, m, n, ldy, component->output_scale_factor);
                isbmm8_gna(m, n, A, lda, B, ldb, bias, C, ldc);
                //  PrintMatrixInt32("C int32", C, m, n, ldc, component->output_scale_factor);
            } else if (component->op.affine.num_bytes_per_weight == 2) {
                int16_t *A = reinterpret_cast<int16_t*>(transform->ptr_weights);
                int16_t *B = reinterpret_cast<int16_t*>(component->ptr_inputs);
                int32_t *C = reinterpret_cast<int32_t*>(component->ptr_outputs);
                int32_t *bias = reinterpret_cast<int32_t*>(transform->ptr_biases);
                for (uint32_t i = 0; i < m; i++) {
                    for (uint32_t j = 0; j < n; j++) {
                        C[i*ldc+j] = bias[i];
                    }
                }
                //  PrintMatrixInt16("A int16", A, 1, m, lda, component->op.affine.weight_scale_factor);
                //  PrintMatrixInt16("trans(B) int16", B, k, n, ldb, component->op.affine.weight_scale_factor);
                //  PrintMatrixInt32("C int32", C, m, n, ldc, component->output_scale_factor);
                cblas_isbmm16(m, n, A, lda, B, ldb, C, ldc);
                //  PrintMatrixInt32("C int32", C, m, n, ldc, component->output_scale_factor);
            } else {
                fprintf(stderr, "Bad weight width in ApplyDiagonalTransform!\n");
                throw -1;
            }
            break;
#endif  // #ifdef INTEGER_REF
        case 4: {
            auto A = reinterpret_cast<float *>(transform->ptr_weights);
            auto B = reinterpret_cast<float *>(component->ptr_inputs);
            auto C = reinterpret_cast<float *>(component->ptr_outputs);
            auto bias = reinterpret_cast<float *>(transform->ptr_biases);
            for (uint32_t i = 0; i < m; i++) {
                for (uint32_t j = 0; j < n; j++) {
                    C[i * ldc + j] = bias[i];
                }
            }
            //  PrintMatrixFloat32("A float", A, 1, m, lda);
            //  PrintMatrixFloat32("B float", B, k, n, ldb);
            //  PrintMatrixFloat32("C float before", C, m, n, ldc);
            for (uint32_t j = 0; j < n; j++) {
                float *Bcol = B + j * ldb;
                float *Ccol = C + j * ldc;
                cblas_ssbmv1(CblasRowMajor, CblasLower, m, 0, 1.0, A, 1, Bcol, 1, 1.0, Ccol, 1);
            }
            //  PrintMatrixFloat32("C float after", C, m, n, ldc);
        }
            break;
        default:fprintf(stderr, "Bad data width in ApplyDiagonalTransform!\n");
            throw -1;
    }
}

__inline void ApplyRecurrentTransform(intel_dnn_component_t *component, uint32_t row, void *ptr_feedbacks) {
    intel_recurrent_t *transform = &component->op.recurrent;
    int k1 = component->num_columns_in;
    int k2 = component->num_columns_out;
    int n = k2;

    if (component->op.recurrent.ptr_feedbacks == nullptr) {
        fprintf(stderr, "nullptr feedback pointer in ApplyRecurrentTransform()!\n");
        throw -1;
    }

    switch (component->num_bytes_per_input) {
#ifdef INTEGER_REF
        case 2:
            if (component->op.recurrent.num_bytes_per_weight == 1) {
                int16_t *A1 = reinterpret_cast<int16_t*>(component->ptr_inputs) + row * component->num_columns_in;
                int16_t *A2 = reinterpret_cast<int16_t*>(ptr_feedbacks);
                int8_t *X = reinterpret_cast<int8_t*>(transform->ptr_weights);
                intel_compound_bias_t *B = reinterpret_cast<intel_compound_bias_t*>(transform->ptr_biases);
                int32_t *C = reinterpret_cast<int32_t*>(component->ptr_outputs) + row * component->num_columns_out;
                //  PrintMatrixInt16("A1 int", A1, 1, k1, k1, component->op.recurrent.weight_scale_factor);
                //  PrintMatrixInt16("A2 int", A2, 1, k2, k2);
                //  PrintMatrixInt8("X int", X, k, n, n, component->op.recurrent.weight_scale_factor);
                //  PrintMatrixInt32("B int", B, 1, 2*n, 2*n, component->output_scale_factor);
                igemv8_gna_split(n, k1, k2, A1, A2, X, B, C);
                //  PrintMatrixInt32("C int", C, 1, n, n, component->output_scale_factor);
            } else if (component->op.recurrent.num_bytes_per_weight == 2) {
                int16_t *A1 = reinterpret_cast<int16_t*>(component->ptr_inputs) + row * component->num_columns_in;
                int16_t *A2 = reinterpret_cast<int16_t*>(ptr_feedbacks);
                int16_t *X = reinterpret_cast<int16_t*>(transform->ptr_weights);
                int32_t *B = reinterpret_cast<int32_t*>(transform->ptr_biases);
                int32_t *C = reinterpret_cast<int32_t*>(component->ptr_outputs) + row * component->num_columns_out;
                //  PrintMatrixInt16("A1 int", A1, 1, k1, k1, component->op.recurrent.weight_scale_factor);
                //  PrintMatrixInt16("A2 int", A2, 1, k2, k2, component->op.recurrent.weight_scale_factor);
                //  PrintMatrixInt16("X int", X, k, n, n, component->op.recurrent.weight_scale_factor);
                //  PrintMatrixInt32("B int", B, 1, n, n, component->output_scale_factor);
                igemv16_split(n, k1, k2, A1, A2, X, B, C);
                //  PrintMatrixInt32("C int", C, 1, n, n, component->output_scale_factor);
            } else {
                fprintf(stderr, "Weight width not supported in ApplyRecurrentTransform!\n");
                throw -1;
            }
            break;
#endif  // #ifdef INTEGER_REF
        case 4: {
            auto A1 = reinterpret_cast<float *>(component->ptr_inputs) + row * component->num_columns_in;
            auto A2 = reinterpret_cast<float *>(ptr_feedbacks);
            auto X = reinterpret_cast<float *>(transform->ptr_weights);
            auto B = reinterpret_cast<float *>(transform->ptr_biases);
            auto C = reinterpret_cast<float *>(component->ptr_outputs) + row * component->num_columns_out;
            //  PrintMatrixFloat32("A1 float", A1, 1, k1, k1);
            //  PrintMatrixFloat32("A2 float", A2, 1, k2, k2);
            //  PrintMatrixFloat32("X float", X, k, n, n);
            //  PrintMatrixFloat32("B float", B, 1, n, n);
            sgemv_split(n, k1, k2, A1, A2, X, B, C);
            //  PrintMatrixFloat32("C float", C, 1, n, n);
        }
            break;
        default:fprintf(stderr, "Bad data width in ApplyRecurrentTransform!\n");
            throw -1;
    }
}

__inline void ApplyConvolutional1DTransform(intel_dnn_component_t *component) {
    switch (component->num_bytes_per_input) {
#ifdef INTEGER_REF
        case 2:
            CNNFilter16(component);
            break;
#endif  // #ifdef INTEGER_REF
        case 4:
            //  PrintMatrixFloat32("Input float", reinterpret_cast<float*>(component->ptr_inputs),
            //  component->num_rows_in, component->num_columns_in, component->num_columns_in);
            //  PrintMatrixFloat32("Filt float", reinterpret_cast<float*>(component->op.conv1D.ptr_filters),
            //  component->op.conv1D.num_filters,
            //  component->op.conv1D.num_filter_rows*component->op.conv1D.num_feature_map_columns*component->op.conv1D.num_feature_maps,
            //  component->op.conv1D.num_filter_rows*component->op.conv1D.num_feature_map_columns*component->op.conv1D.num_feature_maps);
            //  PrintMatrixFloat32("Bias float", reinterpret_cast<float*>(component->op.conv1D.ptr_biases), 1,
            // component->op.conv1D.num_filters, component->op.conv1D.num_filters);
            CNNFilter32(component);
            //  PrintMatrixFloat32("Output float", reinterpret_cast<float*>(component->ptr_outputs, component->num_rows_out,
            // component->num_columns_out, component->num_columns_out);
            break;
        default:fprintf(stderr, "Bad data width in ApplyConvolutionalTransform!\n");
            throw -1;
    }
}

__inline void ApplyPiecewiseLinearTransform(intel_dnn_component_t *component,
                                            intel_dnn_number_type_t number_type,
                                            uint32_t listsize) {
    if (number_type == kDnnFloat) {
        // PrintMatrixFloat32("PWL Input float", reinterpret_cast<float*>(component->ptr_inputs), component->num_rows_in,
        // component->num_columns_in, component->num_columns_in);
        PwlApply32(component, listsize);
        // PrintMatrixFloat32("PWL Output float", reinterpret_cast<float*>(component->ptr_outputs), component->num_rows_out,
        // component->num_columns_out, component->num_columns_out);
#ifdef INTEGER_REF
        } else if (component->num_bytes_per_output == 2) {
            PwlApply16(component, listsize);
#endif  // #ifdef INTEGER_REF
    } else {
        fprintf(stderr, "Bad data width in ApplyPiecewiseLinearTransform!\n");
        throw -1;
    }
}

__inline void ApplyPiecewiseLinearTransform(intel_dnn_component_t *component,
                                            intel_dnn_number_type_t number_type,
                                            uint32_t listsize,
                                            uint32_t num_row) {
    if (number_type == kDnnFloat) {
        PwlApply32(component, num_row, num_row, 0, listsize - 1);
#ifdef INTEGER_REF
        } else if (component->num_bytes_per_output == 2) {
            PwlApply16(component, num_row, num_row, 0, listsize-1);
#endif  // #ifdef INTEGER_REF
    } else {
        fprintf(stderr, "Bad data width in ApplyPiecewiseLinearTransform!\n");
        throw -1;
    }
}

__inline void ApplyMaxPoolTransform(intel_dnn_component_t *component, intel_dnn_number_type_t number_type) {
    if (component->num_bytes_per_input == 4) {
        // PrintMatrixFloat32("Input float", reinterpret_cast<float*>(component->ptr_inputs), component->num_rows_in,
        // component->num_columns_in, component->num_columns_in);
        CNNMaxPool(component, number_type);
        // PrintMatrixFloat32("Output float", reinterpret_cast<float*>(component->ptr_outputs), component->num_rows_out,
        // component->num_columns_out, component->num_columns_out);
    } else {
        fprintf(stderr, "Bad data width in ApplyMaxPoolTransform!\n");
        throw -1;
    }
}

__inline void ApplyTranspose(intel_dnn_component_t *component) {
    int m = component->num_rows_in;
    int n = component->num_columns_in;
    int lda = component->num_columns_in;
    int ldb = component->num_columns_out;
    // B = Transpose(A) where A is mxn and B is nxm
    switch (component->num_bytes_per_input) {
#ifdef INTEGER_REF
        case 1:
            {
                int8_t *A = reinterpret_cast<int8_t*>(component->ptr_inputs);
                int8_t *B = reinterpret_cast<int8_t*>(component->ptr_outputs);
                for (uint32_t row = 0; row < m; row++) {
                    for (uint32_t col = 0; col < n; col++) {
                        B[col*ldb+row] = A[row*lda+col];
                    }
                }
            }
            break;
        case 2:
            {
                int16_t *A = reinterpret_cast<int16_t*>(component->ptr_inputs);
                int16_t *B = reinterpret_cast<int16_t*>(component->ptr_outputs);
                for (uint32_t row = 0; row < m; row++) {
                    for (uint32_t col = 0; col < n; col++) {
                        B[col*ldb+row] = A[row*lda+col];
                    }
                }
            }
            break;
#endif  // #ifdef INTEGER_REF
        case 4: {
            auto A = reinterpret_cast<float *>(component->ptr_inputs);
            auto B = reinterpret_cast<float *>(component->ptr_outputs);
            for (uint32_t row = 0; row < m; row++) {
                for (uint32_t col = 0; col < n; col++) {
                    B[col * ldb + row] = A[row * lda + col];
                }
            }
        }
            break;
        default:fprintf(stderr, "Bad data width in ApplyInterleave!\n");
            throw -1;
    }
}

__inline void ApplyCopy(intel_dnn_component_t *component) {
    auto src = reinterpret_cast<uint8_t *>(component->ptr_inputs);
    auto dst = reinterpret_cast<uint8_t *>(component->ptr_outputs);
    int32_t m = component->op.copy.num_copy_rows;
    int32_t n = component->op.copy.num_copy_columns;
    int32_t lda = component->num_columns_in;
    int32_t ldb = component->num_columns_out;
    if (m > component->num_rows_in) {
        fprintf(stderr, "Error:  attempt to copy more columns than matrix has!\n");
        throw -1;
    } else {
        switch (component->num_bytes_per_input) {
#ifdef INTEGER_REF
            case 2:
                {
                    int16_t *A = reinterpret_cast<int16_t*>(src);
                    int16_t *B = reinterpret_cast<int16_t*>(dst);
                    for (uint32_t row = 0; row < m; row++) {
                        for (uint32_t col = 0; col < n; col++) {
                            B[row*ldb + col] = A[row*lda + col];
                        }
                    }
                }
                break;
#endif  // #ifdef INTEGER_REF
            case 4: {
                auto A = reinterpret_cast<float *>(src);
                auto B = reinterpret_cast<float *>(dst);
                for (uint32_t row = 0; row < m; row++) {
                    for (uint32_t col = 0; col < n; col++) {
                        B[row * ldb + col] = A[row * lda + col];
                    }
                }
            }
                break;
            default:fprintf(stderr, "Bad data width in ApplyCopy!\n");
                throw -1;
        }
    }
}

uint32_t AmIntelDnn::CopyActiveList(std::vector<std::vector<uint32_t> > &active_list, uint32_t list_index) {
    if (component[component.size() - 1].orientation_out == kDnnInterleavedOrientation) {
        num_active_outputs_ = component[component.size() - 1].num_rows_out;
    } else {
        num_active_outputs_ = component[component.size() - 1].num_columns_out;
    }

    if (!active_list.empty()) {
        if (list_index >= active_list.size()) {
            fprintf(stderr, "Index %d beyond end of active list in CopyActiveList()\n", list_index);
            throw -1;
        }
        if (active_list[list_index].size() > component[component.size() - 1].num_rows_out) {
            fprintf(stderr, "Active list too large in CopyActiveList()\n");
            throw -1;
        }

        if (ptr_active_outputs_ != nullptr) {
            num_active_outputs_ = active_list[list_index].size();
            memcpy(ptr_active_outputs_, active_list[list_index].data(), num_active_outputs_ * sizeof(uint32_t));
        }
    }

    return (num_active_outputs_);
}

void AmIntelDnn::Propagate() {
    for (uint32_t i = 0; i < component.size(); i++) {
        intel_dnn_component_t *comp = &component[i];
        uint32_t *ptr_active_outputs = nullptr;
        uint32_t num_active_outputs = (comp->orientation_out == kDnnInterleavedOrientation)
                                      ? comp->num_rows_out : comp->num_columns_out;

        if (i == component.size() - 1) {  // active list applies to last component
            ptr_active_outputs = ptr_active_outputs_;
            num_active_outputs = num_active_outputs_;
        } else if (i == component.size() - 2) {  // also applies to last two components when last is PWL
            if ((component[i].operation == kDnnAffineOp) && (component[i + 1].operation == kDnnPiecewiselinearOp)) {
                ptr_active_outputs = ptr_active_outputs_;
                num_active_outputs = num_active_outputs_;
            }
        }

        switch (comp->operation) {
            case kDnnAffineOp :ApplyAffineTransform(comp, ptr_active_outputs, num_active_outputs);
                break;
            case kDnnDiagonalOp:ApplyDiagonalTransform(comp);
                break;
            case kDnnRecurrentOp:
                if ((i < component.size() - 1) && (component[i + 1].operation == kDnnPiecewiselinearOp)) {
                    intel_dnn_component_t *comp_pwl = &component[i + 1];
                    for (uint32_t j = 0; j < comp->num_rows_in; j++) {
                        void *ptr_feedbacks =
                            reinterpret_cast<void *>(reinterpret_cast<int32_t *>(comp->op.recurrent.ptr_feedbacks) + j * comp_pwl->num_columns_out);
                        ApplyRecurrentTransform(comp, j, ptr_feedbacks);
                        //  PrintOutputs(i);
                        ApplyPiecewiseLinearTransform(comp_pwl, number_type_, num_active_outputs, j);
                    }
                    i++;  // skip next component
                } else {
                    fprintf(stderr, "Missing PiecewiseLinear component after Recurrent component in Propagate!\n");
                    throw -1;
                }
                break;
            case kDnnConvolutional1dOp:ApplyConvolutional1DTransform(comp);
                break;
            case kDnnPiecewiselinearOp:ApplyPiecewiseLinearTransform(comp, number_type_, num_active_outputs);
                break;
            case kDnnMaxPoolOp:ApplyMaxPoolTransform(comp, number_type_);
                break;
            case kDnnInterleaveOp:ApplyTranspose(comp);
                break;
            case kDnnDeinterleaveOp:ApplyTranspose(comp);
                break;
            case kDnnCopyOp:ApplyCopy(comp);
                break;
            default:fprintf(stderr, "Bad operation in Propagate!\n");
                throw -1;
                break;
        }
        //  PrintOutputs(i); fflush(stdout);
    }
}

intel_dnn_macro_operation_t AmIntelDnn::MacroOperation(uint32_t component_index) {
    return (component[component_index].macro_operation);
}

void AmIntelDnn::SetMacroOperation(uint32_t component_index, intel_dnn_macro_operation_t macro_operation) {
    component[component_index].macro_operation = macro_operation;
}

float AmIntelDnn::InputScaleFactor(uint32_t component_index) {
    float scale_factor = 1.0;

    if (component_index == 0) {
        scale_factor = input_scale_factor_;
    } else {
        if (component[component_index - 1].operation == kDnnAffineOp) {
            scale_factor = component[component_index - 1].output_scale_factor;
        } else if (component[component_index - 1].operation == kDnnDiagonalOp) {
            scale_factor = component[component_index - 1].output_scale_factor;
        } else if (component[component_index - 1].operation == kDnnConvolutional1dOp) {
            scale_factor = component[component_index - 1].output_scale_factor;
        } else if (component[component_index - 1].operation == kDnnRecurrentOp) {
            scale_factor = component[component_index - 1].output_scale_factor;
        } else if (component[component_index - 1].operation == kDnnInterleaveOp) {
            scale_factor = component[component_index - 1].output_scale_factor;
        } else if (component[component_index - 1].operation == kDnnDeinterleaveOp) {
            scale_factor = component[component_index - 1].output_scale_factor;
        } else if (component[component_index - 1].operation == kDnnCopyOp) {
            scale_factor = component[component_index - 1].output_scale_factor;
        }
    }

    return (scale_factor);
}

float AmIntelDnn::WeightScaleFactor(uint32_t component_index) {
    float scale_factor = 1.0;

    if (component[component_index].operation == kDnnAffineOp) {
        scale_factor = component[component_index].op.affine.weight_scale_factor;
    } else if (component[component_index].operation == kDnnDiagonalOp) {
        scale_factor = component[component_index].op.affine.weight_scale_factor;
    } else if (component[component_index].operation == kDnnConvolutional1dOp) {
        scale_factor = component[component_index].op.conv1D.weight_scale_factor;
    } else if (component[component_index].operation == kDnnRecurrentOp) {
        scale_factor = component[component_index].op.recurrent.weight_scale_factor;
    }

    return (scale_factor);
}

float AmIntelDnn::OutputScaleFactor(intel_dnn_component_t &comp) {
    return comp.output_scale_factor;
}

void AmIntelDnn::SetOutputScaleFactor(uint32_t component_index, float scale_factor) {
    component[component_index].output_scale_factor = scale_factor;
}

void AmIntelDnn::PrintOutputs(uint32_t component_index) {
    float scale_factor = OutputScaleFactor(component_index);
    uint32_t num_rows = component[component_index].num_rows_out;
    uint32_t num_columns = component[component_index].num_columns_out;

    printf("component %d : %s\n", component_index, intel_dnn_operation_name[component[component_index].operation]);
    if (number_type_ == kDnnFloat) {
        auto ptr_output = reinterpret_cast<float *>(component[component_index].ptr_outputs);
        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < num_columns; j++) {
                printf("%d %d : %e\n", i, j, ptr_output[i * num_columns + j] / scale_factor);
            }
        }
    } else {
        switch (component[component_index].num_bytes_per_output) {
            case 1: {
                auto ptr_output = reinterpret_cast<int8_t *>(component[component_index].ptr_outputs);
                for (int i = 0; i < num_rows; i++) {
                    for (int j = 0; j < num_columns; j++) {
                        printf("%d %d : %e\n", i, j, static_cast<float>(ptr_output[i * num_columns + j]) / scale_factor);
                    }
                }
            }
                break;
            case 2: {
                auto ptr_output = reinterpret_cast<int16_t *>(component[component_index].ptr_outputs);
                for (int i = 0; i < num_rows; i++) {
                    for (int j = 0; j < num_columns; j++) {
                        printf("%d %d : %e\n", i, j, static_cast<float>(ptr_output[i * num_columns + j]) / scale_factor);
                    }
                }
            }
                break;
            case 4: {
                auto ptr_output = reinterpret_cast<int32_t *>(component[component_index].ptr_outputs);
                for (int i = 0; i < num_rows; i++) {
                    for (int j = 0; j < num_columns; j++) {
                        printf("%d %d : %e\n", i, j, static_cast<float>(ptr_output[i * num_columns + j]) / scale_factor);
                    }
                }
            }
                break;
            default:
                fprintf(stderr,
                        "Bad num_bytes_per_output in component %d in AmIntelDnn::PrintOutputs()\n",
                        component_index);
                throw -1;
        }
    }
}

uint32_t AmIntelDnn::CompareScores(void *ptr_refscorearray, intel_score_error_t *score_error, uint32_t num_frames) {
    intel_dnn_component_t *ptr_component = &component[component.size() - 1];
    intel_dnn_orientation_t orientation = ptr_component->orientation_out;
    float scale_factor = OutputScaleFactor(component.size() - 1);
    uint32_t num_errors = 0;
    uint32_t num_rows = (orientation == kDnnInterleavedOrientation) ? ptr_component->num_rows_out : num_frames;
    uint32_t num_columns = (orientation == kDnnInterleavedOrientation) ? num_frames : ptr_component->num_columns_out;
    uint32_t num_row_step_ref =
        (orientation == kDnnInterleavedOrientation) ? ptr_component->num_rows_out : ptr_component->num_columns_out;
    uint32_t num_row_step = ptr_component->num_columns_out;

    if (ptr_component->operation == kDnnAffineOp) {
        num_rows = num_active_outputs_;
    }

    ClearScoreError(score_error);

    if (number_type_ == kDnnFloat) {
        auto A = reinterpret_cast<float *>(ptr_component->ptr_outputs);
        auto B = reinterpret_cast<float *>(ptr_refscorearray);
        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < num_columns; j++) {
                float score = A[i * num_row_step + j];
                float refscore =
                    (orientation == kDnnInterleavedOrientation) ? B[j * num_row_step_ref + i] : B[i * num_row_step_ref
                        + j];
                float scaled_score = score / scale_factor;
                float error = fabs(refscore - scaled_score);
                float rel_error = error / (fabs(refscore) + 1e-20);
                float squared_error = error * error;
                float squared_rel_error = rel_error * rel_error;
                score_error->num_scores++;
                score_error->sum_error += error;
                score_error->sum_squared_error += squared_error;
                if (error > score_error->max_error) {
                    score_error->max_error = error;
                }
                score_error->sum_rel_error += rel_error;
                score_error->sum_squared_rel_error += squared_rel_error;
                if (rel_error > score_error->max_rel_error) {
                    score_error->max_rel_error = rel_error;
                }
                if (error > score_error->threshold) {
                    num_errors++;
                }
            }
        }
    } else if (number_type_ == kDnnInt) {
        auto B = reinterpret_cast<float *>(ptr_refscorearray);
        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < num_columns; j++) {
                float score;
                if (ptr_component->num_bytes_per_output == 4) {
                    auto A = reinterpret_cast<int32_t *>(ptr_component->ptr_outputs);
                    score = static_cast<float>(A[i * num_row_step + j]);
                } else if (ptr_component->num_bytes_per_output == 2) {
                    auto A = reinterpret_cast<int16_t *>(ptr_component->ptr_outputs);
                    score = static_cast<float>(A[i * num_row_step + j]);
                } else {
                    fprintf(stderr,
                            "Unsupported output width (%d) in AmIntelDnn::CompareScores()!\n",
                            ptr_component->num_bytes_per_output);
                    throw -1;
                }
                float refscore =
                    (orientation == kDnnInterleavedOrientation) ? B[j * num_row_step_ref + i] : B[i * num_row_step_ref
                        + j];
                float scaled_score = score / scale_factor;
                float error = fabs(refscore - scaled_score);
                float rel_error = error / (fabs(refscore) + 1e-20);
                float squared_error = error * error;
                float squared_rel_error = rel_error * rel_error;
                score_error->num_scores++;
                score_error->sum_error += error;
                score_error->sum_squared_error += squared_error;
                if (error > score_error->max_error) {
                    score_error->max_error = error;
                }
                score_error->sum_rel_error += rel_error;
                score_error->sum_squared_rel_error += squared_rel_error;
                if (rel_error > score_error->max_rel_error) {
                    score_error->max_rel_error = rel_error;
                }
                if (error > score_error->threshold) {
                    num_errors++;
                }
            }
        }
    } else {
        fprintf(stderr, "Unknown number type in AmIntelDnn::CompareScores()!\n");
        throw -1;
    }

    score_error->num_errors = num_errors;

    return (num_errors);
}

void AmIntelDnn::WriteGraphWizModel(const char *filename) {
    auto & components = component;

#define IS_AFFINE(k)\
    (components[k].operation == kDnnAffineOp ||\
     components[k].operation == kDnnDiagonalOp)

#define IS_CONV(k)\
    (components[k].operation == kDnnConvolutional1dOp)

#define IS_RELU(k)\
    (components[k].operation == kDnnPiecewiselinearOp &&\
     components[k].op.pwl.func_id == kActRelu)


#define IS_DIAG(k)\
    (components[k].operation == kDnnDiagonalOp)

#define OUTPUTS(idx)\
    components[idx].ptr_outputs, components[idx].num_rows_out*components[idx].num_columns_out * components[idx].num_bytes_per_output

#define INPUTS(idx)\
    components[idx].ptr_inputs, components[idx].num_rows_in*components[idx].num_columns_in * components[idx].num_bytes_per_input

#define BIASES(idx)\
    components[idx].op.affine.ptr_biases,  components[idx].num_rows_in*components[idx].num_columns_in * components[idx].op.affine.num_bytes_per_bias

#define WEIGHTS(idx)\
    components[idx].op.affine.ptr_weights, components[idx].op.affine.num_bytes_per_weight * components[idx].num_rows_in*components[idx].num_columns_in * \
            (IS_DIAG(idx) ? 1 : components[idx].num_rows_out*components[idx].num_columns_out)

    auto intersected = [](void * ptra, size_t asize, void * ptrb, size_t bsize) {
        return !(((reinterpret_cast<char*>(ptra) + asize) <= ptrb) || ((reinterpret_cast<char*>(ptrb) + bsize) <= ptra));
    };

    auto equals = [](void * ptra, size_t asize, void * ptrb, size_t bsize) {
        // return !((((char*)ptra + asize) < ptrb) || (((char*)ptrb + bsize) < ptra));
        return ptra >= ptrb  && ptra < reinterpret_cast<char*>(ptrb) + bsize;
    };

    std::fstream graph("graph.dot", std::ios::out);
    graph << "strict digraph {";
    std::set<void*> weights;
    std::set<void*> biases;
    std::set<void*> outputs;
    std::set<std::string> layersNames;

    auto generate_layer_name = [&](int k) {
        std::string l;
        if (components[k].operation == kDnnPiecewiselinearOp) {
            l += intel_dnn_activation_name[components[k].op.pwl.func_id];
        } else {
            l += intel_dnn_operation_name[components[k].operation];
        }
        l += "_" + std::to_string(k);
        if (components[k].operation == kDnnPiecewiselinearOp) {
            graph << l << " [shape=box, style=filled, fillcolor=yellow";
        } else {
            graph << l << " [shape=box";
        }

        graph << ", label=<<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">\n"
            "  <TR><TD  colspan=\"2\">" <<  l << "</TD></TR>\n"
            "  <TR><TD  colspan=\"2\">" <<  components[k].num_rows_in << "x" <<  components[k].num_rows_out<< "</TD></TR>\n";
        if (IS_AFFINE(k)) {
            graph << "  <TR><TD> wscale</TD><TD>" <<  components[k].op.affine.weight_scale_factor<< "</TD></TR>\n";
            graph << "  <TR><TD> wbit</TD><TD>" <<  components[k].op.affine.num_bytes_per_weight<< "</TD></TR>\n";
            graph << "  <TR><TD> bbit</TD><TD>" <<  components[k].op.affine.num_bytes_per_bias<< "</TD></TR>\n";
        }
        if (IS_RELU(k)) {
            graph << "  <TR><TD> negative_slope</TD><TD>" <<  components[k].op.pwl.func_id.negative_slope<< "</TD></TR>\n";
        }
        if (IS_CONV(k)) {
            auto &conv = components[k].op.conv1D;
            graph << "  <TR><TD> num_filters</TD><TD>" <<  conv.num_filters<< "</TD></TR>\n";
            graph << "  <TR><TD> num_filter_rows</TD><TD>" <<  conv.num_filter_rows<< "</TD></TR>\n";
            graph << "  <TR><TD> num_filter_coefficients</TD><TD>" <<  conv.num_filter_coefficients<< "</TD></TR>\n";
            graph << "  <TR><TD> num_feature_maps</TD><TD>" <<  conv.num_feature_maps<< "</TD></TR>\n";
            graph << "  <TR><TD> num_feature_map_rows</TD><TD>" <<  conv.num_feature_map_rows<< "</TD></TR>\n";
            graph << "  <TR><TD> num_feature_map_columns</TD><TD>" <<  conv.num_feature_map_columns<< "</TD></TR>\n";
            graph << "  <TR><TD> wscale</TD><TD>" <<  conv.weight_scale_factor<< "</TD></TR>\n";
            graph << "  <TR><TD> wbit</TD><TD>" <<  conv.num_bytes_per_weight<< "</TD></TR>\n";
            graph << "  <TR><TD> bbit</TD><TD>" <<  conv.num_bytes_per_bias<< "</TD></TR>\n";
        }
        graph<<   "  <TR><TD> num_rows_in</TD><TD>" <<  components[k].num_rows_in<< "</TD></TR>\n"
                  "  <TR><TD> num_columns_in</TD><TD>" <<  components[k].num_columns_in<< "</TD></TR>\n"
                  "  <TR><TD> num_rows_out</TD><TD>" <<  components[k].num_rows_out<< "</TD></TR>\n"
                  "  <TR><TD> num_columns_out</TD><TD>" <<  components[k].num_columns_out<< "</TD></TR>\n"
                  "  <TR><TD> oscale</TD><TD>" <<  components[k].output_scale_factor<< "</TD></TR>\n"
                  "  <TR><TD> ibit</TD><TD>" <<  components[k].num_bytes_per_input<< "</TD></TR>\n"
                  "  <TR><TD> obit</TD><TD>" <<  components[k].num_bytes_per_output<< "</TD></TR>\n"
            "</TABLE>>];\n";

        return l;
    };


    for (int k = 0; k < components.size(); ++k) {
        std::string l = generate_layer_name(k);
        layersNames.insert(l);
        int lidx = std::distance(layersNames.begin(), layersNames.find(l));
        int widx = 0;
        int bidx = 0;

        if (IS_AFFINE(k)) {
            weights.insert(components[k].op.affine.ptr_weights);
            biases.insert(components[k].op.affine.ptr_biases);

            widx = std::distance(weights.begin(), weights.find(components[k].op.affine.ptr_weights));
            bidx = std::distance(biases.begin(), biases.find(components[k].op.affine.ptr_biases));
        }


        auto lw =  "weights_" +  std::to_string(lidx) + "_" + std::to_string(widx);;
        auto lb =  "biases_" +  std::to_string(lidx) + "_" + std::to_string(bidx);

        if (IS_AFFINE(k)) {
            graph << lw << " -> " << l << "[style=bold];";
            graph << lb << " -> " << l << "[style=bold];";
        }

        graph << "\n";

        bool inputConnected = false;

        for (int k2 = 0; k2 < components.size(); ++k2) {
            if (k2 == k) continue;


            std::string r = generate_layer_name(k2);

            int w2idx = 0;
            int b2idx = 0;

            if (IS_AFFINE(k2)) {
                weights.insert(components[k2].op.affine.ptr_weights);
                biases.insert(components[k2].op.affine.ptr_biases);

                w2idx = std::distance(weights.begin(), weights.find(components[k2].op.affine.ptr_weights));
                b2idx = std::distance(biases.begin(), biases.find(components[k2].op.affine.ptr_biases));
            }

            auto rw =  "weights_" + std::to_string(w2idx);
            auto rb =  "biases_" + std::to_string(b2idx);

            // ----------------------------------------------------------
            // output to input connections
            if (intersected(OUTPUTS(k2), INPUTS(k))) {
                graph << r <<" -> "<< l << ";";
                inputConnected = true;
            }

            // ----------------------------------------------------------
            // output to biases connections
            if (IS_AFFINE(k) && intersected(OUTPUTS(k2), BIASES(k))) {
                graph << r << " -> " << lb << " [label=\"OB\", fontcolor=blue, color=blue, style=dashed];";
            }

            // ----------------------------------------------------------
            // output to weights connections
            if (IS_AFFINE(k) && equals(OUTPUTS(k2), WEIGHTS(k))) {
                graph << r << " -> " << lw << " [label=\"OW\", fontcolor=magenta, color=magenta, style=dashed];";
            }

            // ----------------------------------------------------------
            // weights to input connections
            if (IS_AFFINE(k2) && equals(WEIGHTS(k2), INPUTS(k))) {
                graph << rw << " -> " << l << " [label=\"WI\", fontcolor=red, color=red, style=dashed];";
                inputConnected = true;
            }

            // ----------------------------------------------------------
            // weights to bias connections
            if (IS_AFFINE(k2) && IS_AFFINE(k) && equals(WEIGHTS(k2), BIASES(k))) {
                graph << rw << " -> " << lb << " [label=\"WB\", fontcolor=darkgreen,color=darkgreen, style=dashed];";
            }
        }
        if (!inputConnected) {
            // drawing tmp connection
            outputs.insert(components[k].ptr_inputs);
            auto tidx = std::distance(outputs.begin(), outputs.find(components[k].ptr_inputs));
            graph << tidx << " -> " << l
                  << " [label=\"FROM_TMP\", fontcolor=darkgreen,color=orange, style=dashed];";
        }
    }

    for (int k = 0; k < components.size(); ++k) {
        std::string l = generate_layer_name(k);

        int tidx = 0;
        for (auto tmpOutPtrs : outputs) {
            if (components[k].ptr_outputs == tmpOutPtrs) {
                graph << l << " -> " << tidx << " [label=\"TO_TMP\", fontcolor=darkgreen,color=orange, style=dashed];";
            }
            tidx++;
        }
    }

    graph << "}";
}

void AmIntelDnn::WriteDnnText(const char *filename, intel_dnn_number_type_t number_type) {
    if ((number_type_ == kDnnFloat) && (number_type == kDnnInt)) {
        fprintf(stderr, "Error trying to write floating point DNN as integer in AmIntelDnn::WriteDnnText().\n");
        fprintf(stderr, "  Please convert to integer first.\n");
        throw -1;
    }
#ifndef LIGHT_DUMP
    std::ofstream out_file1(filename, std::ios::out);
    std::ofstream &out_file = out_file1;
#else
    std::ofstream out_file((std::string(filename) + ".light").c_str(), std::ios::out);
#endif
    if (out_file.good()) {
        uint32_t num_inputs = component[0].num_rows_in;
        uint32_t num_outputs =
            (component[component.size() - 1].orientation_out == kDnnInterleavedOrientation) ? component[component.size()
                - 1].num_rows_out : component[component.size() - 1].num_columns_out;
        uint32_t num_layers = num_gna_layers();
        uint32_t num_group = this->num_group_in();
        uint32_t layer = 0;

        out_file << "<intel_dnn_file>\n";
        out_file << "<number_type> " << intel_dnn_number_type_name[number_type] << "\n";
        out_file << "<softmax_type> " << intel_dnn_softmax_name[softmax_type] << "\n";
        out_file << "<num_memory_bytes> " << std::dec << num_bytes_dnn_memory_ << "\n";
        out_file << "<num_group> " << std::dec << num_group << "\n";
        out_file << "<number_inputs> " << std::dec << num_inputs << "\n";
        out_file << "<num_outputs> " << std::dec << num_outputs << "\n";
        out_file << "<num_layers> " << std::dec << num_layers << "\n";
        for (uint32_t i = 0; i < component.size(); i++) {
#ifdef LIGHT_DUMP
            std::stringstream out_file_name;
            out_file_name << getDumpFolderName() << std::setfill('0') << std::setw(2) << i << "_"
                          << intel_dnn_operation_name[component[i].operation]
                          << "-" << component[i].num_rows_in
                          << "-" << component[i].num_rows_out;
            if (component[i].operation == kDnnPiecewiselinearOp) {
                out_file_name << "-" << intel_dnn_activation_name[component[i].op.pwl.func_id.type];
            }
            std::ofstream out_file((out_file_name.str() + ".txt").c_str(), std::ios::out);
#endif

            uint32_t num_rows_in = component[i].num_rows_in;
            uint32_t num_columns_in = component[i].num_columns_in;
            uint32_t num_rows_out = component[i].num_rows_out;
            uint32_t num_columns_out = component[i].num_columns_out;
            uint32_t num_bytes_per_input = component[i].num_bytes_per_input;
            uint32_t num_bytes_per_output = component[i].num_bytes_per_output;
            if ((component[i].operation == kDnnAffineOp)
                || (component[i].operation == kDnnDiagonalOp)
                || (component[i].operation == kDnnRecurrentOp)
                || (component[i].operation == kDnnConvolutional1dOp)
                || (component[i].operation == kDnnInterleaveOp)
                || (component[i].operation == kDnnDeinterleaveOp)
                || (component[i].operation == kDnnCopyOp)) {
                out_file << "<layer_index> " << std::dec << layer << "\n";
                layer++;
            }
            out_file << "<component_operation> " << intel_dnn_operation_name[component[i].operation] << "\n";
            out_file << "<macro_operation> " << intel_dnn_macro_operation_name[component[i].macro_operation] << "\n";
            out_file << "<num_rows_in> " << std::dec << num_rows_in << "\n";
            out_file << "<num_columns_in> " << std::dec << num_columns_in << "\n";
            out_file << "<num_rows_out> " << std::dec << num_rows_out << "\n";
            out_file << "<num_columns_out> " << std::dec << num_columns_out << "\n";
            out_file << "<orientation_in> " << std::dec << (component[i].orientation_in == kDnnInterleavedOrientation ?
            "interleaved" : "deinterleaved") << "\n";
            out_file << "<orientation_out> " << std::dec << (component[i].orientation_out == kDnnInterleavedOrientation ?
                                                            "interleaved" : "deinterleaved") << "\n";

            if ((number_type_ == kDnnInt) && (number_type == kDnnFloat)) {
                out_file << "<num_bytes_per_input> " << std::dec << sizeof(float) << "\n";
                out_file << "<num_bytes_per_output> " << std::dec << sizeof(float) << "\n";
            } else {
                out_file << "<num_bytes_per_input> " << std::dec << num_bytes_per_input << "\n";
                out_file << "<num_bytes_per_output> " << std::dec << num_bytes_per_output << "\n";
            }
            out_file << "<input_address> " << "0x" << std::setfill('0') << std::setw(8) << std::hex
                     << MemoryOffset(component[i].ptr_inputs, ptr_dnn_memory_) << "\n";
            out_file << "<output_address> " << "0x" << std::setfill('0') << std::setw(8) << std::hex
                     << MemoryOffset(component[i].ptr_outputs, ptr_dnn_memory_) << "\n";
            switch (component[i].operation) {
                case kDnnAffineOp:
                case kDnnDiagonalOp: {
                    uint32_t num_bytes_per_weight = component[i].op.affine.num_bytes_per_weight;
                    uint32_t num_bytes_per_bias = component[i].op.affine.num_bytes_per_bias;
                    float weight_scale_factor = component[i].op.affine.weight_scale_factor;
                    float output_scale_factor = component[i].output_scale_factor;
                    uint32_t num_weight_rows = (component[i].operation == kDnnDiagonalOp) ? 1 : num_rows_out;
                    uint32_t num_weight_columns = num_rows_in;
                    if ((number_type_ == kDnnInt) && (number_type == kDnnFloat)) {
                        out_file << "<num_bytes_per_weight> " << std::dec << 4 << "\n";
                        out_file << "<num_bytes_per_bias> " << std::dec << 4 << "\n";
                    } else {
                        out_file << "<num_bytes_per_weight> " << std::dec << num_bytes_per_weight << "\n";
                        out_file << "<num_bytes_per_bias> " << std::dec << num_bytes_per_bias << "\n";
                    }
                    if ((number_type_ == kDnnInt) && (number_type == kDnnFloat)) {
                        out_file << std::setprecision(12) << std::scientific << "<weight_scale_factor> " << 1.0 << "\n";
                        out_file << std::setprecision(12) << std::scientific << "<output_scale_factor> " << 1.0 << "\n";
                    } else {
                        out_file << std::setprecision(12) << std::scientific << "<weight_scale_factor> "
                                 << weight_scale_factor << "\n";
                        out_file << std::setprecision(12) << std::scientific << "<output_scale_factor> "
                                 << output_scale_factor << "\n";
                    }
                    out_file << "<weight_address> " << "0x" << std::setfill('0') << std::setw(8) << std::hex
                             << MemoryOffset(component[i].op.affine.ptr_weights, ptr_dnn_memory_) << "\n";
                    out_file << "<bias_address> " << "0x" << std::setfill('0') << std::setw(8) << std::hex
                             << MemoryOffset(component[i].op.affine.ptr_biases, ptr_dnn_memory_) << "\n";

                    std::ofstream out_wfile((out_file_name.str() + "_weights.txt").c_str(), std::ios::out);
                    std::ofstream out_bfile((out_file_name.str() + "_biases.txt").c_str(), std::ios::out);

                    if (num_bytes_per_weight == 1) {
                        int8_t *ptr_weight = reinterpret_cast<int8_t *>(component[i].op.affine.ptr_weights);
                        intel_compound_bias_t *ptr_bias = reinterpret_cast<intel_compound_bias_t *>(component[i].op.affine.ptr_biases);
#ifdef DUMP_WB
                        for (uint32_t row = 0; row < num_weight_rows; row++) {
                            for (uint32_t col = 0; col < num_weight_columns; col++) {
                                if (number_type == kDnnFloat) {
                                    float val =
                                        static_cast<float>(ptr_weight[row * num_weight_columns + col]) * ptr_bias[row].multiplier
                                            / weight_scale_factor;
                                    out_wfile << std::setprecision(4) << val << " ";
                                } else {
                                    out_wfile <<  int((int8_t) ptr_weight[row * num_weight_columns + col]) << " ";
                                }
                                out_wfile << "\n";
                            }
                        }
#endif
                    } else if (num_bytes_per_weight == 2) {
                        int16_t *ptr_weight = reinterpret_cast<int16_t *>(component[i].op.affine.ptr_weights);
#ifdef DUMP_WB
                        for (uint32_t row = 0; row < num_weight_rows; row++) {
                            for (uint32_t col = 0; col < num_weight_columns; col++) {
                                if (number_type == kDnnFloat) {
                                    out_wfile << std::setprecision(12)
                                              << ptr_weight[row * num_weight_columns + col] / weight_scale_factor << " ";
                                } else {
                                    out_wfile << ptr_weight[row * num_weight_columns + col] << " ";
                                }
                                out_wfile << "\n";
                            }
                        }
#endif
                    } else if (number_type_ == kDnnFloat) {
                        float *ptr_weight = reinterpret_cast<float *>(component[i].op.affine.ptr_weights);
#ifdef DUMP_WB
                        for (uint32_t row = 0; row < num_weight_rows; row++) {
                            for (uint32_t col = 0; col < num_weight_columns; col++) {
                                out_wfile << std::setprecision(5)
                                          << ptr_weight[row * num_weight_columns + col] << " ";
                                out_wfile << "\n";
                            }
                        }
#endif
                    } else {
                        fprintf(stderr, "Unsupported weight type in WriteDnnText!\n");
                        throw -1;
                    }
                    if (number_type_ == kDnnInt) {
                        if (num_bytes_per_weight == 1) {
                            intel_compound_bias_t
                                *ptr_biases = reinterpret_cast<intel_compound_bias_t *>(component[i].op.affine.ptr_biases);
#ifdef DUMP_WB
                            for (uint32_t row = 0; row < num_rows_out; row++) {
                                out_bfile << std::setw(8) << ptr_biases[row].bias << ", ";
                                out_bfile << std::setw(8) << int(ptr_biases[row].multiplier) << "\n";
                            }
#endif
                        } else {
                            int32_t *ptr_biases = reinterpret_cast<int32_t *>(component[i].op.affine.ptr_biases);
#ifdef DUMP_WB
                            for (uint32_t row = 0; row < num_rows_out; row++) {
                                if (number_type == kDnnInt) {
                                    out_bfile << std::setw(8) << ptr_biases[row] << "\n";
                                } else {
                                    out_bfile << std::setw(8) << ptr_biases[row] / output_scale_factor << "\n";
                                }
                            }
#endif
                        }

                    } else {
                        float *ptr_biases = reinterpret_cast<float *>(component[i].op.affine.ptr_biases);
#ifdef DUMP_WB

                        for (uint32_t row = 0; row < num_rows_out; row++) {
                            out_bfile << std::setprecision(5) << ptr_biases[row] << "\n";
                        }
#endif
                    }
                }
                break;
                case kDnnConvolutional1dOp: {
                    uint32_t num_filters = component[i].op.conv1D.num_filters;
                    uint32_t num_filter_rows = component[i].op.conv1D.num_filter_rows;
                    uint32_t num_filter_coefficients = component[i].op.conv1D.num_filter_coefficients;
                    uint32_t num_feature_maps = component[i].op.conv1D.num_feature_maps;
                    uint32_t num_feature_map_rows = component[i].op.conv1D.num_feature_map_rows;
                    uint32_t num_feature_map_columns = component[i].op.conv1D.num_feature_map_columns;
                    uint32_t num_filter_outputs =
                        component[i].op.conv1D.num_feature_map_rows - component[i].op.conv1D.num_filter_rows + 1;
                    uint32_t num_bytes_per_weight = component[i].op.conv1D.num_bytes_per_weight;
                    uint32_t num_bytes_per_bias = component[i].op.conv1D.num_bytes_per_bias;
                    float weight_scale_factor = component[i].op.conv1D.weight_scale_factor;
                    float output_scale_factor = component[i].output_scale_factor;
                    out_file << "<num_filters> " << std::dec << num_filters << "\n";
                    out_file << "<num_filter_coefficients> " << std::dec << num_filter_coefficients << "\n";
                    out_file << "<num_filter_rows> " << std::dec << num_filter_rows << "\n";
                    out_file << "<num_feature_maps> " << std::dec << num_feature_maps << "\n";
                    out_file << "<num_feature_map_rows> " << std::dec << num_feature_map_rows << "\n";
                    out_file << "<num_feature_map_columns> " << std::dec << num_feature_map_columns << "\n";
                    if ((number_type_ == kDnnInt) && (number_type == kDnnFloat)) {
                        out_file << "<num_bytes_per_weight> " << std::dec << 4 << "\n";
                        out_file << "<num_bytes_per_bias> " << std::dec << 4 << "\n";
                    } else {
                        out_file << "<num_bytes_per_weight> " << std::dec << num_bytes_per_weight << "\n";
                        out_file << "<num_bytes_per_bias> " << std::dec << num_bytes_per_bias << "\n";
                    }
                    if ((number_type_ == kDnnInt) && (number_type == kDnnFloat)) {
                        out_file << std::setprecision(12) << std::scientific << "<weight_scale_factor> " << 1.0 << "\n";
                        out_file << std::setprecision(12) << std::scientific << "<output_scale_factor> " << 1.0 << "\n";
                    } else {
                        out_file << std::setprecision(12) << std::scientific << "<weight_scale_factor> "
                                 << weight_scale_factor << "\n";
                        out_file << std::setprecision(12) << std::scientific << "<output_scale_factor> "
                                 << output_scale_factor << "\n";
                    }
                    out_file << "<filter_address> " << "0x" << std::setfill('0') << std::setw(8) << std::hex
                             << MemoryOffset(component[i].op.conv1D.ptr_filters, ptr_dnn_memory_) << "\n";
                    out_file << "<bias_address> " << "0x" << std::setfill('0') << std::setw(8) << std::hex
                             << MemoryOffset(component[i].op.conv1D.ptr_biases, ptr_dnn_memory_) << "\n";


                    std::ofstream out_wfile((out_file_name.str() + "_weights.txt").c_str(), std::ios::out);
                    std::ofstream out_bfile((out_file_name.str() + "_biases.txt").c_str(), std::ios::out);


                    if (num_bytes_per_weight == 1) {
                        int8_t *ptr_weight = reinterpret_cast<int8_t *>(component[i].op.conv1D.ptr_filters);
                        intel_compound_bias_t *ptr_bias = reinterpret_cast<intel_compound_bias_t *>(component[i].op.conv1D.ptr_biases);
#ifdef DUMP_WB
                        for (uint32_t row = 0; row < num_filters; row++) {
                            for (uint32_t col = 0; col < num_filter_coefficients; col++) {
                                if (number_type == kDnnFloat) {
                                    float val = static_cast<float>(ptr_weight[row * num_filter_coefficients + col])
                                        * ptr_bias[row].multiplier / weight_scale_factor;
                                    out_wfile << std::setprecision(12) <<val << "\n";
                                } else {
                                    out_wfile << "0x" << std::setfill('0') << std::setw(2) << std::hex
                                             << int((uint8_t) ptr_weight[row * num_filter_coefficients + col]) << "\n";
                                }
                            }
                        }
#endif
                    } else if (num_bytes_per_weight == 2) {
                        int16_t *ptr_weight = reinterpret_cast<int16_t *>(component[i].op.conv1D.ptr_filters);
#ifdef DUMP_WB
                        for (uint32_t row = 0; row < num_filters; row++) {
                            for (uint32_t col = 0; col < num_filter_coefficients; col++) {
                                if (number_type == kDnnFloat) {
                                    out_wfile << std::setprecision(12)
                                             << ptr_weight[row * num_filter_coefficients + col] / weight_scale_factor
                                             << "\n";
                                } else {
                                    out_wfile << "0x" << std::setfill('0') << std::setw(4) << std::hex
                                             << ptr_weight[row * num_filter_coefficients + col] << "\n";
                                }
                            }
                        }
#endif
                    } else if (number_type_ == kDnnFloat) {
                        float *ptr_weight = reinterpret_cast<float *>(component[i].op.conv1D.ptr_filters);
#ifdef DUMP_WB
                        for (uint32_t row = 0; row < num_filters; row++) {
                            for (uint32_t col = 0; col < num_filter_coefficients; col++) {
                                out_wfile << std::setprecision(12)
                                         << ptr_weight[row * num_filter_coefficients + col] << "\n";
                            }
                            out_wfile << "\n";
                        }
#endif
                    } else {
                        fprintf(stderr, "Unsupported filter weight type in WriteDnnText!\n");
                        throw -1;
                    }

                    if (number_type_ == kDnnInt) {
                        if (number_type == kDnnInt) {
                            if (num_bytes_per_weight == 1) {
                                intel_compound_bias_t
                                    *ptr_biases = reinterpret_cast<intel_compound_bias_t *>(component[i].op.conv1D.ptr_biases);
#ifdef DUMP_WB
                                for (uint32_t row = 0; row < num_filters; row++) {
                                    out_bfile << "0x" << std::setfill('0') << std::setw(8) << std::hex
                                             << ptr_biases[row].bias << " ";
                                    out_bfile << "0x" << std::setfill('0') << std::setw(8) << std::hex
                                             << int(ptr_biases[row].multiplier) << "\n";
                                }
#endif
                            } else {
                                int32_t *ptr_biases = reinterpret_cast<int32_t *>(component[i].op.conv1D.ptr_biases);
#ifdef DUMP_WB
                                for (uint32_t row = 0; row < num_filters; row++) {
                                    out_bfile << "0x" << std::setfill('0') << std::setw(8) << std::hex << ptr_biases[row]
                                             << "\n";
                                }
#endif
                            }
                        } else {
                            int32_t *ptr_biases = reinterpret_cast<int32_t *>(component[i].op.conv1D.ptr_biases);
#ifdef DUMP_WB
                            for (uint32_t row = 0; row < num_filters; row++) {
                                out_bfile << std::setprecision(12)
                                         << ptr_biases[row] / output_scale_factor << "\n";
                            }
#endif
                        }
                    } else {
                        float *ptr_biases = reinterpret_cast<float *>(component[i].op.conv1D.ptr_biases);
#ifdef DUMP_WB
                        for (uint32_t row = 0; row < num_filters; row++) {
                            out_bfile << std::setprecision(12) << ptr_biases[row] << "\n";
                        }
#endif
                    }
                    out_file << "\n";
                }
                    break;
                case kDnnRecurrentOp: {
                    float weight_scale_factor = component[i].op.recurrent.weight_scale_factor;
                    float output_scale_factor = component[i].output_scale_factor;
                    uint32_t num_vector_delay = component[i].op.recurrent.num_vector_delay;
                    uint32_t num_bytes_per_weight = component[i].op.recurrent.num_bytes_per_weight;
                    uint32_t num_bytes_per_bias = component[i].op.recurrent.num_bytes_per_bias;
                    uint32_t num_weight_rows = num_columns_out;
                    uint32_t num_weight_columns = num_columns_in + num_columns_out;
                    out_file << "<num_vector_delay> " << std::dec << num_vector_delay << "\n";
                    if ((number_type_ == kDnnInt) && (number_type == kDnnFloat)) {
                        out_file << "<num_bytes_per_weight> " << std::dec << 4 << "\n";
                        out_file << "<num_bytes_per_bias> " << std::dec << 4 << "\n";
                    } else {
                        out_file << "<num_bytes_per_weight> " << std::dec << num_bytes_per_weight << "\n";
                        out_file << "<num_bytes_per_bias> " << std::dec << num_bytes_per_bias << "\n";
                    }
                    if ((number_type_ == kDnnInt) && (number_type == kDnnFloat)) {
                        out_file << std::setprecision(12) << std::scientific << "<weight_scale_factor> " << 1.0 << "\n";
                        out_file << std::setprecision(12) << std::scientific << "<output_scale_factor> " << 1.0 << "\n";
                    } else {
                        out_file << std::setprecision(12) << std::scientific << "<weight_scale_factor> "
                                 << weight_scale_factor << "\n";
                        out_file << std::setprecision(12) << std::scientific << "<output_scale_factor> "
                                 << output_scale_factor << "\n";
                    }
                    out_file << "<weight_address> " << "0x" << std::setfill('0') << std::setw(8) << std::hex
                             << MemoryOffset(component[i].op.recurrent.ptr_weights, ptr_dnn_memory_) << "\n";
                    out_file << "<bias_address> " << "0x" << std::setfill('0') << std::setw(8) << std::hex
                             << MemoryOffset(component[i].op.recurrent.ptr_biases, ptr_dnn_memory_) << "\n";
                    out_file << "<feedback_address> " << "0x" << std::setfill('0') << std::setw(8) << std::hex
                             << MemoryOffset(component[i].op.recurrent.ptr_feedbacks, ptr_dnn_memory_) << "\n";
                    if (num_bytes_per_weight == 1) {
                        int8_t *ptr_weight = reinterpret_cast<int8_t *>(component[i].op.recurrent.ptr_weights);
                        intel_compound_bias_t
                            *ptr_bias = reinterpret_cast<intel_compound_bias_t *>(component[i].op.recurrent.ptr_biases);
#ifdef DUMP_WB
                        for (uint32_t row = 0; row < num_weight_rows; row++) {
                            out_file << "<weight_row> ";
                            for (uint32_t col = 0; col < num_weight_columns; col++) {
                                if (number_type == kDnnFloat) {
                                    float val =
                                        static_cast<float>(ptr_weight[row * num_weight_columns + col]) * ptr_bias[col].multiplier
                                            / weight_scale_factor;
                                    out_file << std::setprecision(12) << std::scientific << val << " ";
                                } else {
                                    out_file << "0x" << std::setfill('0') << std::setw(2) << std::hex
                                             << int((uint8_t) ptr_weight[row * num_weight_columns + col]) << " ";
                                }
                            }
                            out_file << "\n";
                        }
#endif
                    } else if (num_bytes_per_weight == 2) {
                        int16_t *ptr_weight = reinterpret_cast<int16_t *>(component[i].op.recurrent.ptr_weights);
#ifdef DUMP_WB
                        for (uint32_t row = 0; row < num_weight_rows; row++) {
                            out_file << "<weight_row> ";
                            for (uint32_t col = 0; col < num_weight_columns; col++) {
                                if (number_type == kDnnFloat) {
                                    out_file << std::setprecision(12) << std::scientific
                                             << ptr_weight[row * num_weight_columns + col] / weight_scale_factor << " ";
                                } else {
                                    out_file << "0x" << std::setfill('0') << std::setw(4) << std::hex
                                             << ptr_weight[row * num_weight_columns + col] << " ";
                                }
                            }
                            out_file << "\n";
                        }
#endif
                    } else if (number_type_ == kDnnFloat) {
                        float *ptr_weight = reinterpret_cast<float *>(component[i].op.recurrent.ptr_weights);
#ifdef DUMP_WB
                        for (uint32_t row = 0; row < num_weight_rows; row++) {
                            out_file << "<weight_row> ";
                            for (uint32_t col = 0; col < num_weight_columns; col++) {
                                out_file << std::setprecision(12) << std::scientific
                                         << ptr_weight[row * num_weight_columns + col] << " ";
                            }
                            out_file << "\n";
                        }
#endif
                    } else {
                        fprintf(stderr, "Unsupported weight type in WriteDnnText!\n");
                        throw -1;
                    }
                    if (number_type_ == kDnnInt) {
                        if (number_type == kDnnInt) {
                            if (num_bytes_per_weight == 1) {
                                intel_compound_bias_t
                                    *ptr_biases = reinterpret_cast<intel_compound_bias_t *>(component[i].op.recurrent.ptr_biases);
                                out_file << "<compound_bias>" << " ";
#ifdef DUMP_WB
                                for (uint32_t col = 0; col < num_columns_out; col++) {
                                    out_file << "0x" << std::setfill('0') << std::setw(8) << std::hex
                                             << ptr_biases[col].bias << " ";
                                    out_file << "0x" << std::setfill('0') << std::setw(8) << std::hex
                                             << ptr_biases[col].multiplier << " ";
                                }
#endif
                            } else {
                                int32_t *ptr_biases = reinterpret_cast<int32_t *>(component[i].op.recurrent.ptr_biases);
                                out_file << "<bias>" << " ";
#ifdef DUMP_WB
                                for (uint32_t col = 0; col < num_columns_out; col++) {
                                    out_file << "0x" << std::setfill('0') << std::setw(8) << std::hex << ptr_biases[col]
                                             << " ";
                                }
#endif
                            }
                        } else {
                            int32_t *ptr_biases = reinterpret_cast<int32_t *>(component[i].op.recurrent.ptr_biases);
                            out_file << "<bias>" << " ";
#ifdef DUMP_WB
                            for (uint32_t col = 0; col < num_columns_out; col++) {
                                out_file << std::setprecision(12) << std::scientific
                                         << ptr_biases[col] / output_scale_factor << " ";
                            }
#endif
                        }
                    } else {
                        float *ptr_biases = reinterpret_cast<float *>(component[i].op.recurrent.ptr_biases);
                        out_file << "<bias>" << " ";
#ifdef DUMP_WB
                        for (uint32_t col = 0; col < num_columns_out; col++) {
                            out_file << std::setprecision(12) << std::scientific << ptr_biases[col] << " ";
                        }
#endif
                    }
                    out_file << "\n";
                }
                    break;
                case kDnnMaxPoolOp: {
                    uint32_t num_pool_type = (component[i].op.maxpool.do_sum_not_max) ? 2 : 1;
                    out_file << "<pool_type> " << std::dec << num_pool_type << "\n";
                    out_file << "<pool_size> " << std::dec << component[i].op.maxpool.num_inputs << "\n";
                    out_file << "<pool_step> " << std::dec << component[i].op.maxpool.num_inputs_step << "\n";
                    out_file << "<pool_num_rows> " << std::dec << component[i].op.maxpool.num_inputs_stride << "\n";
                    out_file << std::setprecision(12) << std::scientific << "<output_scale_factor> "
                             << component[i].output_scale_factor << "\n";
                }
                    break;
                case kDnnPiecewiselinearOp: {
                    intel_pwl_segment_t *ptr_segment = component[i].op.pwl.ptr_segments;
                    DnnActivationType func_id = component[i].op.pwl.func_id.type;
                    uint32_t num_segments = component[i].op.pwl.num_segments;
                    float output_scale_factor = component[i].output_scale_factor;
                    out_file << "<func_id> " << intel_dnn_activation_name[func_id] << "\n";
                    out_file << "<num_bytes_per_slope> " << std::dec << sizeof(int16_t) << "\n";
                    out_file << "<num_bytes_per_intercept> " << std::dec << sizeof(int16_t) << "\n";
                    out_file << "<num_bytes_per_offset> " << std::dec << sizeof(int32_t) << "\n";
                    if (number_type == kDnnFloat) {
                        out_file << std::setprecision(12) << std::scientific << "<output_scale_factor> " << 1.0 << "\n";
                        out_file << "<num_segments> " << std::dec << 0 << "\n";
                        out_file << "<segment_address> " << "0x" << std::setfill('0') << std::setw(8) << std::hex
                                 << MemoryOffset(component[i].op.pwl.ptr_segments, ptr_dnn_memory_) << "\n";
                    } else {
                        out_file << std::setprecision(12) << std::scientific << "<output_scale_factor> "
                                 << output_scale_factor << "\n";
                        out_file << "<num_segments> " << std::dec << num_segments << "\n";
                        out_file << "<segment_address> " << "0x" << std::setfill('0') << std::setw(8) << std::hex
                                 << MemoryOffset(component[i].op.pwl.ptr_segments, ptr_dnn_memory_) << "\n";
                        if (number_type_ == kDnnInt) {
                            out_file << "<slope> ";
                            for (int segment = 0; segment < num_segments; segment++) {
                                out_file << "0x" << std::setfill('0') << std::setw(4) << std::hex
                                         << ptr_segment[segment].slope << " ";
                            }
                            out_file << "\n";
                            out_file << "<intercept> ";
                            for (int segment = 0; segment < component[i].op.pwl.num_segments; segment++) {
                                out_file << "0x" << std::setfill('0') << std::setw(4) << std::hex
                                         << ptr_segment[segment].yBase << " ";
                            }
                            out_file << "\n";
                            out_file << "<offset> ";
                            for (int segment = 0; segment < component[i].op.pwl.num_segments; segment++) {
                                out_file << "0x" << std::setfill('0') << std::setw(8) << std::hex
                                         << ptr_segment[segment].xBase << " ";
                            }
                            out_file << "\n";
                        } else if (num_segments > 0) {
                            fprintf(stderr,
                                    "Number of segments must be zero in floating point model in WriteDnnText!\n");
                            throw -1;
                        }
                    }
                }
                    break;
                case kDnnInterleaveOp:
                    out_file << std::setprecision(12) << std::scientific << "<output_scale_factor> "
                             << component[i].output_scale_factor << "\n";
                    break;
                case kDnnDeinterleaveOp:
                    out_file << std::setprecision(12) << std::scientific << "<output_scale_factor> "
                             << component[i].output_scale_factor << "\n";
                    break;
                case kDnnCopyOp:
                    out_file << std::setprecision(12) << std::scientific << "<output_scale_factor> "
                             << component[i].output_scale_factor << "\n";
                    out_file << "<num_copy_rows> " << std::dec << component[i].op.copy.num_copy_rows << "\n";
                    out_file << "<num_copy_columns> " << std::dec << component[i].op.copy.num_copy_columns << "\n";
                    break;
                default:
                    out_file << "<Error!!!> Unsupported Component :  "
                             << intel_dnn_operation_name[component[i].operation] << "\n";
                    //  fprintf(stderr, "Component type %s not yet supported in AmIntelDnn::WriteDnnText()!\n",
                    //    intel_dnn_operation_name[component[i].operation]);
                    //  throw -1;
                    break;
            }
        }
        if (ptr_active_outputs() != nullptr) {
            out_file << "<activelist_address> " << "0x" << std::setfill('0') << std::setw(8) << std::hex
                     << MemoryOffset(ptr_active_outputs(), ptr_dnn_memory_) << "\n";
        }
        out_file << "<end_of_file>\n";
        out_file.close();
    } else {
        fprintf(stderr, "Failed to open %s for writing!\n", filename);
        throw -1;
    }
}

void AmIntelDnn::InitGNAStruct(intel_nnet_type_t *ptr_nnet) {
    intel_nnet_layer_t *pLayer;

    if (ptr_nnet == nullptr)
        THROW_GNA_EXCEPTION << "Invalid input parameter";
    if (ptr_nnet->pLayers != nullptr)
        THROW_GNA_EXCEPTION << "InitGNAStruct can't work on prellocated layers array";
    if (component.empty())
        THROW_GNA_EXCEPTION << "empty model in AmIntelDnn::FillGNAStruct()";

    ptr_nnet->nLayers = 0;
    for (auto && c : component) {
        if (c.operation == kDnnAffineOp
            || (c.operation == kDnnDiagonalOp)
            || (c.operation == kDnnConvolutional1dOp)
            || (c.operation == kDnnDeinterleaveOp)
            || (c.operation == kDnnInterleaveOp)
            || (c.operation == kDnnRecurrentOp)
            || (c.operation == kDnnCopyOp)
            ) {
            ptr_nnet->nLayers++;
        }
    }
    ptr_nnet->nGroup = num_group_in();
    ptr_nnet->pLayers = reinterpret_cast<intel_nnet_layer_t *>(_mm_malloc(ptr_nnet->nLayers * sizeof(intel_nnet_layer_t), 64));
    if (ptr_nnet->pLayers == nullptr)
        THROW_GNA_EXCEPTION << "out of memory in AmIntelDnn::FillGNAStruct()";
    pLayer = ptr_nnet->pLayers;

    for (int i = 0; i < component.size(); i++) {
        // std::cout << "Component + " << i <<"=GNA_" << std::distance(ptr_nnet->pLayers, pLayer) << "\n";
        switch (component[i].operation) {
            case kDnnAffineOp:
                pLayer->nInputRows = component[i].num_rows_in;
                pLayer->nInputColumns = component[i].num_columns_in;
                pLayer->nOutputRows = component[i].num_rows_out;
                pLayer->nOutputColumns = component[i].num_columns_out;
                pLayer->nBytesPerInput = component[i].num_bytes_per_input;
                pLayer->nBytesPerOutput = component[i].num_bytes_per_output;  //  will be overwritten if PWL op is needed
                pLayer->nBytesPerIntermediateOutput = sizeof(int32_t);
                pLayer->pInputs = component[i].ptr_inputs;
                pLayer->pOutputsIntermediate = component[i].ptr_outputs;
                pLayer->pOutputs = component[i].ptr_outputs;  //  will be overwritten if PWL op is needed
                pLayer->nLayerKind = INTEL_AFFINE;
                {
                    pLayer->pLayerStruct = _mm_malloc(sizeof(intel_affine_layer_t), 64);
                    if (pLayer->pLayerStruct == nullptr) {
                        THROW_GNA_EXCEPTION << "could not allocate memory for INTEL_AFFINE layer structure.";
                    }
                    auto pAffineLayer = reinterpret_cast<intel_affine_layer_t *>(pLayer->pLayerStruct);
                    pAffineLayer->pwl.pSegments = nullptr;
                    pAffineLayer->pwl.nSegments = 0;

                    pAffineLayer->affine.nBytesPerBias = component[i].op.affine.num_bytes_per_bias;
                    pAffineLayer->affine.nBytesPerWeight = component[i].op.affine.num_bytes_per_weight;
                    pAffineLayer->affine.pBiases = component[i].op.affine.ptr_biases;
                    pAffineLayer->affine.pWeights = component[i].op.affine.ptr_weights;
                }
                if (i == component.size() - 1 || component[i + 1].operation != kDnnPiecewiselinearOp) {
                    pLayer++;
                }
                break;
            case kDnnDiagonalOp:
                pLayer->nInputRows = component[i].num_rows_in;
                pLayer->nInputColumns = component[i].num_columns_in;
                pLayer->nOutputRows = component[i].num_rows_out;
                pLayer->nOutputColumns = component[i].num_columns_out;
                pLayer->nBytesPerInput = component[i].num_bytes_per_input;
                pLayer->nBytesPerOutput = component[i].num_bytes_per_output;  //  will be overwritten if PWL op is needed
                pLayer->nBytesPerIntermediateOutput = sizeof(int32_t);
                pLayer->pInputs = component[i].ptr_inputs;
                pLayer->pOutputsIntermediate = component[i].ptr_outputs;
                pLayer->pOutputs = component[i].ptr_outputs;  //  will be overwritten if PWL op is needed
                pLayer->nLayerKind = INTEL_AFFINE_DIAGONAL;
                {
                    pLayer->pLayerStruct = _mm_malloc(sizeof(intel_affine_layer_t), 64);
                    if (pLayer->pLayerStruct == nullptr) {
                        THROW_GNA_EXCEPTION << "could not allocate memory for INTEL_AFFINE_DIAGONAL layer structure.";
                    }
                    auto pDiagonalLayer = reinterpret_cast<intel_affine_layer_t *>(pLayer->pLayerStruct);
                    pDiagonalLayer->pwl.pSegments = nullptr;
                    pDiagonalLayer->pwl.nSegments = 0;

                    pDiagonalLayer->affine.nBytesPerBias = component[i].op.affine.num_bytes_per_bias;
                    pDiagonalLayer->affine.nBytesPerWeight = component[i].op.affine.num_bytes_per_weight;
                    pDiagonalLayer->affine.pBiases = component[i].op.affine.ptr_biases;
                    pDiagonalLayer->affine.pWeights = component[i].op.affine.ptr_weights;
                }
                if (i == component.size() - 1 || component[i + 1].operation != kDnnPiecewiselinearOp) {
                    pLayer++;
                }
                break;
            case kDnnRecurrentOp:
                pLayer->nInputRows = component[i].num_rows_in;
                pLayer->nInputColumns = component[i].num_columns_in;
                pLayer->nOutputRows = component[i].num_rows_out;
                pLayer->nOutputColumns = component[i].num_columns_out;
                pLayer->nBytesPerInput = component[i].num_bytes_per_input;
                pLayer->nBytesPerOutput = component[i].num_bytes_per_output;  //  will be overwritten if PWL op is needed
                pLayer->nBytesPerIntermediateOutput = sizeof(int32_t);
                pLayer->pInputs = component[i].ptr_inputs;
                pLayer->pOutputsIntermediate = component[i].ptr_outputs;
                pLayer->pOutputs = component[i].ptr_outputs;  //  will be overwritten if PWL op is needed
                pLayer->nLayerKind = INTEL_RECURRENT;
                {
                    pLayer->pLayerStruct = _mm_malloc(sizeof(intel_recurrent_layer_t), 64);
                    if (pLayer->pLayerStruct == nullptr) {
                        THROW_GNA_EXCEPTION << "could not allocate memory for INTEL_RECURRENT layer structure.";
                    }
                    auto pRecurrentLayer = reinterpret_cast<intel_recurrent_layer_t *>(pLayer->pLayerStruct);
                    pRecurrentLayer->pFeedbackBuffer = component[i].op.recurrent.ptr_feedbacks;
                    pRecurrentLayer->pwl.pSegments = nullptr;
                    pRecurrentLayer->pwl.nSegments = 0;

                    pRecurrentLayer->affine.nBytesPerBias = component[i].op.recurrent.num_bytes_per_bias;
                    pRecurrentLayer->affine.nBytesPerWeight = component[i].op.recurrent.num_bytes_per_weight;
                    pRecurrentLayer->affine.pBiases = component[i].op.recurrent.ptr_biases;
                    pRecurrentLayer->affine.pWeights = component[i].op.recurrent.ptr_weights;
                }
                if (i == component.size() - 1 || component[i + 1].operation != kDnnPiecewiselinearOp) {
                    pLayer++;
                }
                break;
            case kDnnConvolutional1dOp:
                pLayer->nInputRows = component[i].num_rows_in;
                pLayer->nInputColumns = component[i].num_columns_in;
                pLayer->nOutputRows = component[i].num_rows_out;
                pLayer->nOutputColumns = component[i].num_columns_out;
                pLayer->nBytesPerInput = component[i].num_bytes_per_input;
                pLayer->nBytesPerOutput = component[i].num_bytes_per_output;  //  will be overwritten
                pLayer->nBytesPerIntermediateOutput = sizeof(int32_t);
                pLayer->pInputs = component[i].ptr_inputs;
                pLayer->pOutputsIntermediate = component[i].ptr_outputs;
                pLayer->pOutputs = component[i].ptr_outputs;  //  will be overwritten
                pLayer->nLayerKind = INTEL_CONVOLUTIONAL;
                {
                    pLayer->pLayerStruct = _mm_malloc(sizeof(intel_convolutional_layer_t), 64);
                    if (pLayer->pLayerStruct == nullptr) {
                        THROW_GNA_EXCEPTION << "could not allocate memory for INTEL_CONVOLUTIONAL layer structure.";
                    }
                    auto pConvolutionalLayer = reinterpret_cast<intel_convolutional_layer_t *>(pLayer->pLayerStruct);
                    pConvolutionalLayer->nBytesBias = component[i].op.conv1D.num_bytes_per_bias;
                    pConvolutionalLayer->nBytesFilterCoefficient = component[i].op.conv1D.num_bytes_per_weight;
                    pConvolutionalLayer->nFilters = component[i].op.conv1D.num_filters;
                    pConvolutionalLayer->nFilterRows = component[i].op.conv1D.num_filter_rows;
                    pConvolutionalLayer->nFilterCoefficients = component[i].op.conv1D.num_filter_coefficients;
                    pConvolutionalLayer->nFeatureMaps = component[i].op.conv1D.num_feature_maps;
                    pConvolutionalLayer->nFeatureMapRows = component[i].op.conv1D.num_feature_map_rows;
                    pConvolutionalLayer->nFeatureMapColumns = component[i].op.conv1D.num_feature_map_columns;
                    pConvolutionalLayer->poolType = INTEL_NO_POOLING;  //  will be overwritten
                    pConvolutionalLayer->nPoolSize = 0;  //  will be overwritten
                    pConvolutionalLayer->nPoolStride = 0;  //  will be overwritten
                    pConvolutionalLayer->pwl.nSegments = 0;  //  will be overwritten
                    pConvolutionalLayer->pwl.pSegments = nullptr;  //  will be overwritten
                    pConvolutionalLayer->pBiases = component[i].op.conv1D.ptr_biases;
                    pConvolutionalLayer->pFilters = component[i].op.conv1D.ptr_filters;
                }
                if (i == component.size() - 1 || ((component[i + 1].operation != kDnnMaxPoolOp)
                        && (component[i + 1].operation != kDnnPiecewiselinearOp))) {
                    pLayer++;
                }
                break;
            case kDnnMaxPoolOp:
                if (i == 0) {
                    THROW_GNA_EXCEPTION << "Pooling component with no preceeding component";
                } else if (pLayer->nLayerKind == INTEL_CONVOLUTIONAL) {
                    if (pLayer->pLayerStruct == nullptr) {
                        THROW_GNA_EXCEPTION "INTEL_CONVOLUTIONAL layer structure was not initialized.";
                    }
                    auto pConvolutionalLayer = reinterpret_cast<intel_convolutional_layer_t *>(pLayer->pLayerStruct);
                    // it is possible to have activation preceding to maxpool
                    if (pConvolutionalLayer->pwl.nSegments != 0) {
                        THROW_GNA_EXCEPTION << "Encountered activation component before pooling component at." << i;
                    } else {
                        pConvolutionalLayer->poolType =
                            (component[i].op.maxpool.do_sum_not_max) ? INTEL_SUM_POOLING : INTEL_MAX_POOLING;
                        pConvolutionalLayer->nPoolSize = component[i].op.maxpool.num_inputs;
                        pConvolutionalLayer->nPoolStride = component[i].op.maxpool.num_inputs_step;


                        // number of output columns correction - based on GNA-library expectations
                        auto nFltSize = pConvolutionalLayer->nFilterCoefficients;
                        auto fltStrideSz = pConvolutionalLayer->nFeatureMaps * pConvolutionalLayer->nFeatureMapColumns;  // always move 1 "row"
                        auto maxNCOE = (pLayer->nInputColumns - nFltSize) / fltStrideSz + 1;
                        // FLAT input matrix, pooled outputs per filter
                        pLayer->nOutputColumns = pConvolutionalLayer->nFilters * ((maxNCOE - 1) / pConvolutionalLayer->nPoolStride + 1);

                        // old code
                        // pLayer->nOutputColumns /= pConvolutionalLayer->nPoolStride;
                    }
                } else {
                    THROW_GNA_EXCEPTION << "Pooling component applied to non-convolutional layer";
                }
                break;
            case kDnnPiecewiselinearOp:
                pLayer->pOutputs = component[i].ptr_outputs;
                pLayer->nBytesPerOutput = component[i].num_bytes_per_output;
                if (pLayer->pLayerStruct == nullptr) {
                    THROW_GNA_EXCEPTION << pLayer->nLayerKind << " layer structure was not initialized.";
                }
                if (i == 0) {
                    THROW_GNA_EXCEPTION << "PWL component with no preceding component.";
                } else if ((component[i - 1].operation == kDnnAffineOp)
                    || (component[i - 1].operation == kDnnDiagonalOp)) {
                    auto pAffineLayer = reinterpret_cast<intel_affine_layer_t *>(pLayer->pLayerStruct);
                    pAffineLayer->pwl.nSegments = component[i].op.pwl.num_segments;
                    pAffineLayer->pwl.pSegments = component[i].op.pwl.ptr_segments;
                } else if (component[i - 1].operation == kDnnRecurrentOp) {
                    auto pRecurrentLayer = reinterpret_cast<intel_recurrent_layer_t *>(pLayer->pLayerStruct);
                    pRecurrentLayer->pwl.nSegments = component[i].op.pwl.num_segments;
                    pRecurrentLayer->pwl.pSegments = component[i].op.pwl.ptr_segments;
                } else if ((component[i - 1].operation == kDnnConvolutional1dOp)
                    || ((component[i - 1].operation == kDnnMaxPoolOp)
                        && (component[i - 2].operation == kDnnConvolutional1dOp))) {
                    auto pConvolutionalLayer = reinterpret_cast<intel_convolutional_layer_t *>(pLayer->pLayerStruct);
                    pConvolutionalLayer->pwl.nSegments = component[i].op.pwl.num_segments;
                    pConvolutionalLayer->pwl.pSegments = component[i].op.pwl.ptr_segments;
                    if (component[i - 1].operation != kDnnMaxPoolOp) {
                        pLayer->nOutputColumns = component[i].num_columns_out;
                    }
                }
                pLayer++;

                break;
            case kDnnInterleaveOp:
                pLayer->nInputRows = component[i].num_rows_in;
                pLayer->nInputColumns = component[i].num_columns_in;
                pLayer->nOutputRows = component[i].num_rows_out;
                pLayer->nOutputColumns = component[i].num_columns_out;
                pLayer->nBytesPerInput = component[i].num_bytes_per_input;
                pLayer->nBytesPerOutput = component[i].num_bytes_per_output;
                pLayer->nBytesPerIntermediateOutput = sizeof(int32_t);
                pLayer->pInputs = component[i].ptr_inputs;
                pLayer->pOutputsIntermediate = nullptr;
                pLayer->pOutputs = component[i].ptr_outputs;
                pLayer->nLayerKind = INTEL_INTERLEAVE;
                pLayer->pLayerStruct = nullptr;
                pLayer++;
                break;
            case kDnnDeinterleaveOp:
                pLayer->nInputRows = component[i].num_rows_in;
                pLayer->nInputColumns = component[i].num_columns_in;
                pLayer->nOutputRows = component[i].num_rows_out;
                pLayer->nOutputColumns = component[i].num_columns_out;
                pLayer->nBytesPerInput = component[i].num_bytes_per_input;
                pLayer->nBytesPerOutput = component[i].num_bytes_per_output;
                pLayer->nBytesPerIntermediateOutput = sizeof(int32_t);
                pLayer->pInputs = component[i].ptr_inputs;
                pLayer->pOutputsIntermediate = nullptr;
                pLayer->pOutputs = component[i].ptr_outputs;
                pLayer->nLayerKind = INTEL_DEINTERLEAVE;
                pLayer->pLayerStruct = nullptr;
                pLayer++;
                break;
            case kDnnCopyOp:
                pLayer->nInputRows = component[i].num_columns_in;
                pLayer->nInputColumns = component[i].num_rows_in;
                pLayer->nOutputRows = component[i].num_columns_out;
                pLayer->nOutputColumns = component[i].num_rows_out;
                pLayer->nBytesPerInput = component[i].num_bytes_per_input;
                pLayer->nBytesPerOutput = component[i].num_bytes_per_output;
                pLayer->nBytesPerIntermediateOutput = sizeof(int32_t);
                pLayer->pInputs = component[i].ptr_inputs;
                pLayer->pOutputsIntermediate = nullptr;
                pLayer->pOutputs = component[i].ptr_outputs;
                pLayer->nLayerKind = INTEL_COPY;
                pLayer->pLayerStruct = nullptr;
                {
                    pLayer->pLayerStruct = _mm_malloc(sizeof(intel_copy_layer_t), 64);
                    if (pLayer->pLayerStruct == nullptr) {
                        THROW_GNA_EXCEPTION << pLayer->nLayerKind << " could not allocate memory for INTEL_COPY layer structure.";
                    }
                    auto *pCopyLayer = reinterpret_cast<intel_copy_layer_t *>(pLayer->pLayerStruct);
                    pCopyLayer->nCopyRows = component[i].op.copy.num_copy_columns;
                    pCopyLayer->nCopyCols = component[i].op.copy.num_copy_rows;
                }
                pLayer++;
                break;
            default: {
                THROW_GNA_EXCEPTION << "GNA does yet not support " << intel_dnn_operation_name[component[i].operation];
            }
        }
    }
    // enable debugging of partial array of components
    ptr_nnet->nLayers = std::distance(ptr_nnet->pLayers, pLayer);
}

void AmIntelDnn::DestroyGNAStruct(intel_nnet_type_t *ptr_nnet) {
    ptr_nnet->nGroup = 0;
    if (ptr_nnet->pLayers != nullptr) {
        for (int i = 0; i < ptr_nnet->nLayers; i++) {
            switch (ptr_nnet->pLayers[i].nLayerKind) {
                case INTEL_AFFINE:break;
                case INTEL_AFFINE_DIAGONAL:break;
                case INTEL_RECURRENT:break;
                case INTEL_CONVOLUTIONAL:break;
                case INTEL_INTERLEAVE:break;
                case INTEL_DEINTERLEAVE:break;
                case INTEL_COPY:break;
                default:break;
            }
            if (ptr_nnet->pLayers[i].pLayerStruct != nullptr) {
                _mm_free(ptr_nnet->pLayers[i].pLayerStruct);
            }
        }
        if (ptr_nnet->pLayers != nullptr) {
            _mm_free(ptr_nnet->pLayers);
        }
    }
    ptr_nnet->nLayers = 0;
}

void AmIntelDnn::GetScaledOutput(float *ptr_output, uint32_t component_index) {
    if (component_index > num_components()) {
        fprintf(stderr, "Illegal component index %d in GetScaledOutput\n", component_index);
        throw -1;
    }
    if (ptr_output != nullptr) {
        float scale_factor = OutputScaleFactor(component_index);
        uint32_t num_elements = component[component_index].num_rows_out * component[component_index].num_columns_out;
        if (number_type_ == kDnnFloat) {
            float *ptr_input = reinterpret_cast<float *>(component[component_index].ptr_outputs);
            for (uint32_t i = 0; i < num_elements; i++) {
                ptr_output[i] = ptr_input[i] / scale_factor;
            }
        } else if (component[component_index].num_bytes_per_output == 2) {
            int16_t *ptr_input = reinterpret_cast<int16_t *>(component[component_index].ptr_outputs);
            for (uint32_t i = 0; i < num_elements; i++) {
                ptr_output[i] = static_cast<float>(ptr_input[i]) / scale_factor;
            }
        } else {
            int32_t *ptr_input = reinterpret_cast<int32_t *>(component[component_index].ptr_outputs);
            for (uint32_t i = 0; i < num_elements; i++) {
                ptr_output[i] = static_cast<float>(ptr_input[i]) / scale_factor;
            }
        }
    } else {
        fprintf(stderr, "Output pointer is nullptr in GetScaledOutput\n");
        throw -1;
    }
}

void AmIntelDnn::WriteInputAndOutputTextGNA(intel_nnet_type_t * nnet) {
#ifdef LIGHT_DUMP
    if (nnet) {
        for (int i = 0; i < nnet->nLayers; i++) {
            auto component = nnet->pLayers;
            std::stringstream out_file_name;
            auto getLayerType = [](intel_layer_kind_t kind){
                switch (kind){
                    case INTEL_AFFINE : return "affine";
                    case INTEL_AFFINE_DIAGONAL : return "diag";
                    case INTEL_RECURRENT : return "recurrent";
                    case INTEL_CONVOLUTIONAL : return "convolution";
                    case INTEL_INTERLEAVE : return "interleave";
                    case INTEL_DEINTERLEAVE : return "deinterleave";
                    case INTEL_COPY : return "copy";
                    default: return "unknown";
                }
            };
            out_file_name << std::setfill('0') << std::setw(2) << i << "_"
                          << getLayerType(component[i].nLayerKind)
                          << "-" << nnet->pLayers[i].nInputRows
                          << "-" << nnet->pLayers[i].nOutputRows;

            auto inputfileName = getDumpFolderNameGNA() + out_file_name.str() + "_input.txt";
            auto outFileName = getDumpFolderNameGNA() + out_file_name.str() + "_output.txt";
            auto pwlFileName = getDumpFolderNameGNA() + out_file_name.str() + "_pwl.txt";
            auto refOutputFileName = getRefFolderName() + out_file_name.str() + "_output.txt";

            std::ofstream out_file(outFileName.c_str(), std::ios::out);
            std::ofstream pwl_file(pwlFileName.c_str(), std::ios::out);
            std::ifstream ref_out_file(refOutputFileName.c_str(), std::ios::in);
            std::ofstream in_file(inputfileName.c_str(), std::ios::out);

            float  summOfDiff = 0.f;
            float  summOfSqDiff = 0.f;
            float  maxD = 0.0f;
            int    numItems = 0;

            auto write_pwl = [&pwl_file](intel_pwl_func_t & pwl) {
                for (int k =0; k < pwl.nSegments; k++) {
                    pwl_file << pwl.pSegments[k].slope << ", " << pwl.pSegments[k].xBase << ", " << pwl.pSegments[k].yBase << "\n";
                }
            };
            if (nnet->pLayers[i].nLayerKind == INTEL_AFFINE || nnet->pLayers[i].nLayerKind == INTEL_AFFINE_DIAGONAL) {
                auto affine = reinterpret_cast<intel_affine_layer_t*>(nnet->pLayers[i].pLayerStruct);
                write_pwl(affine->pwl);
            }
            if (nnet->pLayers[i].nLayerKind == INTEL_CONVOLUTIONAL) {
                auto conv = reinterpret_cast<intel_convolutional_layer_t*>(nnet->pLayers[i].pLayerStruct);
                write_pwl(conv->pwl);
            }

            for (int k = 0; k < component[i].nOutputRows; k++) {
                for (int j = 0; j < component[i].nOutputColumns; j++) {
                    float floatValue = 0.f;
                    if (component[i].nBytesPerOutput == 4) {
                        auto value = (reinterpret_cast<int32_t *>(component[i].pOutputs)[k * component[i].nOutputColumns + j]);
                        floatValue = (static_cast<float>(value) / 1.0);
                    } else {
                        auto value = reinterpret_cast<int16_t *>(component[i].pOutputs)[k * component[i].nOutputColumns + j];
                        floatValue = (static_cast<float>(value) / 1.0);
                    }
                    out_file << std::setw(8) << floatValue << "\n";
                    if (ref_out_file) {
                        float ref_value = 0.f;
                        ref_out_file >> ref_value;
                        float diff = (ref_value - floatValue);
                        diff = diff  < 0 ? -diff : diff;
                        summOfDiff += diff;
                        summOfSqDiff += diff * diff;
                        maxD = std::max(maxD, diff);
                        numItems++;
                    }
                }
            }
            if (numItems) {
                auto rmse = sqrt(summOfSqDiff / numItems);
                auto avg = summOfDiff / numItems;
                std :: cout << std::left << std::setw(55) << out_file_name.str()
                            << " RMSE="<< std::fixed << std::setprecision(5) << std::right << std::setw(8) << rmse
                            << " avg=" << std::fixed << std::setprecision(5) << std::right << std::setw(8) << avg
                            << " maxD="<< std::fixed << std::setprecision(5) << std::right << std::setw(8) << maxD << std::endl;
            }


            for (int k = 0; k < component[i].nInputRows; k++) {
                for (int j = 0; j < component[i].nInputColumns; j++) {
                    if (component[i].nBytesPerInput == 4) {
                        in_file << std::setw(8)
                                << (reinterpret_cast<int32_t *>(component[i].pInputs)[k * component[i].nInputColumns + j]);
                    } else {
                        in_file << std::setw(8)
                                << (reinterpret_cast<int16_t *>(component[i].pInputs)[k * component[i].nInputColumns + j]);
                    }
                    in_file << "\n";
                }
            }
        }
    }
#endif
}

void AmIntelDnn::WriteInputAndOutputText() {
#ifdef LIGHT_DUMP
    for (int i = 0; i < num_components(); i++) {
        std::stringstream out_file_name;
        out_file_name << std::setfill('0') << std::setw(2) << i << "_"
                      << intel_dnn_operation_name[component[i].operation]
                      << "-" << component[i].num_rows_in
                      << "-" << component[i].num_rows_out;
        if (component[i].operation == kDnnPiecewiselinearOp) {
            out_file_name << "-" << intel_dnn_activation_name[component[i].op.pwl.func_id];
        }
        auto inputfileName = getDumpFolderName() + out_file_name.str() + "_input.txt";
        auto outFileName = getDumpFolderName() + out_file_name.str() + "_output.txt";
        auto refOutputFileName = getRefFolderName() + out_file_name.str() + "_output.txt";

        std::ofstream out_file(outFileName.c_str(), std::ios::out);
        std::ifstream ref_out_file(refOutputFileName.c_str(), std::ios::in);
        std::ofstream in_file(inputfileName.c_str(), std::ios::out);

        float  summOfDiff = 0.f;
        float  summOfSqDiff = 0.f;
        float  maxD = 0.0f;
        int    numItems = 0;

        for (int k = 0; k < component[i].num_rows_out; k++) {
            for (int j = 0; j < component[i].num_columns_out; j++) {
                float floatValue = 0.f;
                if (component[i].num_bytes_per_output == 4) {
                    if (number_type_ == kDnnInt) {
                        auto value = reinterpret_cast<int32_t *>(component[i].ptr_outputs)[k * component[i].num_columns_out+ j];
                        floatValue = static_cast<float>(value);

                    } else {
                        floatValue = reinterpret_cast<float*>(component[i].ptr_outputs)[k * component[i].num_columns_out+ j];
                    }
                } else {
                    auto value = reinterpret_cast<int16_t *>(component[i].ptr_outputs)[k * component[i].num_columns_out+ j];
                    floatValue = static_cast<float>(value);
                }
                out_file << std::setw(8) << floatValue / component[i].output_scale_factor << "\n";

                if (ref_out_file) {
                    float ref_value = 0.f;
                    ref_out_file >> ref_value;
                    float diff = (ref_value - floatValue);
                    diff = diff < 0.f ? -diff : diff;
                    summOfDiff += diff;
                    summOfSqDiff += diff * diff;
                    maxD = std::max(maxD, diff);
                    numItems++;
                }
            }
        }
        if (numItems) {
            auto rmse = sqrt(summOfSqDiff / numItems);
            auto avg = summOfDiff / numItems;
            std :: cout << std::left << std::setw(55) << out_file_name.str()
                        << " RMSE="<< std::fixed << std::setprecision(5) << std::right << std::setw(8) << rmse
                        << " avg=" << std::fixed << std::setprecision(5) << std::right << std::setw(8) << avg
                        << " maxD="<< std::fixed << std::setprecision(5) << std::right << std::setw(8) << maxD << std::endl;
        }

        float input_scale_factor = component[i].output_scale_factor;
        if (component[i].operation == kDnnAffineOp ||
            component[i].operation == kDnnDiagonalOp) {
            input_scale_factor /= component[i].op.affine.weight_scale_factor;
        } else if (component[i].operation == kDnnConvolutional1dOp) {
            input_scale_factor /= component[i].op.conv1D.weight_scale_factor;
        } else if (component[i].operation == kDnnPiecewiselinearOp) {
            input_scale_factor = 1.f;
        }

        for (int k = 0; k < component[i].num_rows_in; k++) {
            for (int j = 0; j < component[i].num_columns_in; j++) {
                float floatValue = 0.f;
                if (component[i].num_bytes_per_input == 4) {
                    if (number_type_ == kDnnInt) {
                        auto value = reinterpret_cast<int32_t *>(component[i].ptr_inputs)[k * component[i].num_columns_in + j];
                        floatValue = static_cast<float>(value);
                    } else {
                        floatValue = reinterpret_cast<float *>(component[i].ptr_inputs)[k * component[i].num_columns_in + j];
                    }
                } else {
                    auto value = reinterpret_cast<int16_t *>(component[i].ptr_inputs)[k * component[i].num_columns_in+ j];
                    floatValue = static_cast<float>(value);
                }
                in_file << std::setw(8) << floatValue / input_scale_factor << "\n";
            }
        }
#endif
    }
}

bool isCompatibleDnn(AmIntelDnn dnn1, AmIntelDnn dnn2) {
    bool isCompatible = true;

    // compare basic structures to see if they are compatible
    if (dnn1.num_components() != dnn2.num_components()) isCompatible = false;
    for (int i = 0; i < dnn1.num_components(); i++) {
        if (dnn1.component[i].num_rows_in != dnn2.component[i].num_rows_in) isCompatible = false;
        if (dnn1.component[i].num_columns_in != dnn2.component[i].num_columns_in) isCompatible = false;
        if (dnn1.component[i].num_rows_out != dnn2.component[i].num_rows_out) isCompatible = false;
        if (dnn1.component[i].num_columns_out != dnn2.component[i].num_columns_out) isCompatible = false;
        if (dnn1.component[i].operation != dnn2.component[i].operation) isCompatible = false;
    }

    return (isCompatible);
}

void ClearScoreError(intel_score_error_t *error) {
    error->num_scores = 0;
    error->num_errors = 0;
    error->max_error = 0.0;
    error->sum_error = 0.0;
    error->sum_squared_error = 0.0;
    error->max_rel_error = 0.0;
    error->sum_rel_error = 0.0;
    error->sum_squared_rel_error = 0.0;
}

void UpdateScoreError(intel_score_error_t *error, intel_score_error_t *total_error) {
    total_error->num_errors += error->num_errors;
    total_error->num_scores += error->num_scores;
    total_error->sum_error += error->sum_error;
    total_error->sum_squared_error += error->sum_squared_error;
    if (error->max_error > total_error->max_error) {
        total_error->max_error = error->max_error;
    }
    total_error->sum_rel_error += error->sum_rel_error;
    total_error->sum_squared_rel_error += error->sum_squared_rel_error;
    if (error->max_rel_error > total_error->max_rel_error) {
        total_error->max_rel_error = error->max_rel_error;
    }
}

void SoftmaxGoogle(float *ptr_output, float *ptr_input, const uint32_t num_outputs, const uint32_t num_inputs) {
    // Assumes input vector contains log likelihoods
    // The computes x[i] = x[i] - log(sum_j exp(x[j]))
    // This normalizes the likelihoods by the sum of likelihoods but stores them as log likelihoods

    float max_score = ptr_input[0];
    float sum = 0.0;
    float diff;
    // find max score for normalization to [0,1]
    for (uint32_t i = 0; i < num_inputs; i++) {
        if (ptr_input[i] > max_score) {
            max_score = ptr_input[i];
        }
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        sum += exp(ptr_input[i] - max_score);
    }
    if (sum < 1.0e-20) {
        fprintf(stderr, "Warning:  attempt to take log(0) in SoftmaxGoogle()!\n");
        sum = 1.0e-20;
    }
    diff = max_score + log(sum);
    for (uint32_t i = 0; i < num_outputs; i++) {
        ptr_output[i] = ptr_input[i] - diff;
    }
}

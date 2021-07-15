// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dnn_types.h"
#include "gna_types.h"

#include "gna_plugin_log.hpp"

#if GNA_LIB_VER == 2
#include <gna2-model-api.h>
#include <gna/gna_config.hpp>
#endif

namespace GNAPluginNS {
namespace backend {

class AMIntelDNN {
public:
    AMIntelDNN()
            : ptr_active_outputs_(NULL),
              num_active_outputs_(0),
              input_scale_factor_(1.0),
              num_left_context(0),
              num_right_context(0),
              do_rotate_input(false),
              do_rotate_output(false),
              num_rotate_rows(0),
              num_rotate_columns(0),
              num_rotate_output_rows(0),
              num_rotate_output_columns(0),
              softmax_type(kSoftmaxNone),
              ptr_sumgroup_sizes(NULL),
              num_sumgroup_sizes(0),
              ptr_priors(NULL),
              ptr_dnn_memory_(NULL),
              num_bytes_dnn_memory_(0),
              compute_precision_(kDnnNumNumberType) {
    }

    ~AMIntelDNN();

    void Init(void *ptr_memory,
            uint32_t num_memory_bytes,
            intel_dnn_number_type_t compute_precision,
            float scale_factor);

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
    static void InitConvolutional1DComponent(intel_dnn_component_t &comp,
                                             uint32_t num_columns_in,
                                             uint32_t num_columns_out,
                                             uint32_t num_bytes_per_input,
                                             uint32_t num_bytes_per_output,
                                             uint32_t num_bytes_per_weight,
                                             uint32_t num_bytes_per_bias,
                                             uint32_t num_filters,
                                             uint32_t num_filter_coefficients,
                                             uint32_t convStride,
                                             float weight_scale_factor,
                                             float output_scale_factor,
                                             A *&ptr_inputs,
                                             B *&ptr_outputs,
                                             C *&ptr_filters,
                                             D *&ptr_biases) {
        InitConvolutional1DComponentPrivate(comp,
                                            num_columns_in,
                                            num_columns_out,
                                            num_bytes_per_input,
                                            num_bytes_per_output,
                                            num_bytes_per_weight,
                                            num_bytes_per_bias,
                                            num_filters,
                                            num_filter_coefficients,
                                            convStride,
                                            weight_scale_factor,
                                            output_scale_factor,
                                            (void *&) ptr_inputs,
                                            (void *&) ptr_outputs,
                                            (void *&) ptr_filters,
                                            (void *&) ptr_biases,
                                            true);
    }

#if GNA_LIB_VER == 2
    template<class A, class B, class C, class D>
    static void InitConvolutional2DComponent(intel_dnn_component_t& comp,
        OvGnaTensor inputTensor,
        OvGnaTensor outputTensor,
        OvGnaTensor filterTensor,
        OvGnaTensor biasTensor,
        std::array<uint32_t, 2> convStride,
        std::array<uint32_t, 2> zeroPadding,
        float weight_scale_factor,
        float output_scale_factor,
        A*& ptr_inputs,
        B*& ptr_outputs,
        C*& ptr_filters,
        D*& ptr_biases) {
        InitConvolutional2DComponentPrivate(comp,
            inputTensor,
            outputTensor,
            filterTensor,
            biasTensor,
            convStride,
            zeroPadding,
            weight_scale_factor,
            output_scale_factor,
            (void*&)ptr_inputs,
            (void*&)ptr_outputs,
            (void*&)ptr_filters,
            (void*&)ptr_biases);
    }
#endif

    template<class A, class B>
    static void InitMaxpoolComponent(intel_dnn_component_t &cmp,
                                     std::array<uint32_t, 3> inCHW,
                                     std::array<uint32_t, 3> outCHW,
                                     uint32_t num_bytes_per_input,
                                     uint32_t num_bytes_per_output,
                                     std::array<uint32_t, 2> poolingWindowXY,
                                     std::array<uint32_t, 2> poolingStrideXY,
                                     float output_scale_factor,
                                     A *&ptr_inputs,
                                     B *&ptr_outputs) {
        InitMaxpoolComponentPrivate(cmp,
                                    inCHW,
                                    outCHW,
                                    num_bytes_per_input,
                                    num_bytes_per_output,
                                    poolingWindowXY,
                                    poolingStrideXY,
                                    output_scale_factor,
                                    (void *&) ptr_inputs,
                                    (void *&) ptr_outputs,
                                    true);
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
                                             float input_scale_factor,
                                             A *&ptr_inputs,
                                             B *&ptr_outputs,
                                             gna_pwl_segment_t *ptr_segments) {
        InitPiecewiseLinearComponentPrivate(cmp,
                                            function_id,
                                            orientation,
                                            num_rows,
                                            num_columns,
                                            num_bytes_per_input,
                                            num_bytes_per_output,
                                            num_segments,
                                            output_scale_factor,
                                            input_scale_factor,
                                            (void *&) ptr_inputs,
                                            (void *&) ptr_outputs,
                                            ptr_segments,
                                            true);
    }

    template<class A, class B>
    static void InitDeinterleaveComponent(intel_dnn_component_t &cmp,
                                          uint32_t num_rows_in,
                                          uint32_t num_columns_in,
                                          uint32_t num_bytes_per_input,
                                          uint32_t num_bytes_per_output,
                                          float output_scale_factor,
                                          A *&ptr_inputs,
                                          B *&ptr_outputs) {
        InitDeinterleaveComponentPrivate(cmp,
                                         num_rows_in,
                                         num_columns_in,
                                         num_bytes_per_input,
                                         num_bytes_per_output,
                                         output_scale_factor,
                                         (void *&) ptr_inputs,
                                         (void *&) ptr_outputs,
                                         true);
    }

    template<class A, class B>
    static void InitInterleaveComponent(intel_dnn_component_t &cmp,
                                        uint32_t num_rows_in,
                                        uint32_t num_columns_in,
                                        uint32_t num_bytes_per_input,
                                        uint32_t num_bytes_per_output,
                                        float output_scale_factor,
                                        A *&ptr_inputs,
                                        B *&ptr_outputs) {
        InitInterleaveComponentPrivate(cmp,
                                       num_rows_in,
                                       num_columns_in,
                                       num_bytes_per_input,
                                       num_bytes_per_output,
                                       output_scale_factor,
                                       (void *&) ptr_inputs,
                                       (void *&) ptr_outputs,
                                       true);
    }


    template<class A, class B>
    static void InitCopyComponent(intel_dnn_component_t &cmp,
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


    float OutputScaleFactor(uint32_t component_index) {
        return OutputScaleFactor(component[component_index]);
    }

    float OutputScaleFactor(intel_dnn_component_t &comp);

    void WriteGraphWizModel(const char *filename);

    void WriteDnnText(const char *filename, intel_dnn_number_type_t logging_precision);


#if GNA_LIB_VER == 2
    void InitGNAStruct(Gna2Model *gnaModel, const std::string& gnaCompileTarget = InferenceEngine::GNAConfigParams::GNA_TARGET_2_0);
    void DestroyGNAStruct(Gna2Model *gnaModel);
#else

    void InitGNAStruct(intel_nnet_type_t *ptr_nnet);

    void DestroyGNAStruct(intel_nnet_type_t *ptr_nnet);

#endif

    uint32_t *ptr_active_outputs() { return (ptr_active_outputs_); }

    uint32_t num_active_outputs() { return (num_active_outputs_); }

    uint32_t num_components();

    uint32_t num_gna_layers();

    uint32_t num_group_in();

    uint32_t num_group_out();

    uint32_t num_inputs();

    uint32_t num_outputs();

    std::vector<intel_dnn_component_t> component;
    uint32_t num_left_context;
    uint32_t num_right_context;
    uint32_t new_num_conv_columns = 0;
    bool do_rotate_input;
    bool do_rotate_output;
    uint32_t num_rotate_rows = 0;
    uint32_t num_rotate_columns = 0;
    uint32_t num_rotate_output_rows = 0;
    uint32_t num_rotate_output_columns = 0;
    DnnSoftmaxType softmax_type;
    uint32_t *ptr_sumgroup_sizes;
    uint32_t num_sumgroup_sizes;
    float *ptr_priors;

    void WriteInputAndOutputText();

#if GNA_LIB_VER == 1
    void WriteInputAndOutputTextGNA(intel_nnet_type_t *nnet);
#else
    void WriteInputAndOutputTextGNA(const Gna2Model & model);
#endif

    void BeginNewWrite(uint32_t index);

private:
    void *ptr_dnn_memory_;
    uint32_t num_bytes_dnn_memory_;
    uint32_t *ptr_active_outputs_;
    uint32_t num_active_outputs_;
    intel_dnn_number_type_t compute_precision_;
    float input_scale_factor_;
    uint32_t dump_write_index = 0;

    uint32_t CountLayers();

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
                                            std::array<uint32_t, 3> inCHW,
                                            std::array<uint32_t, 3> outCHW,
                                            uint32_t num_bytes_per_input,
                                            uint32_t num_bytes_per_output,
                                            std::array<uint32_t, 2> poolingWindowXY,
                                            std::array<uint32_t, 2> poolingStrideXY,
                                            float output_scale_factor,
                                            void *&ptr_inputs,
                                            void *&ptr_outputs,
                                            bool postInitMem);

    static void InitPiecewiseLinearComponentPrivate(intel_dnn_component_t &cmp,
                                                    DnnActivation function_id,
                                                    intel_dnn_orientation_t orientation,
                                                    uint32_t num_rows,
                                                    uint32_t num_columns,
                                                    uint32_t num_bytes_per_input,
                                                    uint32_t num_bytes_per_output,
                                                    uint32_t num_segments,
                                                    float output_scale_factor,
                                                    float input_scale_factor,
                                                    void *&ptr_inputs,
                                                    void *&ptr_outputs,
                                                    gna_pwl_segment_t *ptr_segments,
                                                    bool postInitMem);

    static void InitInterleaveComponentPrivate(intel_dnn_component_t &cmp,
                                                    uint32_t num_rows_in,
                                                    uint32_t num_columns_in,
                                                    uint32_t num_bytes_per_input,
                                                    uint32_t num_bytes_per_output,
                                                    float output_scale_factor,
                                                    void *&ptr_inputs,
                                                    void *&ptr_outputs,
                                                    bool postInitMem);

    static void InitDeinterleaveComponentPrivate(intel_dnn_component_t &cmp,
                                                    uint32_t num_rows_in,
                                                    uint32_t num_columns_in,
                                                    uint32_t num_bytes_per_input,
                                                    uint32_t num_bytes_per_output,
                                                    float output_scale_factor,
                                                    void *&ptr_inputs,
                                                    void *&ptr_outputs,
                                                    bool postInitMem);

    static void InitConvolutional1DComponentPrivate(intel_dnn_component_t &comp,
                                                    uint32_t num_columns_in,
                                                    uint32_t num_columns_out,
                                                    uint32_t num_bytes_per_input,
                                                    uint32_t num_bytes_per_output,
                                                    uint32_t num_bytes_per_weight,
                                                    uint32_t num_bytes_per_bias,
                                                    uint32_t num_filters,
                                                    uint32_t num_filter_coefficients,
                                                    uint32_t convStride,
                                                    float weight_scale_factor,
                                                    float output_scale_factor,
                                                    void *&ptr_inputs,
                                                    void *&ptr_outputs,
                                                    void *&ptr_filters,
                                                    void *&ptr_biases,
                                                    bool postInitMem);

#if GNA_LIB_VER == 2
    static void InitConvolutional2DComponentPrivate(intel_dnn_component_t& comp,
        OvGnaTensor inputTensor,
        OvGnaTensor outputTensor,
        OvGnaTensor filterTensor,
        OvGnaTensor biasTensor,
        std::array<uint32_t, 2> convStride,
        std::array<uint32_t, 2> zeroPadding,
        float weight_scale_factor,
        float output_scale_factor,
        void*& ptr_inputs,
        void*& ptr_outputs,
        void*& ptr_filters,
        void*& ptr_biases);
#endif

    static void InitAffineComponentPrivate(intel_dnn_component_t &comp,
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
                                           bool postInitMem);

    std::string getDumpFilePrefix(const std::string& folder);
    std::string getDumpFilePrefixGNA();
    std::string getDumpFolderName();
    std::string getRefFolderName();
};
}  // namespace backend
}  // namespace GNAPluginNS

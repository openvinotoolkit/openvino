// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

/** PWL Segment - as read directly by the accelerator */
typedef struct _pwl_segment_t {
    int32_t xBase;                  // X Component of segment starting point, with scaling encoded if needed.
    int16_t yBase;                  // Y Component of segment starting point.
    int16_t slope;                  // Slope of linear function.
} gna_pwl_segment_t;
static_assert(8 == sizeof(gna_pwl_segment_t), "Invalid size of gna_pwl_segment_t");

/** Piecewise-linear activation function (PWL) details */
typedef struct _pwl_func_t {
    uint32_t nSegments;             // Number of segments, set to 0 to disable activation function.
    gna_pwl_segment_t* pSegments; // Activation function segments data or NULL if disabled.
} gna_pwl_func_t;

/**
 * Compound bias
 * Used for nBytesPerWeight=GNA_INT8 and nBytesPerBias=GNA_INT16 only.
 * As read directly by the accelerator.
 */
typedef struct _compound_bias_t {
    int32_t bias;              // 4B Signed integer bias (constant) value.
    uint8_t multiplier;             // Scaling factor that weight elements are multiplied by.
    uint8_t reserved[3];            // Not used.
} gna_compound_bias_t;
static_assert(8 == sizeof(gna_compound_bias_t), "Invalid size of gna_compound_bias_t");

/**
 * Layer operation type.
 * Defines type of layer "core" operation.
 * All nodes/cells within a layer are of the same type,
 * e.g. affine transform cell, convolutional cell, recurrent cell.
 * Affine, convolutional and recurrent layers are in fact "fused operation" layers
 * and "core" operation is fused with activation and/or pooling functions.
 * NOTE: Operation types are exclusive.
 */
typedef enum _layer_operation {
    // Fully connected affine transform (deep feed forward) with activation function. Cast pLayerStruct to intel_affine_layer_t.
    INTEL_AFFINE,
    // Fully connected affine transform (matrix x vector) (deep feed forward) with activation function.Cast pLayerStruct to intel_affine_layer_t.
    INTEL_AFFINE_DIAGONAL,
    /*
     * Fully connected affine transform (with grouped bias vectors) (deep feed forward) with activation function.
     * Cast pLayerStruct to intel_affine_multibias_layer_t.
     */
    INTEL_AFFINE_MULTIBIAS,
    INTEL_CONVOLUTIONAL,            // Convolutional transform with activation function and pooling. Cast pLayerStruct to intel_convolutional_layer_t.
    INTEL_CONVOLUTIONAL_2D,         // Convolutional transform with activation function and pooling. Cast pLayerStruct to nn_layer_cnn2d.
    INTEL_COPY,                     // Auxiliary data copy operation. Cast pLayerStruct to intel_copy_layer_t.
    INTEL_DEINTERLEAVE,             // Auxiliary 2D tensor transpose operation (interleave to flat). No casting, always set pLayerStruct to null.
    INTEL_GMM,                      // Gaussian Mixture Model operation. Cast pLayerStruct to intel_gmm_layer_t.
    INTEL_INTERLEAVE,               // Auxiliary 2D tensor transpose operation (flat to interleave). No casting, always set pLayerStruct to null.
    INTEL_RECURRENT,                // Fully connected affine transform with recurrence and activation function. Cast pLayerStruct to intel_recurrent_layer_t.
    GNA_LAYER_CNN_2D_POOLING,
    LAYER_OPERATION_TYPE_COUT,
} gna_layer_operation;

typedef enum _layer_mode {
    INTEL_INPUT,            // Layer serves as model input layer (usually first layer)
    INTEL_OUTPUT,           // Layer serves as model output layer (usually last layer)
    INTEL_INPUT_OUTPUT,     // Layer serves as model input nad output layer (usually in single layer topology)
    INTEL_HIDDEN,           // Layer serves as model hidden layer (layers between input and output layers)
    LAYER_MODE_COUNT        // Number of Layer modes.
} gna_layer_mode;

/** Layer common configuration descriptor */
typedef struct _nnet_layer_t {
    gna_layer_operation operation;  // Layer operation type.
    gna_layer_mode mode;            // Layer connection mode.
    uint32_t nInputColumns;         // Number of input columns.
    uint32_t nInputRows;            // Number of input rows.
    uint32_t nOutputColumns;        // Number of output columns.
    uint32_t nOutputRows;           // Number of output rows.
    uint32_t nBytesPerInput;        // Precision/mode of input node, use a value from gna_data_mode. Valid values {GNA_INT8, GNA_INT16, GNA_DATA_DISABLED}
    // Precision/ activation mode of output node, use a value from gna_data_mode. Valid values {GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}
    uint32_t nBytesPerOutput;
    uint32_t nBytesPerIntermediateOutput;// Number of bytes per intermediate output node, always set to GNA_INT32.
    void* pLayerStruct;             // Layer detailed configuration, cast to intel_[LAYER_KIND]_layer_t.
    void* pInputs;                  // Signed integer NN or GMM input buffer.
    void* pOutputsIntermediate;     // 4B Signed integer Auxiliary output buffer.
    void* pOutputs;                 // Signed integer output buffer.
} gna_nnet_layer_t;

/** GNA Network descriptor */
typedef struct _nnet_type_t {
    uint32_t nLayers;               // The number of layers in the network.
    uint32_t nGroup;                // Input vector grouping level.
    gna_nnet_layer_t *pLayers;    // Layer configurations.
} gna_nnet_type_t;

/** Affine function details */
typedef struct _affine_func_t {
    uint32_t nBytesPerWeight;       // Precision/mode of weight element, use a value from gna_data_mode.
    uint32_t nBytesPerBias;         // Precision/mode of bias (constant) element, use a value from gna_data_mode.
    void* pWeights;                 // Signed integer weights data buffer.
    void* pBiases;                  // Biases (constants) data buffer. Signed integer biases or gna_compound_bias_t
} gna_affine_func_t;

/** Fully connected affine layer detailed descriptor */
typedef struct _affine_layer_t {
    gna_affine_func_t affine;     // Affine function details.
    gna_pwl_func_t pwl;           // Activation function details.
} gna_affine_layer_t;

/** Pooling function types */
typedef enum _pool_type_t {
    INTEL_NO_POOLING = 0,           // Pooling function disabled.
    INTEL_MAX_POOLING = 1,          // Max Pooling function.
    INTEL_SUM_POOLING = 2,          // Sum Pooling function.
    NUM_POOLING_TYPES               // Number of Pooling function types.
} gna_pool_type_t;

/** Convolutional Layer detailed descriptor */
typedef struct _convolutional_layer_t {
    uint32_t nFilters;              // Number of filters.
    uint32_t nFilterCoefficients;   // Number of filter elements, including 0-padding if necessary.
    uint32_t nFilterRows;           // Number of rows in each filter.
    uint32_t nBytesFilterCoefficient;// Precision/mode of filter coefficient element, use a value from gna_data_mode.
    uint32_t nBytesBias;            // Precision/mode of bias (constant) element, use a value from gna_data_mode.
    uint32_t nFeatureMaps;          // Number of feature maps.
    uint32_t nFeatureMapRows;       // Number of rows in each feature map.
    uint32_t nFeatureMapColumns;    // Number of columns in each feature map.
    void* pFilters;                 // Signed integer Filters data buffer, filters stored one after the other.
    void* pBiases;                  // Signed integer Biases (constants) data buffer, biases are specified per kernel/filter.
    gna_pool_type_t poolType;     // Pooling function type.
    uint32_t nPoolSize;             // Pool size, set 1 to disable pooling.
    uint32_t nPoolStride;           // Pool stride.
    gna_pwl_func_t pwl;           // Activation function details.
} gna_convolutional_layer_t;

/**
 The list of processing acceleration modes.
 Current acceleration modes availability depends on the CPU type.
 Available modes are detected by GNA.

 NOTE:
 - GNA_HARDWARE: in some GNA hardware generations, model components unsupported
   by hardware will be processed using software acceleration.
 When software inference is used, by default "fast" algorithm is used
 and results may be not bit-exact with these produced by hardware device.
 */
typedef enum  _acceleration {
    GNA_HARDWARE = static_cast<int>(0xFFFFFFFE), // GNA Hardware acceleration enforcement
    GNA_AUTO     = 0x3,             // GNA selects the best available acceleration
    GNA_SOFTWARE = 0x5,             // GNA selects the best available software acceleration
    GNA_GENERIC  = 0x7,             // Enforce the usage of generic software mode
    GNA_SSE4_2   = 0x9,             // Enforce the usage of SSE 4.2 CPU instruction set
    GNA_AVX1     = 0xB,             // Enforce the usage of AVX1 CPU instruction set
    GNA_AVX2     = 0xD              // Enforce the usage of AVX2 CPU instruction set
} gna_acceleration;

static_assert(4 == sizeof(gna_acceleration), "Invalid size of gna_acceleration");

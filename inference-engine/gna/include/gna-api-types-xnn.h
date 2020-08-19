/*
    Copyright 2018 Intel Corporation.
    This software and the related documents are Intel copyrighted materials,
    and your use of them is governed by the express license under which they
    were provided to you (Intel OBL Software License Agreement (OEM/IHV/ISV
    Distribution & Single User) (v. 11.2.2017) ). Unless the License provides
    otherwise, you may not use, modify, copy, publish, distribute, disclose or
    transmit this software or the related documents without Intel's prior
    written permission.
    This software and the related documents are provided as is, with no
    express or implied warranties, other than those that are expressly
    stated in the License.
*/

/******************************************************************************
 *
 * GNA 2.0 API
 * Gaussian Mixture Models (GMM) and Neural Network (XNN) Accelerator
 *
 * API Neural Network types definition
 *
 *****************************************************************************/

#ifndef __GNA_API_TYPES_XNN_H
#define __GNA_API_TYPES_XNN_H

#if !defined(_WIN32)
#include <assert.h>
#endif

#include <stdint.h>

#include "gna-api-status.h"

#ifdef __cplusplus
extern "C" {
#endif


    // TODO:3:API redesign: split old structures into transforms and components Like CNN2D



/******************************************************************************
 * GNA Enumerations
 *****************************************************************************/

/**
 * Mode and precision (elements) of used data.
 * Used as (OR'ed) binary flags for reporting capabilities, e.g. (GNA_INT32 | GNA_DATA_DISABLED).
 * NOTE: Not all modes are supported by all data types.
 */
typedef enum _data_mode
{
    GNA_DATA_NOT_SUPPORTED = (int)GNA_NOT_SUPPORTED,
    GNA_INT8 = 1, // TODO:3: refactor as below to GNA_INT8, as if we start supporting unsigned ints this will be ambiguous
    GNA_INT16 = 2,
    GNA_INT32 = 4,
    GNA_INT64 = 8,
    GNA_UINT8 = 1024 + 1,
    GNA_UINT16 = 1024 + 2,
    GNA_UINT32 = 1024 + 4,
    GNA_UINT64 = 1024 + 8,
    GNA_DATA_RICH_FORMAT = 8,                   // 8B Rich bias intel_compound_bias_t data is used, only with GNA_INT8 weight mode.
    GNA_DATA_CONSTANT_SCALAR = 32,              // Single 4B (GNA_INT32) signed integer scalar is used instead of tensor.
    GNA_DATA_ACTIVATION_DISABLED = 64,          // Output data only - activation function is disabled, 4B non-activated output is used.
    GNA_DATA_EXTERNAL_IO_BUFFER = 128,          // External Input/output buffer (only for ANNA)
    GNA_DATA_DISABLED = (int)GNA_DISABLED,           // No data is read
} gna_data_mode;

// TODO:3: Split into data precision and data mode

/** Pooling function types */
typedef enum _pool_type_t
{
    INTEL_NO_POOLING = 0,           // Pooling function disabled.
    INTEL_MAX_POOLING = 1,          // Max Pooling function.
    INTEL_SUM_POOLING = 2,          // Sum Pooling function.
    NUM_POOLING_TYPES               // Number of Pooling function types.

} intel_pool_type_t;

//TODO:3: use new type for naming and value consistency
///**
// * Mode of pooling operation.
// * Used as (OR'ed) binary flags for reporting capabilities, e.g. (GNA_POOLING_SUM | GNA_POOLING_MAX);
// */
//typedef enum _pooling_mode
//{
//    GNA_POOLING_NOT_SUPPORTED = (int)GNA_NOT_SUPPORTED,
//    GNA_POOLING_MAX = 1,
//    GNA_POOLING_SUM = 2,
//    GNA_POOLING_DISABLED = (int)GNA_DISABLED,
//} gna_pooling_mode;

/**
 * Mode of bias for convolution operation.
 * Used as (OR'ed) binary flags for reporting capabilities, e.g. (GNA_POOLING_SUM | GNA_POOLING_MAX);
 */
typedef enum _bias_mode
{
    GNA_BIAS_NOT_SUPPORTED = (int)GNA_NOT_SUPPORTED,
    GNA_BIAS_PER_KERNEL = 1,
    GNA_BIAS_PER_STRIDE = 2,
} gna_bias_mode; // TODO:3: incorporate into data type and remove

// TODO:3: Currently used only internally
/**
 * Order of data tensor used by inputs, outputs, biases, weights etc.
 * Used as (OR'ed) binary flags for reporting capabilities, e.g. (GNA_TENSOR_NHWD | GNA_TENSOR_NDHW);
 */
typedef enum _tensor_order
{
    GNA_TENSOR_SCALAR = 0,      // Scalar, order = 0
    GNA_TENSOR_W = 1,           // Width (1D vector)
    GNA_TENSOR_H = 2,           // Height (1D vector)
    GNA_TENSOR_NW = 4,          // Grouping, Width (2D Matrix) AKA INTERLEAVED
    GNA_TENSOR_NH = 8,          // Grouping, Height (2D Matrix) AKA INTERLEAVED
    GNA_TENSOR_WN = 16,         // Width, Grouping (2D Matrix) AKA DEINTERLEAVED/FLAT
    GNA_TENSOR_WH = 32,         // Width, Height (2D Matrix) (Weights)
    GNA_TENSOR_HN = 64,         // Height, Grouping (2D Matrix) AKA DEINTERLEAVED/FLAT
    GNA_TENSOR_HW = 128,        // Height, Width (2D Matrix) common for all 2D tensors
    GNA_TENSOR_HD = 256,        // Height, Depth/Channel, (2D Matrix)
    GNA_TENSOR_HDW = 512,       // Height, Depth/Channel, Width (3D Tensor)
    GNA_TENSOR_NWH = 1024,      // Grouping, Width, Height (3D Tensor)
    GNA_TENSOR_WHD = 2048,      //  Width, Height, Depth/Channel (3D Tensor)
    GNA_TENSOR_NHWD = 4096,     // N -Grouping[=1]/Number of filters, Height, Width, Depth/Channel (GNA 2D CNN default) (4D Tensor)
    GNA_TENSOR_NDHW = 8192,     // N -Grouping[=1]/Number of filters, Depth/Channel, Height, Width, (TensorFlow) (4D Tensor)
    GNA_TENSOR_ORDER_ANY = -1,  // ordering as in gna_tensor_dim beginning with GNA_DIM_N
    GNA_TENSOR_NHW = 4097,     // Temporary value for Bias Shape
    GNA_TENSOR_N = 3,     // Temporary value for Bias Shape
    GNA_TENSOR_HWD = 513,

    // TODO:3:change to char tensor_format[16?] and implement ~~ uint32 tensor_format.Value...

} gna_tensor_order;

// TODO:3: Currently used only internally
/**
 * Helper Tensor dimension selector for dimension map.
 */
typedef enum _tensor_dim
{
    GNA_DIM_S,          // Scalar
    GNA_DIM_N,          // Grouping (Batch size)
    GNA_DIM_W,          // Width
    GNA_DIM_H,          // Height
    GNA_DIM_D,          // Depth (for 2D operations same as Channel)
    //GNA_DIM_C,          // Channel (for 2D operations same as Depth)

    GNA_DIM_X,
    GNA_DIM_Y,
    GNA_DIM_Z,
} gna_tensor_dim;

// TODO:3: Subject to change due to naming inconsistency
/**
 * Layer operation type.
 * Defines type of layer "core" operation.
 * All nodes/cells within a layer are of the same type,
 * e.g. affine transform cell, convolutional cell, recurrent cell.
 * Affine, convolutional and recurrent layers are in fact "fused operation" layers
 * and "core" operation is fused with activation and/or pooling functions.
 * NOTE: Operation types are exclusive.
 */
typedef enum _layer_operation
{
    INTEL_AFFINE,                   // Fully connected affine transform (deep feed forward) with activation function. Cast pLayerStruct to intel_affine_layer_t.
    INTEL_AFFINE_DIAGONAL,          // Fully connected affine transform (matrix x vector) (deep feed forward) with activation function.Cast pLayerStruct to intel_affine_layer_t.
    INTEL_AFFINE_MULTIBIAS,         // Fully connected affine transform (with grouped bias vectors) (deep feed forward) with activation function. Cast pLayerStruct to intel_affine_multibias_layer_t.
    INTEL_CONVOLUTIONAL,            // Convolutional transform with activation function and pooling. Cast pLayerStruct to intel_convolutional_layer_t.
    INTEL_CONVOLUTIONAL_2D,         // Convolutional transform with activation function and pooling. Cast pLayerStruct to nn_layer_cnn2d.
    INTEL_CONVOLUTIONAL_1D,         // FOR INTERNAL USE ONLY
    INTEL_COPY,                     // Auxiliary data copy operation. Cast pLayerStruct to intel_copy_layer_t.
    INTEL_DEINTERLEAVE,             // Auxiliary 2D tensor transpose operation (interleave to flat). No casting, always set pLayerStruct to null.
    INTEL_GMM,                      // Gaussian Mixture Model operation. Cast pLayerStruct to intel_gmm_layer_t.
    INTEL_INTERLEAVE,               // Auxiliary 2D tensor transpose operation (flat to interleave). No casting, always set pLayerStruct to null.
    INTEL_RECURRENT,                // Fully connected affine transform with recurrence and activation function. Cast pLayerStruct to intel_recurrent_layer_t.
    // TODO:3: verify list of CNN layers
    GNA_LAYER_CNN_2D_ADDITION,// TODO:3: add layer support + capabilities
    GNA_LAYER_CNN_2D_CONVERSION,// TODO:3: add layer support + capabilities
    GNA_LAYER_CNN_2D_POOLING,   // TODO:3: add layer support + capabilities
    LAYER_OPERATION_TYPE_COUT,      // Number of Layer operation types.

    //// GNA-next names
    //GNA_LAYER_GMM,
    //GNA_LAYER_AFFINE,
    //GNA_LAYER_AFFINE_MULTIBIAS,
    //GNA_LAYER_AFFINE_DIAGONAL,
    //GNA_LAYER_CNN_1D,
    //GNA_LAYER_CNN_2D,
    //GNA_LAYER_CNN_2D_ADDITION,
    //GNA_LAYER_CNN_2D_POOLING,
    //GNA_LAYER_CNN_2D_CONVERSION,
    //GNA_LAYER_RECURRENT,
    //GNA_LAYER_DEINTERLEAVE,
    //GNA_LAYER_INTERLEAVE,
    //GNA_LAYER_COPY,
    //GNA_LAYER_TRESHOLD,             // ANNA only
    //GNA_LAYER_OPERATION_COUNT

} gna_layer_operation;

/**
 TODO:3:remove
 */
typedef enum _layer_mode
{
    INTEL_INPUT,            // Layer serves as model input layer (usually first layer)
    INTEL_OUTPUT,           // Layer serves as model output layer (usually last layer)
    INTEL_INPUT_OUTPUT,     // Layer serves as model input nad output layer (usually in single layer topology)
    INTEL_HIDDEN,           // Layer serves as model hidden layer (layers between input and output layers)

    LAYER_MODE_COUNT        // Number of Layer modes.

} gna_layer_mode;


/******************************************************************************
 * GNA Neural Network Model structures
 *****************************************************************************/

// TODO:3: remove
/** Bias (constant) data type */
typedef int32_t intel_bias_t;

/**
 * Compound bias
 * Used for nBytesPerWeight=GNA_INT8 and nBytesPerBias=GNA_INT16 only.
 * As read directly by the accelerator.
 */
typedef struct _compound_bias_t
{
    intel_bias_t bias;              // 4B Signed integer bias (constant) value.
    uint8_t multiplier;             // Scaling factor that weight elements are multiplied by.
    uint8_t reserved[3];            // Not used.

} intel_compound_bias_t;

static_assert(8 == sizeof(intel_compound_bias_t), "Invalid size of intel_compound_bias_t");

// TODO: 3.0 verify GNA_DATA_DISABLED and GNA_DATA_CONSTANT_SCALAR usage
/** Affine function details */
typedef struct _affine_func_t
{
    uint32_t nBytesPerWeight;       // Precision/mode of weight element, use a value from gna_data_mode. Valid values {GNA_INT8, GNA_INT16, GNA_DATA_CONSTANT_SCALAR}
    uint32_t nBytesPerBias;         // Precision/mode of bias (constant) element, use a value from gna_data_mode. Valid values {GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_RICH_FORMAT, GNA_DATA_DISABLED}
    void* pWeights;                 // Signed integer weights data buffer.
    void* pBiases;                  // Biases (constants) data buffer. Signed integer biases or intel_compound_bias_t (for nBytesPerWeight=GNA_INT8 and nBytesPerBias=GNA_INT16 only).

} intel_affine_func_t;

/** Weight element scaling factor for intel_affine_multibias_func_t - as read directly by the accelerator */
typedef struct _weight_scaling_factor_t
{
    uint8_t reserved0[4];           // Not used.
    uint8_t multiplier;             // Scaling factor that weight elements are multiplied by.
    uint8_t reserved1[3];           // Not used.

} intel_weight_scaling_factor_t;
static_assert(8 == sizeof(intel_weight_scaling_factor_t), "Invalid size of intel_weight_scaling_factor_t");

/** Affine function with bias grouping details */
typedef struct _affine_multibias_func_t
{
    uint32_t nBytesPerWeight;       // Precision/mode of weight element, use a value from gna_data_mode. Valid values {GNA_INT8, GNA_INT16, GNA_DATA_CONSTANT_SCALAR}
    void* pWeights;                 // Signed integer weights data buffer.
    intel_weight_scaling_factor_t* weightScaleFactors; // Weight element scaling factors data buffer for nBytesPerWeight=GNA_INT8 and nBytesPerBias=GNA_DATA_RICH_FORMAT only or NULL for other cases.
    uint32_t nBytesPerBias;         // Precision/mode of bias (constant) element, use a value from gna_data_mode. Valid values {GNA_INT8, GNA_INT16, GNA_INT32}
    uint32_t biasVectorCount;       // Number of the bias vectors in 2D bias array.
    uint32_t biasVectorIndex;       // Index of the bias vector (column) for the current layer.
    void* pBiases;                  // 2D array containing set of signed integer bias vectors.

} intel_affine_multibias_func_t;

/** PWL Segment - as read directly by the accelerator */
typedef struct _pwl_segment_t
{
    int32_t xBase;                  // X Component of segment starting point, with scaling encoded if needed.
    int16_t yBase;                  // Y Component of segment starting point.
    int16_t slope;                  // Slope of linear function.

} intel_pwl_segment_t;
static_assert(8 == sizeof(intel_pwl_segment_t), "Invalid size of intel_pwl_segment_t");

/** Piecewise-linear activation function (PWL) details */
typedef struct _pwl_func_t
{
    uint32_t nSegments;             // Number of segments, set to 0 to disable activation function.
    intel_pwl_segment_t* pSegments; // Activation function segments data or NULL if disabled.

} intel_pwl_func_t;

// TODO:3:GNA-4/next proposal
///** Piecewise-linear activation function (PWL) details */
//typedef struct _activation_function
//{
//    gna_data_mode mode;             // Activation function mode and precision. Valid values: {GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}
//    bool isReLuFunction;        // Set to true to indicate ReLu function is used in PWL.
//    uint32_t zeroSegmentIndex;      // Index of first PWL segment with positive xBase value. Set when isReLuFunction=true.
//    uint32_t segmentCount;          // Number of segments.
//    uint32_t* segmentXpoints;       // Array of segment starting points.
//    uint16_t* segmentValues;        // Array of segment values.
//    uint16_t* segmentSlopes;        // Array of segment function slopes.
//    uint32_t* segmentShifts;        // Array of segment value shift. // TODO: provide function calculation recipe.
//} gna_activation_function;

/** Fully connected affine layer detailed descriptor */
typedef struct _affine_layer_t
{
    intel_affine_func_t affine;     // Affine function details.
    intel_pwl_func_t pwl;           // Activation function details.

} intel_affine_layer_t;

/** Fully connected affine layer with bias grouping, detailed descriptor */
typedef struct _affine_multibias_layer_t
{
    intel_affine_multibias_func_t affine;// Affine function with bias grouping.
    intel_pwl_func_t pwl;           // Activation function details.

} intel_affine_multibias_layer_t;

/** Recurrent Layer detailed descriptor */
typedef struct _recurrent_layer_t
{
    intel_affine_func_t affine;     // Affine function details.
    intel_pwl_func_t pwl;           // Activation function details.
    uint32_t feedbackFrameDelay;    // Feedback input Delay in term of number of frames (feature vectors) in request.
} intel_recurrent_layer_t;

/** Convolutional Layer detailed descriptor */
typedef struct _convolutional_layer_t
{
    uint32_t nFilters;              // Number of filters.
    uint32_t nFilterCoefficients;   // Number of filter elements, including 0-padding if necessary.
    uint32_t nFilterRows;           // Number of rows in each filter.
    uint32_t nBytesFilterCoefficient;// Precision/mode of filter coefficient element, use a value from gna_data_mode. Valid values {GNA_INT8, GNA_INT16, GNA_DATA_CONSTANT_SCALAR}
    uint32_t nBytesBias;            // Precision/mode of bias (constant) element, use a value from gna_data_mode. Valid values {GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_DISABLED}
    uint32_t nFeatureMaps;          // Number of feature maps.
    uint32_t nFeatureMapRows;       // Number of rows in each feature map.
    uint32_t nFeatureMapColumns;    // Number of columns in each feature map.
    void* pFilters;                 // Signed integer Filters data buffer, filters stored one after the other.
    void* pBiases;                  // Signed integer Biases (constants) data buffer, biases are specified per kernel/filter.
    intel_pool_type_t poolType;     // Pooling function type.
    uint32_t nPoolSize;             // Pool size, set 1 to disable pooling.
    uint32_t nPoolStride;           // Pool stride.
    intel_pwl_func_t pwl;           // Activation function details.

} intel_convolutional_layer_t;

// TODO:3: consider renaming to shape
/** 3D Tensor dimensions (shape) */
typedef struct _3d_dimensions
{
    // Number of elements in dimension W // GNA_DIM_W
    uint32_t width;

    // Number of in elements in dimension H // GNA_DIM_H
    uint32_t height;

    // Number of in elements in dimension D // GNA_DIM_D
    uint32_t depth;

} gna_3d_dimensions;

///** 4D Tensor dimensions (shape) */
//typedef struct _4d_dimensions
//{
//    uint32_t width;                 // Number of elements in dimension W // GNA_DIM_W
//    uint32_t height;                // Number of in elements in dimension H // GNA_DIM_H
//    uint32_t depth;                 // Number of in elements in dimension D // GNA_DIM_D
//    uint32_t channels;              // Number of in channels per element  // GNA_DIM_C
//
//} gna_4d_dimensions;

/** Convolution filter details */
typedef struct _convolution_filter
{
    // TODO:3: verify if tensor order is fixed?
    // Precision/mode of filter coefficient element.
    // Valid values {GNA_INT8, GNA_INT16, GNA_DATA_CONSTANT_SCALAR}
    gna_data_mode dataMode;

    // # of convolution filters // GNA_DIM_N
    uint32_t count;

    // Shape of each filter. (3D: WxHxD)
    gna_3d_dimensions dimensions;

    // Integer filters' coefficients data buffer.
    // Filters stored one after the other in GNA_TENSOR_NHWD order.
    // Each Kernel must start at address which is 16B aligned.
    void* filtersData;

} gna_convolution_filter;

/** Convolution biases (constants) details */
typedef struct _convolution_bias
{
    // TODO:3: verify if tensor order is fixed?
    gna_bias_mode mode;             // Mode of bias for convolution operation. {GNA_BIAS_PER_KERNEL (HWD=1), GNA_BIAS_PER_STRIDE (HWD each filter dimensions)}
    gna_data_mode dataMode;         // Precision/mode of bias (constant) element. {GNA_INT8, GNA_INT16, GNA_DATA_CONSTANT_SCALAR}
    void* biasesData;               // Biases data buffer. Signed integer biases.

} gna_convolution_bias;

/** Convolution function details */
typedef struct _convolution_func
{
    // Convolution filter (kernel) details
    gna_convolution_filter filters;

    // Convolution filter stride shape (2D: WxH)
    gna_3d_dimensions stride;

    // Automatic zero-padding dimensions (2D: WxH).
    // Used to maintain same input-output volume shape
    // or when input dimensions have no common natural divider with filter and stride.
    gna_3d_dimensions zeroPadding;

    // Convolution biases tensor
    gna_convolution_bias biases;

    // Convolution output is in TODO:3:CNN2D order
} gna_convolution_func;

/** Pooling function details */
typedef struct _pooling_func
{
    intel_pool_type_t type;         // Pooling function type.
    gna_3d_dimensions stride;       // Pooling window stride dimensions (2D: WxH)
    gna_3d_dimensions window;       // Pooling window shape (2D: WxH)
} gna_pooling_func;

// TODO:3: Design 2D CNN
/** 2D Convolutional Layer fused with Activation and Pooling operations detailed descriptor */
typedef struct _convolutional_fused_layer_2d
{
    // Input tensor shape (3D: WxHxD)
    gna_3d_dimensions inputDimensions;
    gna_convolution_func convolution;
    gna_pooling_func pooling;
    intel_pwl_func_t activation; // pwl
} gna_convolutional_fused_layer_2d;

/** 2D Convolutional Layer detailed descriptor */
typedef struct _convolutional2d_layer_t
{
    // Input tensor shape (3D: WxHxD)
    gna_3d_dimensions inputDimensions;

    gna_pooling_func pooling;
} gna_pooling_layer_2d;

/** Copying Layer detailed configuration */
typedef struct _copy_layer_t
{
    uint32_t nCopyRows;             // Number of rows affected (1-8).
    uint32_t nCopyCols;             // Number of columns in a row to copy.

} intel_copy_layer_t;

/** Layer common configuration descriptor */
typedef struct _nnet_layer_t
{
    gna_layer_operation operation;  // Layer operation type.
    gna_layer_mode mode;            // Layer connection mode.
    uint32_t nInputColumns;         // Number of input columns.
    uint32_t nInputRows;            // Number of input rows.
    uint32_t nOutputColumns;        // Number of output columns.
    uint32_t nOutputRows;           // Number of output rows.
    uint32_t nBytesPerInput;        // Precision/mode of input node, use a value from gna_data_mode. Valid values {GNA_INT8, GNA_INT16, GNA_DATA_DISABLED}
    uint32_t nBytesPerOutput;       // Precision/ activation mode of output node, use a value from gna_data_mode. Valid values {GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}
    uint32_t nBytesPerIntermediateOutput;// Number of bytes per intermediate output node, always set to GNA_INT32.
    void* pLayerStruct;             // Layer detailed configuration, cast to intel_[LAYER_KIND]_layer_t.
    void* pInputs;                  // Signed integer NN or GMM input buffer.
    void* pOutputsIntermediate;     // 4B Signed integer Auxiliary output buffer.
    void* pOutputs;                 // Signed integer output buffer.

} intel_nnet_layer_t;

/** GNA Network descriptor */
typedef struct _nnet_type_t
{
    uint32_t nLayers;               // The number of layers in the network.
    uint32_t nGroup;                // Input vector grouping level.
    intel_nnet_layer_t *pLayers;    // Layer configurations.

} intel_nnet_type_t;

/******************************************************************************
 * GNA Constant values
 *****************************************************************************/

 //TODO:3:CAPS: move to caps

/** Size of memory alignment for data tensors */
const uint32_t GNA_MEM_ALIGN = 64;

/** Number of input groups constraint - max */
const uint32_t XNN_N_GROUP_MAX = 8;

/** Number of input groups constraint for Copy layer 3.0- max */
const uint32_t COPY_N_GROUP_MAX = 255;

/** Total number of input elements constraint - must be multiple of */
const uint32_t XNN_N_IN_ELEMS_MPLY = 8;

/** Total number of output elements constraint - must be multiple of */
const uint32_t RNN_N_OUT_ELEMS_MPLY = 32;

/** Total number of input elements constraint - max elements */
const uint32_t XNN_N_IN_ELEMS_MAX = UINT16_MAX;

/** Number of pwl segments constraint - max  */
const uint32_t XNN_N_PWL_SEGS_MAX = 128;

/** Number of pwl segments constraint - min  */
const uint32_t XNN_N_PWL_SEGS_MIN = 2;

/** Weight elements size constraint - max size B */
const uint32_t XNN_W_ELEM_SZ_MAX = 2;

/** CNN minimum number of filter coefficients */
const uint32_t CNN_N_FLT_COEFF_MIN = 8;

/** CNN maximum number of filter coefficients */
const uint32_t CNN_N_FLT_COEFF_MAX = 768;

/** CNN 2D minimum number of kernel elements in one dimension */
const uint32_t CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MIN = 1;

/** CNN 2D maximum number of kernel elements in one dimension */
const uint32_t CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX = 256;

/** CNN 1D maximum number of kernel elements in one dimension For int8_t */
const uint32_t CNN_1D_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX = 2048;

/** CNN number of filter coefficients constraint - must be multiple of */
const uint32_t CNN_N_FLT_COEFF_MPLY = 4;

/** CNN maximum number of filters */
const uint32_t CNN_N_FLT_MAX = ((UINT16_MAX + 1) - 4);

/** CNN 2D maximum number of kernels */
const uint32_t CNN_N_KERNELS_MAX = UINT16_MAX;

/** CNN D maximum number of kernels */
const uint32_t CNN_1D_N_KERNELS_MAX = 8192;

/** CNN minimum size of pooling window */
const uint32_t CNN_POOL_SIZE_MIN = 1;

/** CNN maximum size of pooling window */
const uint32_t CNN_POOL_SIZE_MAX = 6;

const uint32_t GNA_COVARIANCE_SIZE_MIN = 1;
#ifdef __cplusplus
}
#endif

#endif  // ifndef __GNA_API_TYPES_XNN_H

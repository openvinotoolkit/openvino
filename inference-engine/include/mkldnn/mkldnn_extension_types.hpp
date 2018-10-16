// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the main MKL-DNN Extension API to work with weights, and primitives in memory
 * @file mkldnn_extension_types.hpp
 */
#pragma once

#include "ie_common.h"
#include "ie_precision.hpp"

namespace InferenceEngine {
namespace MKLDNNPlugin {

/**
 * @deprecated use new extensibility API
 * @brief Defines formats from MKL-DNN which are supported in the MKL-DNN plugin of IE.
 */
enum MemoryFormat {
    /** Undefined memory format, used for empty memory descriptors. */
    format_undef = 0,
    /** Unspecified format. The primitive selects a format
     * automatically. */
    any,
    /** A tensor in a generic format described by the stride and blocking
     * values in each dimension. See #mkldnn_blocking_desc_t for more
     * information. */
    blocked,
    /** 1D data tensor. */
    x,
    /** 2D data tensor. */
    nc,
    /** 4D data tensor in the @c nchw format typically used in Caffe. */
    nchw,
    /** 4D data tensor in the @c nhwc format typically used in TensorFlow. */
    nhwc,
    /** 4D data tensor in the @c chwn format typically used in Neon. */
    chwn,
    /** 4D data tensor in the @c nchw format with channels data laid out in
     * memory in 8-element blocks. */
    nChw8c,
    /** 4D data tensor in the @c nchw format with channels data laid out in
     * memory in 16-element blocks. */
    nChw16c,
    /** 2D weights tensor in the format (input channels, output channels). */
    oi,
    /** 2D weights tensor in the format (input channels, output channels). */
    io,
    /** 4D weights tensor in the format (input channels, output channels,
     * width, height). */
    oihw,
    /** 4D weights tensor in the format (input channels, height, width,
     * output channels). */
    ihwo,
    /** 4D weights tensor in the format (height, width, input channels,
     * output channels). */
    hwio,
    /** 4D weights tensor in the @c oihw format with both input and output
     * channels data laid out in memory in 8-element blocks. */
    OIhw8i8o,
    /** 4D weights tensor in the @c oihw format with both input and output
     * channels data laid out in memory in 16-element blocks. */
    OIhw16i16o,
    /** 4D weights tensor in the @c oihw format with output channels data
     * laid out in memory in 16-element blocks and input channels data
     * laid out in memory in 8-element blocks blocked by pairs. */
    OIhw8i16o2i,
    /** 4D weights tensor in the @c oihw format with input channels data
     * laid out in memory in 16-element blocks and output channels data
     * laid out in memory in 8-element blocks blocked by pairs. */
    OIhw8o16i2o,
    /** 4D weights tensor in the @c oihw format with both input and output
     * channels data laid out in memory in 8-element blocks. */
    OIhw8o8i,
    /** 4D weights tensor in the @c oihw format with both input and output
     * channels data laid out in memory in 16-element blocks. */
    OIhw16o16i,
    /** 4D weights tensor in the format (output channels, input channels,
     * height, width) with output channels data laid out in memory in 8-element
     * blocks. */
    Oihw8o,
    /** 4D weights tensor in the format (output channels, input channels,
     * height, width) with output channels data laid out in memory in
     * 16-element blocks. */
    Oihw16o,
    /** 4D weights tensor in the format (output channels, width, height, input
     * channels) with output channels data laid out in memory in 8-element
     * blocks. */
    Ohwi8o,
    /** 4D weights tensor in the format (output channels, width, height, input
     * channels) with output channels data laid out in memory in 16-element
     * blocks. */
    Ohwi16o,
    /** 4D weights tensor in the @c oihw format with both input and output
     * channels data laid out in memory in 16-element and 4-element blocks. */
    OhIw16o4i,
    /** 5D weights tensor in the @c oihw format with extra outer dimension for
     * groups. */
    goihw,
    /** 5D weights tensor in the blocked version of @c goihw format with both
     * input and output channels data laid out in memory in 8-element blocks.
     */
    gOIhw8i8o,
    /** 5D weights tensor in the blocked version of @c goihw format with both
     * input and output channels data laid out in memory in 16-element blocks.
     */
    gOIhw16i16o,
    /** 5D weights tensor in the @c oihw format with output channels data
     * laid out in memory in 16-element blocks and input channels data
     * laid out in memory in 8-element blocks blocked by pairs. */
    gOIhw8i16o2i,
    /** 5D weights tensor in the @c oihw format with input channels data
     * laid out in memory in 16-element blocks and output channels data
     * laid out in memory in 8-element blocks blocked by pairs. */
    gOIhw8o16i2o,
    /** 5D weights tensor in the blocked version of @c goihw format with both
     * input and output channels data laid out in memory in 8-element blocks.
     */
    gOIhw8o8i,
    /** 5D weights tensor in the blocked version of @c goihw format with both
     * input and output channels data laid out in memory in 16-element blocks.
     */
    gOIhw16o16i,
    /** 5D weights tensor in the blocked version of @c goihw format with output
     * channels data laid out in memory in 8-element blocks. */
    gOihw8o,
    /** 5D weights tensor in the blocked version of @c goihw format with output
     * channels data laid out in memory in 16-element blocks. */
    gOihw16o,
    /** 5D weights tensor in the blocked version of @c goihw format with output
     * channels data laid out in memory in 8-element blocks. */
    gOhwi8o,
    /** 5D weights tensor in the blocked version of @c goihw format with output
     * channels data laid out in memory in 16-element blocks. */
    gOhwi16o,
    /** 5D weights tensor in the @c goihw format with both input and output
     * channels data laid out in memory in 16-element and 4-element blocks. */
    gOhIw16o4i,
    /** 4D weights tensor in the oihw format with input channels data laid out
     * in memory in 8-element blocks. */
    oIhw8i = nChw8c,
    /** 4D weights tensor in the oihw format with input channels data laid out
     * in memory in 16-element blocks. */
    oIhw16i = nChw16c,
};

/**
 * @deprecated use new extensibility API
 * @brief Stores necessary information about the primitive memory object.
 * Such as precision, dimensions, memory format etc.
 */
struct MKLDNNPrimitiveMemory {
    /**
     * @brief precision type
     */
    Precision precision;
    /**
     * @brief dimensions of the given primitive
     */
    SizeVector dims;
    /**
     * @brief memory type of the given primitive
     */
    MemoryFormat format;
    /**
     * @brief primitive data stored
     */
    void *data;

    /**
     * @brief A constructor.
     */
    MKLDNNPrimitiveMemory() : format(format_undef), data(nullptr) {}
};

/**
 * @deprecated use new extensibility API
 * @brief Stores necessary information about the primitive weights.
 */
struct MKLDNNWeightsMemory {
    /**
     * @brief size of weights
     */
    size_t size;
    /**
     * @brief pointer to weights data
     */
    void *data;

    /**
     * @brief A constructor.
     */
    MKLDNNWeightsMemory() : size(0), data(nullptr) {}
};

}  // namespace MKLDNNPlugin
}  // namespace InferenceEngine

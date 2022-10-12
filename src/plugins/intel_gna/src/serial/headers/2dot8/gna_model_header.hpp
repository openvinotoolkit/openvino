// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include "backend/dnn_types.h"
#include "serial/headers/2dot7/gna_model_header.hpp"
#include "gna_data_types.hpp"

#pragma pack(push, 1)

namespace GNAPluginNS {
namespace Header2dot8 {

/**
 Maximal number of supported shape dimensions.
 */
#define GNA_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS 8

/**
 * @brief Header version 2.8
 */
struct ModelHeader {
    /**
     *@brief MagicNumber – GNAM in ascii table, equals to hex 0x474e414d
     */
    char gnam[4] = {};
    /**
     * @brief if header size is not equal to sizeof ModelHeader - some reserved data append in the end of header
     * usually it is an indicator of working with version of model different that is current export function produce
     */
    uint32_t headerSize = 0u;
    struct Version {
        /**
         * @details Version of format Major – unsigned int, ex: 0x0001
         * every change in the header or in the layers definition should be reflected in version change
         * for backward compatibility new parsers can read old versions of model with certain restrictions
         */
        uint16_t major = 2u;
        /**
         * @details Version of Format Minor – unsigned int,  corresponding to build revision for example
         * changes in minor version are not affected layout of model
         */
        uint32_t minor = 8u;
    } version;
    /**
     * @brief Memory required to be allocated using GNAAlloc()
     */
    uint64_t gnaMemSize = 0ull;
    /**
     * @brief Number of GNA Layers
     */
    uint64_t layersCount = 0ull;
    /**
     * @brief Grouping level
     * This is depricted field and used for old models only (<=2.6)
     */
    uint32_t nGroup = 0u;

    /**
     * Convolution related setting - they are affecting input transformation
     */
    uint32_t nRotateRows = 0u;
    uint32_t nRotateColumns = 0u;
    bool doRotateInput = false;

    uint32_t nInputs = 0u;
    uint32_t nOutputs = 0u;

    /**
     * Convolution related setting - they are affecting output transformation
     */
    uint32_t nRotateOutputRows = 0u;
    uint32_t nRotateOutputColumns = 0u;
    bool doRotateOutput = false;

    uint32_t nTransposeInputs = 0u;
    uint32_t nTransposeOutputs = 0u;

    /**
     * Reserved Data might be here
     */

    ModelHeader() = default;

    ModelHeader(GNAPluginNS::Header2dot1::ModelHeader const &old) {
        gnaMemSize = old.gnaMemSize;
        layersCount = old.layersCount;
        nGroup = old.nGroup;
        nRotateRows = old.nRotateRows;
        nRotateColumns = old.nRotateColumns;
        nInputs = old.nInputs;
        nOutputs = old.nOutputs;
        version.minor = old.version.minor;
    }

    ModelHeader(GNAPluginNS::Header2dot4::ModelHeader const &old) {
        gnaMemSize = old.gnaMemSize;
        layersCount = old.layersCount;
        nGroup = old.nGroup;
        nRotateRows = old.nRotateRows;
        nRotateColumns = old.nRotateColumns;
        nInputs = old.nInputs;
        nOutputs = old.nOutputs;
        nRotateOutputRows = old.nRotateOutputRows;
        nRotateOutputColumns = old.nRotateOutputColumns;
        doRotateOutput = old.doRotateOutput;
        version.minor = old.version.minor;
    }
};
#pragma pack(pop)

/*
 * In runtime endpoint mostly same as in serial version, except of descriptor field
 */
struct RuntimeEndPoint {
    /**
     * if scale factor is different then pased into infer , network might need to be requantized
     */
    float scaleFactor = 0;
    /**
     * Pointer descriptor
     */
    void* descriptor_ptr = nullptr;
    /**
     * Endpoint resolution in bytes.
     */
    uint32_t element_size = 0;
    /**
     * Number of elements
     */
    uint32_t elements_count = 0;
    /**
     * Offset in bytes of pointer descriptor
    */
    uint64_t descriptor_offset = 0ull;
    /**
     Shape specifying dimension values.
    */
    struct Shape {
        /**
         Number of dimensions or rank or order.
        */
        uint32_t NumberOfDimensions = 0;
        /**
         array specifying value of each dimension.
        Set all zeros for scalars.
        */
        uint32_t Dimensions[GNA_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS] = {0};
    } shape;
    /**
     * Blob layout
     */
    uint8_t layout = InferenceEngine::Layout::NC;
    /**
     * Blob precision
     */
    uint8_t precision = InferenceEngine::Precision::FP32;
    /**
     * Number of tensor names
     */
    uint8_t tensor_names_count = 0;

    intel_dnn_orientation_t orientation = kDnnUnknownOrientation;

    RuntimeEndPoint() = default;

    // support of previous versions
    RuntimeEndPoint(const GNAPluginNS::Header2dot6::RuntimeEndPoint &old, uint32_t ngroup) {
        GNAPluginNS::Header2dot7::RuntimeEndPoint ep_v7 = GNAPluginNS::Header2dot7::RuntimeEndPoint(old, ngroup);
        *this = GNAPluginNS::Header2dot8::RuntimeEndPoint(ep_v7);
    }

    RuntimeEndPoint(GNAPluginNS::Header2dot7::RuntimeEndPoint &old) {
        scaleFactor = old.scaleFactor;
        descriptor_ptr = old.descriptor_ptr;
        element_size = old.element_size;
        elements_count = old.elements_count;
        orientation = old.orientation;
        layout = old.layout;
        precision = old.precision;
        descriptor_offset = old.descriptor_offset;
        shape.NumberOfDimensions = old.shape.NumberOfDimensions;
        for (uint32_t i = 0; i < shape.NumberOfDimensions; i++) {
            shape.Dimensions[i] = old.shape.Dimensions[i];
        }
        tensor_names_count = 0;
    }

    RuntimeEndPoint(double scaleFactor,
                    void* descriptor_ptr,
                    uint32_t element_size,
                    uint32_t elements_count,
                    Shape shape,
                    uint8_t layout,
                    uint8_t precision,
                    uint8_t tensor_names_count,
                    intel_dnn_orientation_t orientation) : scaleFactor(static_cast<float>(scaleFactor)),
                                                           descriptor_ptr(descriptor_ptr),
                                                           element_size(element_size),
                                                           elements_count(elements_count),
                                                           shape(shape),
                                                           layout(layout),
                                                           precision(precision),
                                                           tensor_names_count(tensor_names_count),
                                                           orientation(orientation) { }
};
} // namespace Header2dot8
} // namespace GNAPluginNS

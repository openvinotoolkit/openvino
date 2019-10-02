// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <vector>
#include <utility>
#include "gna-api.h"

#pragma pack(push, 1)

/**
 * version history
 * 1.0 - basic support
 * 1.1 - added memory information
 */

#define HEADER_MAJOR 1
#define HEADER_MINOR 1

/**
 * @brief Header version 1.0
 */
struct ModelHeader {
    /**
     *@brief MagicNumber – GNAM in ascii table, equals to hex 0x474e414d
     */
    char gnam[4];
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
        uint16_t major = 0u;
        /**
         * @details Version of Format Minor – unsigned int,  corresponding to build revision for example
         * changes in minor version are not affected layout of model
         */
        uint32_t minor = 0u;
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
     */
    uint32_t nGroup = 0u;

    /**
     * Convolution related setting - they are affecting input transformation
     */
    uint32_t nRotateRows = 0u;
    uint32_t nRotateColumns = 0u;


    struct EndPoint {
        /**
         * if scale factor is different then pased into infer , network might need to be requantized
         */
        float scaleFactor = 0.f;
        /**
         * Offset in bytes of pointer descriptor
         */
        uint64_t descriptor_offset = 0ull;
        /**
         * Endpoint resolution in bytes.
         */
        uint32_t element_size = 0u;
        /**
         * Number of elements
         */
        uint32_t elements_count = 0u;
    };
    EndPoint input;
    EndPoint output;

    /**
     * Reserved Data might be here
     */
};
#pragma pack(pop)

/**
 * @brief implements serialisation tasks for GNAGraph
 */
class GNAModelSerial {
 public:
    /*
     * In runtime endpoint mostly same as in serial version, except pf descriptor field
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

        RuntimeEndPoint() = default;
        RuntimeEndPoint(double scaleFactor,
                    void* descriptor_ptr,
                    uint32_t element_size,
                    uint32_t elements_count) : scaleFactor(scaleFactor),
                                    descriptor_ptr(descriptor_ptr),
                                    element_size(element_size),
                                    elements_count(elements_count) {
        }
    };
    using MemoryType = std::vector<std::pair<void*, uint32_t>>;

private:
    intel_nnet_type_t *ptr_nnet;
    RuntimeEndPoint input, output;
    uint32_t nRotateRows = 0;
    uint32_t nRotateColumns = 0;

    MemoryType states, *pstates = nullptr;

 public:
    /**
     *
     * @brief Used for import/export
     * @param ptr_nnet
     * @param inputScale  - in/out parameter representing input scale factor
     * @param outputScale - in/out parameter representing output scale factor
     */
    GNAModelSerial(intel_nnet_type_t *ptr_nnet, MemoryType &states_holder)
        : ptr_nnet(ptr_nnet) , pstates(&states_holder) {
    }

    /**
     * @brief used for export only since runtime params are not passed by pointer
     * @param ptr_nnet
     * @param runtime
     */
    GNAModelSerial(
        intel_nnet_type_t *ptr_nnet,
        RuntimeEndPoint input,
        RuntimeEndPoint output) : ptr_nnet(ptr_nnet), input(input), output(output) {
    }

    GNAModelSerial & SetInputRotation(uint32_t nRotateRows, uint32_t nRotateColumns) {
      this->nRotateColumns = nRotateColumns;
      this->nRotateRows = nRotateRows;
      return *this;
    }

    /**
     * mark certain part of gna_blob as state (in future naming is possible)
     * @param descriptor_ptr
     * @param size
     * @return
     */
    GNAModelSerial & AddState(void* descriptor_ptr, size_t size) {
        states.emplace_back(descriptor_ptr, size);
        return *this;
    }

    /**
     * @brief calculate memory required for import gna graph
     * @param is - opened input stream
     * @return
     */
    static ModelHeader ReadHeader(std::istream &is);

    /**
     * @brief Import model from FS into preallocated buffer,
     * buffers for pLayers, and pStructs are allocated here and required manual deallocation using mm_free
     * @param ptr_nnet
     * @param basePointer
     * @param is - stream without header structure - TBD heder might be needed
     */
    void Import(void *basePointer, size_t gnaGraphSize, std::istream &is);

    /**
     * save gna graph to an outpus stream
     * @param ptr_nnet
     * @param basePtr
     * @param gnaGraphSize
     * @param os
     */
    void Export(void *basePtr,
                size_t gnaGraphSize,
                std::ostream &os) const;
};
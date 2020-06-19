// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <vector>
#include <utility>

#include <gna-api.h>
#include "descriptions/gna_input_desc.hpp"
#include "descriptions/gna_output_desc.hpp"
#include "gna_plugin_log.hpp"
#if GNA_LIB_VER == 2
#include "gna2-model-api.h"
#endif

#pragma pack(push, 1)

/**
 * version history
 * 1.0 - basic support
 * 1.1 - added memory information
 * 2.0 - for use with GNA2 library
 * 2.1 - multiple i/o support
 */
#if GNA_LIB_VER == 2
#define HEADER_MAJOR 2
#define HEADER_MINOR 1
#else
#define HEADER_MAJOR 1
#define HEADER_MINOR 2
#endif


/**
 * @brief Header version 2.1
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

    uint32_t nInputs = 0u;
    uint32_t nOutputs = 0u;

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

        intel_dnn_orientation_t orientation = kDnnUnknownOrientation;

        RuntimeEndPoint() = default;
        RuntimeEndPoint(double scaleFactor,
                    void* descriptor_ptr,
                    uint32_t element_size,
                    uint32_t elements_count,
                    intel_dnn_orientation_t orientation) : scaleFactor(scaleFactor),
                                    descriptor_ptr(descriptor_ptr),
                                    element_size(element_size),
                                    elements_count(elements_count),
                                    orientation(orientation) {
        }
    };
    using MemoryType = std::vector<std::pair<void*, uint32_t>>;

private:
#if GNA_LIB_VER == 2
    Gna2Model * gna2Model;
#else
    intel_nnet_type_t *ptr_nnet;
#endif
    std::vector<RuntimeEndPoint> inputs;
    std::vector<RuntimeEndPoint> outputs;
    uint32_t nRotateRows = 0;
    uint32_t nRotateColumns = 0;

    MemoryType states, *pstates = nullptr;
    ModelHeader modelHeader;

    void ImportInputs(std::istream &is,
            void* basePtr,
            std::shared_ptr<GNAPluginNS::InputDesc> inputsDesc,
            InferenceEngine::InputsDataMap& dataMap);

    void ImportOutputs(std::istream &is,
            void* basePtr,
            std::vector<GNAPluginNS::OutputDesc> &desc,
            InferenceEngine::OutputsDataMap& dataMap);

 public:
#if GNA_LIB_VER == 2
    GNAModelSerial(Gna2Model * model, MemoryType & states_holder)
        : gna2Model(model), pstates(&states_holder) {
    }

    GNAModelSerial(
        Gna2Model * model,
        const std::shared_ptr<GNAPluginNS::InputDesc> inputDesc,
        const std::vector<GNAPluginNS::OutputDesc>& outputsDesc,
        const InferenceEngine::InputsDataMap& inputsDataMap,
        const InferenceEngine::OutputsDataMap& outputsDataMap) : gna2Model(model),
            inputs(serializeInputs(inputsDataMap, inputDesc)),
            outputs(serializeOutputs(outputsDataMap, outputsDesc)) {
    }

#else
     /**
  *
  * @brief Used for import/export
  * @param ptr_nnet
  * @param inputScale  - in/out parameter representing input scale factor
  * @param outputScale - in/out parameter representing output scale factor
  */
     GNAModelSerial(intel_nnet_type_t *ptr_nnet, MemoryType &states_holder)
         : ptr_nnet(ptr_nnet), pstates(&states_holder) {
     }

     /**
      * @brief used for export only since runtime params are not passed by pointer
      * @param ptr_nnet
      * @param runtime
      */
     GNAModelSerial(
         intel_nnet_type_t *ptr_nnet,
         const std::shared_ptr<GNAPluginNS::InputDesc> inputDesc,
         const std::vector<GNAPluginNS::OutputDesc>& outputsDesc,
         const InferenceEngine::InputsDataMap& inputsDataMap,
         const InferenceEngine::OutputsDataMap& outputsDataMap) : ptr_nnet(ptr_nnet),
                                                                  inputs(serializeInputs(inputsDataMap, inputDesc)),
                                                                  outputs(serializeOutputs(outputsDataMap, outputsDesc)) {
     }
#endif

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
    void Import(void *basePointer,
                                size_t gnaGraphSize,
                                std::istream & is,
                                std::shared_ptr<GNAPluginNS::InputDesc> inputsDesc,
                                std::vector<GNAPluginNS::OutputDesc> &desc,
                                InferenceEngine::InputsDataMap& inputsDataMap,
                                InferenceEngine::OutputsDataMap& outputsDataMap);

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

    static std::vector<GNAModelSerial::RuntimeEndPoint> serializeOutputs(const InferenceEngine::OutputsDataMap& outputsDataMap,
            const std::vector<GNAPluginNS::OutputDesc>& outputsDesc);


    static std::vector<GNAModelSerial::RuntimeEndPoint> serializeInputs(const InferenceEngine::InputsDataMap& inputsDataMap,
                                                                        const std::shared_ptr<GNAPluginNS::InputDesc>);

    void setHeader(ModelHeader header);
};

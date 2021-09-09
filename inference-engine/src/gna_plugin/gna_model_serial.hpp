// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <vector>
#include <utility>
#include <ie_input_info.hpp>

#include "descriptions/gna_input_desc.hpp"
#include "descriptions/gna_output_desc.hpp"
#include "gna_plugin_log.hpp"
#include "serial/headers/latest/gna_model_header.hpp"
#if GNA_LIB_VER == 2
#include "gna2-model-api.h"
#endif


/**
 * @brief implements serialisation tasks for GNAGraph
 */
class GNAModelSerial {
 public:
    using MemoryType = std::vector<std::tuple<void*, uint32_t, std::string, float>>;

private:
#if GNA_LIB_VER == 2
    Gna2Model * gna2Model;
#else
    intel_nnet_type_t *ptr_nnet;
#endif
    std::vector<GNAPluginNS::HeaderLatest::RuntimeEndPoint> inputs;
    std::vector<GNAPluginNS::HeaderLatest::RuntimeEndPoint> outputs;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    TranspositionInfoMap transposeInputsInfo;
    TranspositionInfoMap transposeOutputsInfo;

    MemoryType states, *pstates = nullptr;
    GNAPluginNS::HeaderLatest::ModelHeader modelHeader;

    void ImportInputs(std::istream &is,
            void* basePtr,
            std::shared_ptr<GNAPluginNS::InputDesc> inputsDesc,
            InferenceEngine::InputsDataMap& dataMap);

    void ImportOutputs(std::istream &is,
            void* basePtr,
            std::vector<GNAPluginNS::OutputDesc> &desc,
            InferenceEngine::OutputsDataMap& dataMap);

    void ImportTranspositionInfo(std::istream &is,
            std::string &name,
            std::vector<TranspositionInfo> &transpositionInfo);

    void ExportTranspositionInfo(std::ostream &os,
            const TranspositionInfoMap &transpositionInfoMap) const;

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
        for (auto const& input : inputsDataMap) {
            inputNames.push_back(input.first);
        }

        for (auto const& input : outputsDataMap) {
            outputNames.push_back(input.first);
        }
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

    GNAModelSerial & SetInputRotation(const TranspositionInfoMap &transposeInputsInfo) {
      this->transposeInputsInfo = transposeInputsInfo;
      return *this;
    }

    GNAModelSerial& SetOutputRotation(const TranspositionInfoMap &transposeOutputsInfo) {
        this->transposeOutputsInfo = transposeOutputsInfo;
        return *this;
    }

    /**
     * mark certain part of gna_blob as state (in future naming is possible)
     * @param descriptor_ptr
     * @param size
     * @param layerName
     * @return
     */
    GNAModelSerial & AddState(void* descriptor_ptr, size_t size, std::string layerName = "noname", float scale_factor = 1.0f) {
        states.emplace_back(descriptor_ptr, size, layerName, scale_factor);
        return *this;
    }

    /**
     * @brief calculate memory required for import gna graph
     * @param is - opened input stream
     * @return
     */
    static GNAPluginNS::HeaderLatest::ModelHeader ReadHeader(std::istream &is);

    GNAPluginNS::HeaderLatest::RuntimeEndPoint ReadEndPoint(std::istream &is);

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
                InferenceEngine::OutputsDataMap& outputsDataMap,
                TranspositionInfoMap& inputstranspositionInfo,
                TranspositionInfoMap& outputstranspositionInfo);

    /**
     * save gna graph to an outpus stream
     * @param basePtr
     * @param gnaGraphSize
     * @param os
     */
    void Export(void *basePtr,
                size_t gnaGraphSize,
                std::ostream &os) const;

    static std::vector<GNAPluginNS::HeaderLatest::RuntimeEndPoint> serializeOutputs(const InferenceEngine::OutputsDataMap& outputsDataMap,
            const std::vector<GNAPluginNS::OutputDesc>& outputsDesc);


    static std::vector<GNAPluginNS::HeaderLatest::RuntimeEndPoint> serializeInputs(const InferenceEngine::InputsDataMap& inputsDataMap,
                                                                        const std::shared_ptr<GNAPluginNS::InputDesc>);

    void setHeader(GNAPluginNS::HeaderLatest::ModelHeader header);
};

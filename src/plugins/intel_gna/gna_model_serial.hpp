// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <vector>
#include <utility>
#include <ie_input_info.hpp>

#include "descriptions/gna_desc.hpp"
#include "gna_plugin_log.hpp"
#include "serial/headers/latest/gna_model_header.hpp"
#include "gna2-model-api.h"

#include "gna_device_allocation.hpp"

/**
 * @brief helper class for GNAGraph serialization tasks
 */
class GNAVersionSerializer {
public:
    void Export(std::ostream& os) const;
    std::string Import(std::istream& is) const;
};

/**
 * @brief implements serialization tasks for GNAGraph
 */
class GNAModelSerial {
public:
    using MemoryType = std::vector<std::tuple<void*, uint32_t, std::string, float>>;

private:
    Gna2Model * gna2model_;
    MemoryType states, *pstates_ = nullptr;
    GNAPluginNS::GnaInputs inputs_;
    GNAPluginNS::GnaOutputs outputs_;
    TranspositionInfoMap inputs_transpose_info_;
    TranspositionInfoMap outputs_transpose_info_;
    GNAPluginNS::HeaderLatest::ModelHeader model_header_;
    GNAVersionSerializer version_;

    void ImportInputs(std::istream &is, void* basePtr, GNAPluginNS::GnaInputs &inputs);

    void ImportOutputs(std::istream &is, void* basePtr, GNAPluginNS::GnaOutputs &outputs);

    void ImportTranspositionInfo(std::istream &is, std::string &name, std::vector<TranspositionInfo> &transpositionInfo);

    void ExportTranspositionInfo(std::ostream &os, const TranspositionInfoMap &transpositionInfoMap) const;

    /**
     * @brief Update input or output description to support importing of < 2.8 format where tensor_names were not present
     * @param nodeDesc input or output description to be appended
     */
    void AppendTensorNameIfNeeded(GNAPluginNS::GnaDesc& nodeDesc) const;

 public:
    GNAModelSerial(Gna2Model* model, MemoryType& states_holder)
         : gna2model_(model),
           pstates_(&states_holder) {
    }

    GNAModelSerial(Gna2Model* model,
                   GNAPluginNS::GnaInputs& inputs,
                   GNAPluginNS::GnaOutputs& outputs)
        : gna2model_(model),
          inputs_(inputs),
          outputs_(outputs) {
    }

    void setHeader(GNAPluginNS::HeaderLatest::ModelHeader header) {
        model_header_ = header;
    }

    GNAModelSerial & SetInputRotation(const TranspositionInfoMap &transpose_inputs_info) {
      inputs_transpose_info_ = transpose_inputs_info;
      return *this;
    }

    GNAModelSerial & SetOutputRotation(const TranspositionInfoMap &transpose_outputs_info) {
        outputs_transpose_info_ = transpose_outputs_info;
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
        states.emplace_back(descriptor_ptr, static_cast<uint32_t>(size), layerName, scale_factor);
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
                std::istream &is,
                GNAPluginNS::GnaInputs &inputs,
                GNAPluginNS::GnaOutputs &outputs,
                TranspositionInfoMap& inputstranspositionInfo,
                TranspositionInfoMap& outputstranspositionInfo,
                std::string& modelLibVersion);

    /**
     * save gna graph to an outpus stream
     * @param allocations
     * @param os
     */
    void Export(const GnaAllocations& allocations,
                std::ostream &os) const;
};

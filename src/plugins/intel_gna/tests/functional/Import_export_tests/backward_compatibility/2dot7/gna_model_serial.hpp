// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <vector>
#include <utility>

#include "gna_plugin.hpp"
#include "descriptions/gna_desc.hpp"
#include "request/worker_pool_impl.hpp"
#include "memory/gna_memory_state.hpp"
#include "serial/headers/2dot7/gna_model_header.hpp"
#include "gna2-model-api.h"

#include "gna_device_allocation.hpp"

namespace ov {
namespace intel_gna {
namespace header_2_dot_7 {

/**
 * @brief implements serialisation tasks for GNAGraph
 */
class GNAModelSerial {
 public:
    using MemoryType = std::vector<std::tuple<void*, uint32_t, std::string, float>>;

private:
    Gna2Model * gna2Model;
    std::vector<RuntimeEndPoint> inputs;
    std::vector<RuntimeEndPoint> outputs;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    TranspositionInfoMap transposeInputsInfo;
    TranspositionInfoMap transposeOutputsInfo;

    MemoryType states, *pstates = nullptr;
    ModelHeader modelHeader;

    void ExportTranspositionInfo(std::ostream &os,
            const TranspositionInfoMap &transpositionInfoMap) const;

 public:
    GNAModelSerial(Gna2Model * model, MemoryType & states_holder)
        : gna2Model(model), pstates(&states_holder) {
    }

    GNAModelSerial(
        Gna2Model * model,
        const std::vector<ov::intel_gna::InputDesc>& inputDesc,
        const std::vector<ov::intel_gna::OutputDesc>& outputsDesc,
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
     * save gna graph to an outpus stream
     * @param basePtr
     * @param gnaGraphSize
     * @param os
     */
    void Export(const GnaAllocations& allocations, std::ostream &os) const;

    static std::vector<RuntimeEndPoint> serializeOutputs(const InferenceEngine::OutputsDataMap& outputsDataMap,
            const std::vector<ov::intel_gna::OutputDesc>& outputsDesc);


    static std::vector<RuntimeEndPoint> serializeInputs(const InferenceEngine::InputsDataMap& inputsDataMap,
                                                        const std::vector<ov::intel_gna::InputDesc>& inputsDesc);
};

class GNAPluginLegacy : public GNAPlugin {
public:
    void Export(const std::string& fileName) {
        GNAPlugin::Export(fileName);
    }

    void Export(std::ostream &outStream) override {
    if (inputs_ptr_->empty() || outputs_.empty()) {
        THROW_GNA_EXCEPTION << " network not loaded";
    }

    IE_ASSERT(!inputs_data_map_.empty());

    Gna2Model* model_to_serial = requestWorkerPool_->firstWorker().model();
    auto serial = ov::intel_gna::header_2_dot_7::GNAModelSerial(model_to_serial,
                                 inputs_ptr_->Get(),
                                 outputs_.Get(),
                                 inputs_data_map_,
                                 outputs_data_map_)
                    .SetInputRotation(transpose_inputs_info)
                    .SetOutputRotation(transpose_outputs_info);

    for (auto && memoryConnection : graphCompiler.memory_connection) {
        auto state = std::make_shared<memory::GNAVariableState>(memoryConnection.first, std::make_shared <GNAMemoryLayer>(memoryConnection.second));
        log::debug() << "Scale factor Memory layer " << state->GetScaleFactor() << std::endl;
        serial.AddState(memoryConnection.second.gna_ptr, memoryConnection.second.reserved_size, memoryConnection.first, state->GetScaleFactor());
    }

    serial.Export(gnadevice->getAllAllocations(), outStream);
}
};

}  // namespace header_2_dot_8
}  // namespace intel_gna
}  // namespace ov
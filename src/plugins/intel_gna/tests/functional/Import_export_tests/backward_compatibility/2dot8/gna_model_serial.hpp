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
#include "serial/headers/2dot8/gna_model_header.hpp"
#include "gna2-model-api.h"

#include "gna_device_allocation.hpp"

namespace ov {
namespace intel_gna {
namespace header_2_dot_8 {

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
    ov::intel_gna::GnaInputs inputs_;
    ov::intel_gna::GnaOutputs outputs_;
    TranspositionInfoMap inputs_transpose_info_;
    TranspositionInfoMap outputs_transpose_info_;
    ov::intel_gna::header_2_dot_8::ModelHeader model_header_;
    GNAVersionSerializer version_;

    void AppendTensorNameIfNeeded(ov::intel_gna::GnaDesc& nodeDesc) const;

 public:
    GNAModelSerial(Gna2Model* model, MemoryType& states_holder)
         : gna2model_(model),
           pstates_(&states_holder) {
    }

    GNAModelSerial(Gna2Model* model,
                   ov::intel_gna::GnaInputs& inputs,
                   ov::intel_gna::GnaOutputs& outputs)
        : gna2model_(model),
          inputs_(inputs),
          outputs_(outputs) {
    }

    void setHeader(ov::intel_gna::header_2_dot_8::ModelHeader header) {
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
     * save gna graph to an outpus stream
     * @param allocations
     * @param os
     */
    void Export(const GnaAllocations& allocations,
                std::ostream &os) const;

    void ExportTranspositionInfo(std::ostream &os, const TranspositionInfoMap &transpositionInfoMap) const;

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
    auto serial = ov::intel_gna::header_2_dot_8::GNAModelSerial(model_to_serial,
                                 *(inputs_ptr_),
                                 outputs_)
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
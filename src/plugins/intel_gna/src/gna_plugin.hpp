// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-model-api.h>

#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <legacy/ie_util_internal.hpp>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/am_intel_dnn.hpp"
#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"
#include "descriptions/gna_desc.hpp"
#include "descriptions/gna_flags.hpp"
#include "gna_data_types.hpp"
#include "gna_graph_compiler.hpp"
#include "gna_plugin_config.hpp"
#include "log/debug.hpp"
#include "log/log.hpp"
#include "pre_post_process/input_output_data_handler.hpp"
#include "pre_post_process/transposition_info.hpp"

namespace ov {
namespace intel_gna {
namespace request {
class ModelWrapper;
class WorkerPool;
class Worker;
}  // namespace request

using namespace ov::intel_gna::pre_post_processing;

class GNAPlugin : public InferenceEngine::IInferencePlugin {
protected:
    std::string _pluginName = "GNA";

    Config config{};
    std::shared_ptr<backend::AMIntelDNN> dnn;
    std::shared_ptr<GNAFlags> gnaFlags;
    std::shared_ptr<gna_memory_type> gnamem;
    std::shared_ptr<GnaInputs> inputs_ptr_;
    GnaOutputs outputs_;
    std::shared_ptr<GNAGraphCompiler> m_graph_compiler;
    pre_post_processing::InputOutputDataHandler m_input_output_handler;

    uint32_t activeLayerIndex = 0xffffffff;
    // TODO: transpose_inputs_info and transpose_outputs_info should be moved to GNAModelSerial class when ngraph
    // migration is finished. Those structures are needed to support the exported models <= 2.8.
    pre_post_processing::TranspositionInfoMap transpose_inputs_info;
    pre_post_processing::TranspositionInfoMap transpose_outputs_info;
    PrePostProcessModels m_input_output_subgraphs;

    uint32_t dnn_dump_write_index = 0;
    intel_dnn_number_type_t output_type = kDnnInt;

    std::shared_ptr<GNADeviceHelper> gnadevice;

    std::shared_ptr<request::WorkerPool> requestWorkerPool_;

    /**
     * @brief size of RW segment without extra memory for parallel execution
     */
    size_t rwSegmentSize = 0;

    InferenceEngine::InputsDataMap inputs_data_map_;    //!< Holds information about network inputs info
    InferenceEngine::OutputsDataMap outputs_data_map_;  //!< Holds information about network outputs data

    std::string _network_name;

    std::vector<InferenceEngine::IVariableStateInternal::Ptr> memoryStates;
    bool trivialTopology = false;

public:
    explicit GNAPlugin(const std::map<std::string, std::string>& configMap);
    /**
     * @brief construct from aot rather then from cnn network
     */
    GNAPlugin();

    GNAPlugin(const GNAPlugin&) = delete;
    GNAPlugin(GNAPlugin&&) = default;

    std::string GetName() const noexcept override;
    void SetName(const std::string& pluginName) noexcept override;

    using InferenceEngine::IInferencePlugin::LoadNetwork;
    void LoadNetwork(const InferenceEngine::CNNNetwork& network);

    bool Infer(const InferenceEngine::BlobMap& input, InferenceEngine::BlobMap& result);
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts();
    void AddExtension(const InferenceEngine::IExtensionPtr& extension) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;
    bool Infer(const InferenceEngine::Blob& input, InferenceEngine::Blob& result);
    void Reset();
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;
    uint32_t QueueInference(const InferenceEngine::BlobMap& input, InferenceEngine::BlobMap& result);
    bool Wait(uint32_t idx);
    RequestStatus WaitFor(uint32_t idx, int64_t millisTimeout);

    InferenceEngine::Parameter GetConfig(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::Parameter GetMetric(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    std::shared_ptr<InferenceEngine::RemoteContext> CreateContext(const InferenceEngine::ParamMap& params) override {
        THROW_GNA_EXCEPTION << "Not implemented";
    }
    std::shared_ptr<InferenceEngine::RemoteContext> GetDefaultContext(const InferenceEngine::ParamMap&) override {
        THROW_GNA_EXCEPTION << "Not implemented";
    }

    void Wait(uint32_t sync, InferenceEngine::Blob& result) {
        THROW_GNA_EXCEPTION << "Not implemented";
    }

    void Export(const std::string& fileName);
    void Export(std::ostream& networkModel);
    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
        const std::string& modelFileName,
        const std::map<std::string, std::string>& config) override {
        THROW_GNA_EXCEPTION << "Not implemented";
    }
    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
        std::istream& networkModel,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context,
        const std::map<std::string, std::string>& config) override {
        THROW_GNA_EXCEPTION << "Not implemented";
    }

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
        std::istream& networkModel,
        const std::map<std::string, std::string>& config) override {
        THROW_GNA_EXCEPTION << "Not implemented";
    }

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(std::istream& networkModel);

    /**
     * utility to provide input and output blobs externally to be used by InferenceEngine request API clients
     */
    InferenceEngine::Blob::Ptr GetInputBlob(const std::string& name, InferenceEngine::Precision precision);
    InferenceEngine::Blob::Ptr GetOutputBlob(const std::string& name, InferenceEngine::Precision precision);
    /**
     * helpers to provide inputs info on AOT network
     */
    InferenceEngine::InputsDataMap GetNetworkInputs() {
        return inputs_data_map_;
    }
    InferenceEngine::OutputsDataMap GetNetworkOutputs() {
        return outputs_data_map_;
    }
    std::vector<std::shared_ptr<const ov::Node>> GetOutputs();
    std::vector<std::shared_ptr<const ov::Node>> GetInputs();
    /**
     * helpers to set inputs/output info on AOT network
     */
    void SetNetworkInputs();
    void SetNetworkOutputs();
    /**
     * helpers to update internal inputs/output descriptions from loaded network
     */
    void UpdateInputs(const std::vector<std::shared_ptr<const ov::Node>>& params);
    void UpdateOutputs(const std::vector<std::shared_ptr<const ov::Node>>& results);
    /**
     * QueryState API
     * @return
     */
    INFERENCE_ENGINE_DEPRECATED("Use InferRequest::QueryState instead")
    std::vector<InferenceEngine::IVariableStateInternal::Ptr> QueryState();

    /**
     * QueryMetrics API
     */
    InferenceEngine::Parameter GetAvailableDevices() const;

    ~GNAPlugin();

protected:
    void Init();

    void InitGNADevice();
    void InitGNAMemory();
    void InitGraphCompiler();

    void DumpXNNToFile() const;
    /**
     * @brief Run ngraph model on CPU to modify input or output (transposing, gathering)
     * Method supports only models with 1 input and 1 output.
     * @param input_blob input blob memory
     * @param output_blob output blob memory
     * @param model ngraph function needs to be executed to modify input blob and put result to the output blob
     * @return void
     */
    void PrePostProcess(InferenceEngine::Blob::Ptr input_blob,
                        InferenceEngine::Blob::Ptr output_blob,
                        std::shared_ptr<ov::Model> model);

    /**
     * Run ngraph model on CPU to modify inputs/outputs
     */
    void pre_post_process(InferenceEngine::Blob::Ptr input_blob,
                          InferenceEngine::Blob::Ptr output_blob,
                          std::shared_ptr<ov::Model> model);
    void UpdateFieldsFromConfig();
    void UpdateInputScaleFromNetwork(InferenceEngine::CNNNetwork& network);
    void UpdateInputsAndOutputsInfoFromNetwork(InferenceEngine::CNNNetwork&);
    void UpdateInputsAndOutputsInfoFromModel(std::shared_ptr<const ov::Model> model);
    /**
     * @brief Tries to init an output on the base of a layer data
     * @param portId output port identificator
     * @param layer layer pointer
     * @return true if the output is initiated, false otherwise
     */
    bool TryToInitOutput(const std::string& portName, InferenceEngine::CNNLayerPtr layer);
    bool isFP32ModeActive() const;
    std::shared_ptr<request::ModelWrapper> createModelWrapperForLoadNetwork(bool trivial);
    std::shared_ptr<request::ModelWrapper> createModelWrapperForImportNetwork(uint32_t numberOfOperations);
    std::shared_ptr<request::Worker> createWorkerForLoadNetwork(bool trivial, bool fp32Mode);
    std::shared_ptr<request::Worker> createWorker(std::shared_ptr<request::ModelWrapper> modelWrapper,
                                                  bool trivial,
                                                  bool fp32Mode);

#ifdef PLOT
    void AddDebugProperties(const InferenceEngine::CNNLayerPtr layer,
                            InferenceEngine::ordered_properties& printed_properties,
                            InferenceEngine::ordered_properties& node_properties);
#endif
};

}  // namespace intel_gna
}  // namespace ov

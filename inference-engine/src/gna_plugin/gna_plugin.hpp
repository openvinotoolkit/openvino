// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <unordered_map>
#include <list>
#include <string>
#include <utility>
#include <memory>
#include <vector>
#include <tuple>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"
#include "descriptions/gna_flags.hpp"
#include "descriptions/gna_input_desc.hpp"
#include "descriptions/gna_output_desc.hpp"
#include "backend/am_intel_dnn.hpp"
#include "gna_data_types.hpp"
#include "gna_graph_compiler.hpp"
#include "gna_plugin_log.hpp"
#include "gna_plugin_config.hpp"
#include <legacy/ie_util_internal.hpp>

#if GNA_LIB_VER == 2
#include <gna2-model-api.h>
#endif

namespace GNAPluginNS {
class GNAPlugin : public InferenceEngine::IInferencePlugin {
 protected:
    std::string _pluginName = "GNA";

    Config config {};
    std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnn;
    std::shared_ptr<GNAPluginNS::GNAFlags> gnaFlags;
    std::shared_ptr<GNAPluginNS::gna_memory_type> gnamem;
    std::shared_ptr<GNAPluginNS::InputDesc> inputsDesc;

    GNAPluginNS::GNAGraphCompiler graphCompiler;

    /**
     * @brief - copy of nnet structure and indicator that related infer request not yet synced
     */
#if GNA_LIB_VER == 1
    std::vector<std::tuple<dnn_ptr, int32_t, InferenceEngine::BlobMap>> nnets;
#else
    static constexpr uint32_t FAKE_REQUEST_CONFIG_ID = 0xffffffff;
    std::vector<std::tuple<dnn_ptr>> gnaModels;
    std::vector<std::tuple<uint32_t, int64_t, InferenceEngine::BlobMap>> gnaRequestConfigToRequestIdMap;
#endif

#if GNA_LIB_VER == 2
    uint32_t activeLayerIndex = 0xffffffff;
#endif
    TranspositionInfoMap transpose_inputs_info;
    TranspositionInfoMap transpose_outputs_info;
    uint32_t *ptr_active_indices = nullptr;
    uint32_t num_active_indices = 0;
    uint32_t num_group_in = 0;
    uint32_t dnn_dump_write_index = 0;

    // index matches iterating order of cnnnetwork outputs info
    std::vector<GNAPluginNS::OutputDesc> outputsDesc = std::vector<OutputDesc>();

    intel_dnn_number_type_t output_type = kDnnInt;

#if GNA_LIB_VER == 2
    void createRequestConfigsForGnaModels();
#endif

    static int GetDeviceVersionFromString(const std::string deviceString);

    std::shared_ptr<GNADeviceHelper> gnadevice;
    /**
     * @brief size of RW segment without extra memory for parallel execution
     */
    uint32_t rwSegmentSize = 0;

    InferenceEngine::InputsDataMap inputsDataMap;
    InferenceEngine::OutputsDataMap outputsDataMap;
    std::vector<InferenceEngine::IVariableStateInternal::Ptr> memoryStates;
    bool trivialTopology = false;

 public:
    explicit GNAPlugin(const std::map<std::string, std::string>& configMap);
    /**
     * @brief construct from aot rather then from cnn network
     */
    GNAPlugin();

    std::string GetName() const noexcept override;
    void SetName(const std::string & pluginName) noexcept override;

    void LoadNetwork(InferenceEngine::CNNNetwork &network);

    bool Infer(const InferenceEngine::BlobMap &input, InferenceEngine::BlobMap &result);
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts();
    void AddExtension(const InferenceEngine::IExtensionPtr& extension) override;

    void SetConfig(const std::map<std::string, std::string> &config) override;
    bool Infer(const InferenceEngine::Blob &input, InferenceEngine::Blob &result);
    void Reset();
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork &network,
                                                     const std::map<std::string, std::string>& config) const override;
    uint32_t QueueInference(const InferenceEngine::BlobMap &input, InferenceEngine::BlobMap &result);
    bool Wait(uint32_t idx);
    GnaWaitStatus WaitFor(uint32_t idx, int64_t millisTimeout);

    InferenceEngine::Parameter GetConfig(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::RemoteContext::Ptr CreateContext(const InferenceEngine::ParamMap& params) override { THROW_GNA_EXCEPTION << "Not implemented"; }
    InferenceEngine::RemoteContext::Ptr GetDefaultContext(const InferenceEngine::ParamMap&) override { THROW_GNA_EXCEPTION << "Not implemented"; }

    void Wait(uint32_t sync, InferenceEngine::Blob &result) { THROW_GNA_EXCEPTION << "Not implemented"; }

    void Export(const std::string &fileName);
    void Export(std::ostream &networkModel);
    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(const std::string &modelFileName,
                                                     const std::map<std::string, std::string> &config) override {
        THROW_GNA_EXCEPTION << "Not implemented";
    }
    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(std::istream& networkModel,
                                                     const InferenceEngine::RemoteContext::Ptr& context,
                                                     const std::map<std::string, std::string> &config) override {
        THROW_GNA_EXCEPTION << "Not implemented";
    }

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(std::istream& networkModel,
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
    InferenceEngine::InputsDataMap GetInputs() {return inputsDataMap;}
    InferenceEngine::OutputsDataMap GetOutputs() {return outputsDataMap;}
    /**
     * QueryState API
     * @return
     */
    INFERENCE_ENGINE_DEPRECATED("Use InferRequest::QueryState instead")
    std::vector<InferenceEngine::IVariableStateInternal::Ptr>  QueryState();

     /**
      * QueryMetrics API
      */

     InferenceEngine::Parameter GetAvailableDevices() const;

 protected:
    void Init();

    void InitGNADevice();

    void DumpXNNToFile() const;

    void ImportFrames(void *ptr_dst,
                     const void *ptr_src,
                     InferenceEngine::Precision input_precision,
                     float scaleFactor,
                     intel_dnn_orientation_t orientation,
                     uint32_t num_frames,
                     uint32_t num_group,
                     uint32_t num_vector_elements,
                     uint32_t num_vector_stride);

    void ExportScores(void *ptr_dst,
                     const void *ptr_src,
                     intel_dnn_orientation_t orientation,
                     uint32_t num_frames,
                     uint32_t num_group,
                     uint32_t num_vector_elements,
                     uint32_t num_active_elements,
                     uint32_t num_vector_stride,
                     uint32_t num_bytes_per_element_input,
                     uint32_t num_bytes_per_element);

    template <typename T, typename U>
    void copyInputData(T *dst,
                    const U *src,
                    uint32_t num_frames,
                    uint32_t num_group,
                    uint32_t num_vector_elements,
                    uint32_t num_vector_stride,
                    intel_dnn_orientation_t orientation,
                    float scaleFactor);

    template <typename T, typename U>
    void copyInputDataWithSplit(T *const dst,
                    const U *src,
                    const GNASplitLayer& splitInfo,
                    size_t precision_size,
                    int idx = 0);

    void UpdateFieldsFromConfig();
    void UpdateGnaQuantModeFromNetwork(InferenceEngine::CNNNetwork &);
    void UpdateInputScaleFromNetwork(InferenceEngine::CNNNetwork &);
    /**
     * @brief Tries to init an output on the base of a layer data
     * @param portId output port identificator
     * @param layer layer pointer
     * @return true if the output is initiated, false otherwise
    */
    bool TryToInitOutput(int portId, InferenceEngine::CNNLayerPtr layer);

    /**
     * @brief Fills inputs and outputs transposition info for model convertion from NCHW to NHWC.
     *        Information for transposition is found from convolution/pooling input or output dimensions.
     * @param layers model sorted layers
     */
    void FillInputsAndOutputsTranspositionInfo(const InferenceEngine::CNNNetwork& net);
#ifdef PLOT
    void AddDebugProperties(const InferenceEngine::CNNLayerPtr layer,
        InferenceEngine::ordered_properties& printed_properties,
        InferenceEngine::ordered_properties& node_properties);
#endif
};

}  // namespace GNAPluginNS

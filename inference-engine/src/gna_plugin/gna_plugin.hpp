// Copyright (C) 2018-2020 Intel Corporation
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
#include <cpp_interfaces/interface/ie_imemory_state_internal.hpp>
#include "descriptions/gna_flags.hpp"
#include "descriptions/gna_input_desc.hpp"
#include "descriptions/gna_output_desc.hpp"
#include "backend/am_intel_dnn.hpp"
#include "gna_data_types.hpp"
#include "gna_graph_compiler.hpp"
#include "gna_plugin_policy.hpp"
#include "gna_plugin_log.hpp"
#include "gna_plugin_config.hpp"

#if GNA_LIB_VER == 2
#include <gna2-model-api.h>
#endif

namespace GNAPluginNS {
class GNAPlugin : public InferenceEngine::IInferencePluginInternal, public std::enable_shared_from_this<GNAPlugin> {
 protected:
    std::string _pluginName = "GNA";

    Config config;
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
    uint32_t num_rotate_rows = 0;
    uint32_t num_rotate_columns = 0;
    uint32_t *ptr_active_indices = nullptr;
    uint32_t num_active_indices = 0;
    uint32_t num_group_in = 0;
    uint32_t dnn_dump_write_index = 0;

    // index matches iterating order of cnnnetwork outputs info
    std::vector<GNAPluginNS::OutputDesc> outputsDesc = std::vector<OutputDesc>();

    intel_dnn_number_type_t output_type = kDnnInt;

    GNAPluginNS::Policy policy;

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

 public:
    explicit GNAPlugin(const std::map<std::string, std::string>& configMap);
    /**
     * @brief construct from aot rather then from cnn network
     */
    GNAPlugin();

    std::string GetName() const noexcept override;
    void SetName(const std::string & pluginName) noexcept override;

    void LoadNetwork(InferenceEngine::ICNNNetwork &network);

    void Infer(const InferenceEngine::BlobMap &input, InferenceEngine::BlobMap &result);
    void GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap);
    void AddExtension(InferenceEngine::IExtensionPtr extension) override;

    void SetConfig(const std::map<std::string, std::string> &config) override;
    void LoadNetwork(InferenceEngine::IExecutableNetwork::Ptr &executableNetwork,
                     const InferenceEngine::ICNNNetwork &network,
                     const std::map<std::string, std::string> &config_map) override { THROW_GNA_EXCEPTION << "Not implemented"; }
    InferenceEngine::ExecutableNetwork LoadNetwork(const InferenceEngine::ICNNNetwork &network,
                                  const std::map<std::string, std::string> &config_map,
                                  InferenceEngine::RemoteContext::Ptr context) override { THROW_GNA_EXCEPTION << "Not implemented"; }
    void Infer(const InferenceEngine::Blob &input, InferenceEngine::Blob &result);
    void SetCore(InferenceEngine::ICore*) noexcept override {}
    InferenceEngine::ICore* GetCore() const noexcept override {return nullptr;}
    void Reset();
    void QueryNetwork(const InferenceEngine::ICNNNetwork &network,
                      const std::map<std::string, std::string>& config,
                      InferenceEngine::QueryNetworkResult &res) const override;
    uint32_t QueueInference(const InferenceEngine::BlobMap &input, InferenceEngine::BlobMap &result);
    void Wait(uint32_t idx = 0);

    InferenceEngine::Parameter GetConfig(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::RemoteContext::Ptr CreateContext(const InferenceEngine::ParamMap& params) override { THROW_GNA_EXCEPTION << "Not implemented"; }
    InferenceEngine::RemoteContext::Ptr GetDefaultContext() override { THROW_GNA_EXCEPTION << "Not implemented"; }

    void Wait(uint32_t sync, InferenceEngine::Blob &result) { THROW_GNA_EXCEPTION << "Not implemented"; }

    void Export(const std::string &fileName);
    InferenceEngine::IExecutableNetwork::Ptr ImportNetwork(const std::string &modelFileName,
                                                           const std::map<std::string, std::string> &config) override {
        THROW_GNA_EXCEPTION << "Not implemented";
    }
    InferenceEngine::ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                                     const InferenceEngine::RemoteContext::Ptr& context,
                                                     const std::map<std::string, std::string> &config) override {
        THROW_GNA_EXCEPTION << "Not implemented";
    }
    InferenceEngine::ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                                     const std::map<std::string, std::string> &config) override {
        THROW_GNA_EXCEPTION << "Not implemented";
    }

    InferenceEngine::IExecutableNetwork::Ptr ImportNetwork(const std::string &modelFileName);

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
     std::vector<InferenceEngine::IMemoryStateInternal::Ptr>  QueryState();

     /**
      * test-wise API
      */
     void SetPolicy(GNAPluginNS::Policy p) {policy = p;}

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
};

}  // namespace GNAPluginNS

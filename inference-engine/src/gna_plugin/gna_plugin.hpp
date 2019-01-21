// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp_interfaces/base/ie_plugin_base.hpp"
#include "dnn.h"
#include "gna_memory.hpp"
#include "gna_device.hpp"
#include <map>
#include <list>
#include <string>
#include <utility>
#include <memory>
#include <vector>
#include <tuple>
#include <gna-api-status.h>
#include <gna-api.h>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <graph_tools.hpp>
#include "gna_allocator.hpp"
#include "gna_api_wrapper.hpp"

namespace GNAPluginNS {

void ConvertToInt16(int16_t *ptr_dst,
                    const float *ptr_src,
                    const uint32_t num_rows,
                    const uint32_t num_columns,
                    const float scale_factor);
void ConvertToFloat(float *ptr_dst,
                    int32_t *ptr_src,
                    const uint32_t num_rows,
                    const uint32_t num_columns,
                    const float scale_factor);

int16_t ConvertFloatToInt16(float src);

class GNAPlugin : public InferenceEngine::IInferencePluginInternal, public std::enable_shared_from_this<GNAPlugin> {
 protected:
    AmIntelDnn dnn;
    using dnn_ptr = std::shared_ptr<CPPWrapper<intel_nnet_type_t>>;

    /**
     * @brief - copy of nnet structure and indicator that related infer request not yet synced
     */
    std::vector<std::tuple<dnn_ptr, int32_t, InferenceEngine::BlobMap>> nnets;

    intel_dnn_orientation_t orientation_in = kDnnUnknownOrientation;
    intel_dnn_orientation_t orientation_out = kDnnUnknownOrientation;
    double input_scale_factor = 1.0;
    double output_scale_factor = 1.0;
    uint32_t num_rotate_rows = 0;
    uint32_t num_rotate_columns = 0;


    uint32_t num_feature_maps = 1;
    uint32_t num_memory_bytes;

    std::vector<void *> ptr_inputs_global;
    std::vector<void *> ptr_outputs_global;

    int16_t *ptr_int_inputs = NULL;
    int32_t *ptr_int_outputs = NULL;
    uint32_t *ptr_active_indices = NULL;
    uint32_t num_active_indices = 0;
    uint32_t num_group_in = 0;
    uint32_t num_bytes_weight;
    uint32_t num_bytes_per_output = 0;

    bool use_dynamic_quantization = false;
    bool compact_mode = true;
    bool exclusive_async_requests = false;
    bool uniformPwlDesign = false;
    uint8_t gna_lib_async_threads_num = 1;
    bool gna_openmp_multithreading = false;
    // precision of GNA hardware model
    InferenceEngine::Precision gnaPrecision = InferenceEngine::Precision::I16;

    bool performance_counting = false;
    int  bytes_alllocated_for_input = 0;
    intel_dnn_number_type_t output_type = kDnnInt;
    std::string utterance_name;

    // internal types
    enum LayerType {
        Input,
        Convolution,
        ReLU,
        LeakyReLU,
        Sigmoid,
        TanH,
        Activation,
        Pooling,
        FullyConnected,
        InnerProduct,
        Reshape,
        Split,
        Slice,
        Eltwise,
        ScaleShift,
        Clamp,
        Concat,
        Copy,
        Permute,
        Memory,
        Power,
        Crop,
        NO_TYPE
    };

 public:
    explicit GNAPlugin(const std::map<std::string, std::string>& configMap);
    /**
     * @brief construct from aot rather then from cnn network
     */
    GNAPlugin() = default;

    void LoadNetwork(InferenceEngine::ICNNNetwork &network) override;
    using InferenceEngine::IInferencePluginInternal::Infer;

    void Infer(const InferenceEngine::BlobMap &input, InferenceEngine::BlobMap &result) override;
    void GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) override;
    void AddExtension(InferenceEngine::IExtensionPtr extension) override;
    void SetConfig(const std::map<std::string, std::string> &config) override;
    void LoadNetwork(InferenceEngine::IExecutableNetwork::Ptr &executableNetwork,
                     InferenceEngine::ICNNNetwork &network,
                     const std::map<std::string, std::string> &config) override { THROW_GNA_EXCEPTION << "Not implemented"; }
    void Infer(const InferenceEngine::Blob &input, InferenceEngine::Blob &result) override;
    void SetLogCallback(InferenceEngine::IErrorListener &listener) override {};
    void Reset();
    /**
     * @deprecated Use the version with config parameter
     */
    void QueryNetwork(const InferenceEngine::ICNNNetwork &network,
                      InferenceEngine::QueryNetworkResult &res) const override { }
    void QueryNetwork(const InferenceEngine::ICNNNetwork &network,
                      const std::map<std::string, std::string>& config,
                      InferenceEngine::QueryNetworkResult &res) const override { }
    uint32_t QueueInference(const InferenceEngine::BlobMap &input, InferenceEngine::BlobMap &result);
    void Wait(uint32_t idx = 0);

    uint32_t QueueInference(const InferenceEngine::Blob &input, InferenceEngine::BlobMap &result);
    /**
     *
     * @param sync - points to gna sync point
     * @param idx - points to
     * @param result
     */
    void Wait(uint32_t sync, InferenceEngine::Blob &result);

    void Export(const std::string &fileName);
    InferenceEngine::IExecutableNetwork::Ptr ImportNetwork(const std::string &modelFileName
        , const std::map<std::string, std::string> &config) override { THROW_GNA_EXCEPTION << "Not implemented"; }
    InferenceEngine::IExecutableNetwork::Ptr ImportNetwork(const std::string &modelFileName);


    bool IsExclusiveAsyncRequests() { return exclusive_async_requests; }

    /**
     * utility to provide input and output blobs externally to be used by InferenceEngine request API clients
     */
    InferenceEngine::Blob::Ptr GetInputBlob(InferenceEngine::Precision precision);
    InferenceEngine::Blob::Ptr GetOutputBlob(InferenceEngine::Precision precision);
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

 protected:
    uint32_t num_cnn_rows_out = 0;
    bool done = false;
    std::string dumpXNNPath;
    intel_gna_proc_t gna_proc_type = static_cast<intel_gna_proc_t>(GNA_SOFTWARE & GNA_HARDWARE);

    void DumpXNNToFile() const;
    void CreateLayerPrimitive(InferenceEngine::CNNLayerPtr);
    void AffinePrimitive(InferenceEngine::CNNLayerPtr, bool isDiag = false);
    void DiagonalPrimitive(InferenceEngine::CNNLayerPtr);
    void ConvolutionPrimitive(InferenceEngine::CNNLayerPtr);
    void PermutePrimitive(InferenceEngine::CNNLayerPtr);
    void PoolingPrimitive(InferenceEngine::CNNLayerPtr);
    void PowerPrimitive(InferenceEngine::CNNLayerPtr);
    void ConcatPrimitive(InferenceEngine::CNNLayerPtr);
    void CropPrimitive(InferenceEngine::CNNLayerPtr);
    void EltwisePrimitive(InferenceEngine::CNNLayerPtr);
    void SplitPrimitive(InferenceEngine::CNNLayerPtr);
    void SlicePrimitive(InferenceEngine::CNNLayerPtr);
    void PWLPrimitive(InferenceEngine::CNNLayerPtr);
    void CopyPrimitive(InferenceEngine::CNNLayerPtr);
    bool AreLayersSupported(InferenceEngine::ICNNNetwork& network, std::string& errMessage);
    LayerType LayerTypeFromStr(std::string const &str);
    /**
     * maps tpe of connection to input and output layers also stores gna_pointer for alloc request
     */
    class GNAMemoryLayer {
        InferenceEngine::CNNLayerPtr inputLayer;
        InferenceEngine::CNNLayerPtr outputLayer;
     public:
        GNAMemoryLayer(InferenceEngine::CNNLayerPtr inLayer, InferenceEngine::CNNLayerPtr outLayer) :
            inputLayer(inLayer), outputLayer(outLayer) {
        }

        InferenceEngine::CNNLayerPtr getInput() { return inputLayer; }
        InferenceEngine::CNNLayerPtr getOutput() { return outputLayer; }

        /**
         * pointer to gna memory request
         */
        void *gna_ptr = nullptr;
        /**
         * gna memory of this size is reserved
         */
        size_t  reserved_size = 0;
        /**
         * gna memory of this offset from gna_ptr
         */
        size_t  reserved_offset = 0;
    };

    class GNAConcatLayer {
        InferenceEngine::CNNLayerPtr concatLayer;

     public:
        explicit GNAConcatLayer(InferenceEngine::CNNLayerPtr layer) :
                                        concatLayer(layer)
                                        {}

        InferenceEngine::CNNLayerPtr getConcat() { return concatLayer; }
        /**
         * pointer to gna memory request
         */
        void *gna_ptr = nullptr;
        /**
         * gna memory of this size is reserved for concat
         */
        size_t reserved_size = 0;
        bool output_allocation_flag = false;
        /**
         * gna memory of this offset from gna_ptr
         */
        struct ConcatConnectedLayerInfo {
            ConcatConnectedLayerInfo(const std::string& n,
                                    size_t o) :
                                     name(n),
                                     offset(o) {}
            std::string name = "";
            size_t offset = 0;
        };

        std::vector<ConcatConnectedLayerInfo> concatInputLayers;
    };

    // Split, Slice
    class GNASplitLayer {
        InferenceEngine::CNNLayerPtr splitLayer;

     public:
        explicit GNASplitLayer(InferenceEngine::CNNLayerPtr layer) :
                                        splitLayer(layer),
                                        splitInputLayer()
                                        {}

        InferenceEngine::CNNLayerPtr getSplit() { return splitLayer; }
        /**
         * gna memory of this size is reserved for concat
         */
        size_t reserved_size = 0;
        bool output_allocation_flag = false;
        /**
         * gna memory of this offset from gna_ptr
         */
        struct SplitConnectedLayerInfo {
            SplitConnectedLayerInfo() {}
            SplitConnectedLayerInfo(std::string& n,
                                    size_t o,
                                    size_t p) :
                                     name(n),
                                     offset(o),
                                     pure_size(p) {}

            SplitConnectedLayerInfo& operator=
                    (SplitConnectedLayerInfo const& layerInfo) {
                this->name      = layerInfo.name;
                this->offset    = layerInfo.offset;
                this->pure_size = layerInfo.pure_size;
                return *this;
            }
            std::string name = "";
            size_t offset    = 0;
            size_t pure_size = 0;
        };
        SplitConnectedLayerInfo splitInputLayer;
        std::vector<SplitConnectedLayerInfo> splitOutputLayers;
    };

    class GNACropLayer {
        InferenceEngine::CNNLayerPtr cropLayer;

    public:
        explicit GNACropLayer(InferenceEngine::CNNLayerPtr layer) :
        cropLayer(layer)
        {}

        InferenceEngine::CNNLayerPtr getCrop() { return cropLayer; }
        /**
         * pointer to gna croped memory beginning
         */
        void *gna_ptr = nullptr;
    };
    using MemoryConnection = std::list<std::pair<std::string, GNAMemoryLayer>>;
    using ConcatConnection = std::map<std::string, GNAConcatLayer>;
    using SplitConnection  = std::map<std::string, GNASplitLayer>;
    using CropConnection  = std::map<std::string, GNACropLayer>;
    // layers with extra storage for connections and additional
    // non trivial processing
    MemoryConnection memory_connection;
    ConcatConnection concat_connection;
    SplitConnection  split_connection;
    CropConnection   crop_connection;
    void fillMemoryConnections(std::map<std::string,
                                 std::vector<InferenceEngine::CNNLayerPtr>> &memoryPairs);

    void fillConcatConnections(InferenceEngine::CNNLayerPtr layer);
    void fillSplitConnections(InferenceEngine::CNNLayerPtr layer);
    /**
     * maps layer name to dnn.component, in topological sort prev nodes will be initialized
     */
    using DnnComponentsForLayer = std::list<std::pair<std::string, intel_dnn_component_t>>;
    std::list<std::pair<std::string, intel_dnn_component_t>> dnnComponentsForLayer;

    /**
     * @brief returns corresponding dnn layer for topology layer
     * @param __layer
     * @return
     */
    intel_dnn_component_t * findDnnLayer(InferenceEngine::CNNLayerPtr __layer);

    using allocator_type = PolymorphAllocator<uint8_t>;
    using gna_memory_type = GNAMemory<allocator_type>;

    std::unique_ptr<GNADeviceHelper> gnadevice;
    /**
     * @brief size of RW segment without extra memory for parallel execution
     */
    uint32_t rwSegmentSize = 0;
    std::unique_ptr<gna_memory_type> gnamem;

    /**
     * Connects either memory output, or generic output to a layer
     * @param layer - layer pointer
     * @param ptr - pointer to pointer where to store  output layer information
     * @param sz - sizeof output blob
     * @param ptr_inputs - sizeof output blob
     */
    void connectOutput(InferenceEngine::CNNLayerPtr layer, void *ptr_outputs, void *ptr_inputs, size_t sz);
    /**
     * Connects certain input to this layer
     * @param layer - layer that we connect input to
     * @param pVoid - pointer that  holds current layer pointer in gna_mem request
     * @param num_data_bytes_in - size
     * @param offset - num bytes to advance in buffer
     * @param idx - index of input port that we are connecting
     * @return layer used as input
     */
    struct ConnectionDetails {
        InferenceEngine::CNNLayerPtr  input;
        bool needTransposeWeights = false;
        InferenceEngine::CNNLayerPtr permute;
        ConnectionDetails(InferenceEngine::CNNLayerPtr input,
                          bool bTranspose = false,
                          InferenceEngine::CNNLayerPtr permute = nullptr)
            : input(input)
            , needTransposeWeights(bTranspose)
            , permute(permute) {
        }
    };
    ConnectionDetails connectInput(InferenceEngine::CNNLayerPtr layer,
                      void *pVoid,
                      size_t num_data_bytes_in,
                      size_t offset = 0,
                      int idx = 0);

    void ImportFrames(void *ptr_dst,
                     const void *ptr_src,
                     InferenceEngine::Precision input_precision,
                     intel_dnn_orientation_t orientation,
                     uint32_t num_frames,
                     uint32_t num_group,
                     uint32_t num_vector_elements,
                     uint32_t num_vector_stride);

    void ExportScores(void *ptr_dst,
                     void *ptr_src,
                     intel_dnn_orientation_t orientation,
                     uint32_t num_frames,
                     uint32_t num_group,
                     uint32_t num_vector_elements,
                     uint32_t num_active_elements,
                     uint32_t num_vector_stride,
                     uint32_t num_bytes_per_element_input,
                     uint32_t num_bytes_per_element);

    friend void GNAPluginNS::ConvertToInt16(int16_t *ptr_dst,
                    const float *ptr_src,
                    const uint32_t num_rows,
                    const uint32_t num_columns,
                    const float scale_factor);
    friend void GNAPluginNS::ConvertToFloat(float *ptr_dst,
                    int32_t *ptr_src,
                    const uint32_t num_rows,
                    const uint32_t num_columns,
                    const float scale_factor);

    friend int16_t GNAPluginNS::ConvertFloatToInt16(float src);

    template <typename T, typename U>
    void copyInputData(T *dst,
                    const U *src,
                    uint32_t num_frames,
                    uint32_t num_group,
                    uint32_t num_vector_elements,
                    uint32_t num_vector_stride,
                    intel_dnn_orientation_t orientation);

    template <typename T, typename U>
    void copyInputDataWithSplit(T *const dst,
                    const U *src,
                    const GNASplitLayer& splitInfo,
                    size_t precision_size);
    /**
     * @brief GNA affine layers are always have activation atatched, while IR not
     * @param net - copied net ready for quantisation
     */
    void insertIdentityLayer(std::vector<InferenceEngine::CNNLayerPtr> &layers);

    /**
     * @brief GNA convolution layers have deinterleaved oriantations, while affine one doesn't
     * so between convolution and affine layers permute layers need to be inserted,
     * or removed if they are present in topology
     * @param layers
     */
    void applyOrientations(std::vector<InferenceEngine::CNNLayerPtr> &layers);


    /**
     * brief @search for specific patter in the graph (6 layers are replaced by single one)
     * @param layers
     */
    void substitutePRelu(std::vector<InferenceEngine::CNNLayerPtr> &layers);

    std::vector<InferenceEngine::CNNLayerPtr> getCandidatesForIdentityInsertion(const InferenceEngine::CNNLayerPtr layer);

    /**
     * diagonal layer insertion required in cases where activation followed by split layers, or any other
     * topology changing layers
     */
    void insertDiagonalLayer(std::vector<InferenceEngine::CNNLayerPtr> & layers);

    /**
     * @brief MaxPool can be reordered with activation, on GNA there is a strategy to have conv->maxpool->activation
     * it means maxpool receives 4 bytes, and produces 4 bytes
     */
    void reorderMaxPool(std::vector<InferenceEngine::CNNLayerPtr> & layers);

    /**
     * copy layer insertion required in cases where input layer does not have output memory
     */
    void insertCopyLayer(std::vector<InferenceEngine::CNNLayerPtr> & layers);

    intel_dnn_component_t * find_first_unused_input(InferenceEngine::CNNLayerPtr current);

    InferenceEngine::SizeVector inputDims;
    InferenceEngine::InputsDataMap inputsDataMap;

    InferenceEngine::SizeVector outputDims;
    InferenceEngine::OutputsDataMap outputsDataMap;
};
}  // namespace GNAPluginNS

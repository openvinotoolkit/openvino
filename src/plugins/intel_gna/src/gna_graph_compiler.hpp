// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/am_intel_dnn.hpp"
#include "backend/dnn_components.hpp"
#include "backend/gna_limitations.hpp"
#include "connection_details.hpp"
#include "descriptions/gna_desc.hpp"
#include "descriptions/gna_flags.hpp"
#include "gna_data_types.hpp"
#include "gna_device.hpp"
#include "layers/gna_concat_layer.hpp"
#include "layers/gna_crop_layer.hpp"
#include "layers/gna_memory_layer.hpp"
#include "layers/gna_split_layer.hpp"
#include "memory/gna_memory.hpp"

namespace ov {
namespace intel_gna {

class GNAGraphCompiler {
private:
    std::shared_ptr<backend::AMIntelDNN> dnn;
    std::shared_ptr<gna_memory_type> gnamem;
    std::shared_ptr<GnaInputs> inputs_ptr_;

    // layers with extra storage for connections and additional
    // non trivial processing

    SplitConnection split_connection;
    CropConnection crop_connection;
    const Config& gna_config;

    intel_dnn_component_t* find_first_unused_input(InferenceEngine::CNNLayerPtr current);

    static void printTensorDesc(const std::string& name, const InferenceEngine::TensorDesc& desc);
    static void printConvolutionLayer(const InferenceEngine::ConvolutionLayer& layer);
    static void printPoolingLayer(const InferenceEngine::PoolingLayer& layer);
    static void assertConvolutionLayoutProper(const InferenceEngine::DataPtr&);
    std::vector<uint8_t> static transposeMatrix(uint8_t* ptr_matrix,
                                                size_t element_size,
                                                uint32_t num_rows,
                                                uint32_t num_cols);
    std::vector<uint8_t> static copy_matrix(uint8_t* ptr_matrix,
                                            size_t element_size,
                                            uint32_t num_rows,
                                            uint32_t num_cols);

    bool ShouldUseOnlyConv2DGnaIface() const;

    std::shared_ptr<limitations::cnn2d::AbstractValidator> m_cnn2d_validator;

public:
    backend::DnnComponents dnnComponents;
    MemoryConnection memory_connection;
    ConcatConnection concat_connection;
    ConstConnections const_connections;

    GNAGraphCompiler(const Config& gna_config,
                     std::shared_ptr<backend::AMIntelDNN> dnn_ptr,
                     std::shared_ptr<GnaInputs> inputs_ptr,
                     std::shared_ptr<limitations::cnn2d::AbstractValidator> cnn2d_validator,
                     std::shared_ptr<gna_memory_type> gna_mem_ptr);
    void setGNAMemoryPtr(std::shared_ptr<gna_memory_type> gnaMemPtr);

    void fillMemoryConnections(std::unordered_map<std::string, std::vector<InferenceEngine::CNNLayerPtr>>& memoryPairs);

    void fillConcatConnections(InferenceEngine::CNNLayerPtr layer);
    void fillSplitConnections(InferenceEngine::CNNLayerPtr layer);

    void ValidateCnn2D(const std::string& name,
                       const uint32_t inHeight,
                       const uint32_t inWidth,
                       const uint32_t inChannels,
                       const uint32_t kH,
                       const uint32_t kW,
                       const uint32_t kN,
                       const uint32_t strideH,
                       const uint32_t strideW,
                       const uint32_t dilH,
                       const uint32_t dilW,
                       OvGnaType inPrecision) const;

    void ValidatePooling2D(const std::string& name,
                           const uint32_t windowH,
                           const uint32_t windowW,
                           const uint32_t strideH,
                           const uint32_t strideW) const;

    /**
     * Connects either memory output, or generic output to a layer
     * @param layer - layer pointer
     * @param ptr_outputs - pointer to pointer where to store  output layer information
     * @param ptr_inputs - sizeof output blob
     * @param sz - sizeof output blob
     * @param ptr_inputs - sizeof output blob
     */
    void connectOutput(InferenceEngine::CNNLayerPtr layer, void* ptr_outputs, size_t sz);
    /**
     * Connects certain input to this layer
     * @param layer - layer that we connect input to
     * @param pVoid - pointer that  holds current layer pointer in gna_mem request
     * @param num_data_bytes_in - size
     * @param offset - num bytes to advance in buffer
     * @param idx - index of input port that we are connecting
     * @param connectTo - connectTo is true is alternative to positive or equal to zero offset
     * in case when we would like to use zero offset and connect from  pointer set this to negative
     * @return layer used as input
     */
    ConnectionDetails connectInput(InferenceEngine::CNNLayerPtr layer,
                                   void* pVoid,
                                   size_t num_data_bytes_in,
                                   int32_t offset = 0,
                                   int idx = 0,
                                   bool connectTo = true);

    /**
     * Fill in the Affine layer weights
     * @param layer - affine layer pointer
     * @param ptrWeights - pointer to weights memory
     * @param offset - memory before offset value will be zeroed
     * @param isQuantized - information about layer quantization
     */
    void FillWeightOfAligningFilter(InferenceEngine::CNNLayerPtr layer,
                                    void* ptrWeights,
                                    size_t offset,
                                    bool isQuantized = false);

    void CreateLayerPrimitive(InferenceEngine::CNNLayerPtr);

    void AffinePrimitive(InferenceEngine::CNNLayerPtr, bool isDiag = false);
    void ConvolutionFilterPrimitive(InferenceEngine::CNNLayerPtr);
    void ConcatAlignFilterPrimitive(InferenceEngine::CNNLayerPtr);
    void DiagonalPrimitive(InferenceEngine::CNNLayerPtr);
    void ConstPrimitive(InferenceEngine::CNNLayerPtr);
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
    void FakeQuantizePrimitive(InferenceEngine::CNNLayerPtr);
    void CopyPrimitive(InferenceEngine::CNNLayerPtr);
    void GemmPrimitive(InferenceEngine::CNNLayerPtr);

    void finalizeConvolution1DPrimitive(InferenceEngine::CNNLayerPtr,
                                        uint32_t in_batch,
                                        uint32_t in_channels,
                                        uint32_t in_width,
                                        uint32_t out_batch,
                                        uint32_t out_channels,
                                        uint32_t out_width,
                                        uint32_t in_kernel_x,
                                        uint32_t in_kernel_y,
                                        bool transpose);
    void finalizeConvolution2DPrimitive(InferenceEngine::CNNLayerPtr,
                                        uint32_t in_batch,
                                        uint32_t in_channels,
                                        uint32_t in_height,
                                        uint32_t in_width,
                                        uint32_t out_batch,
                                        uint32_t out_channels,
                                        uint32_t out_height,
                                        uint32_t out_width);

    void Reset();
};

}  // namespace intel_gna
}  // namespace ov

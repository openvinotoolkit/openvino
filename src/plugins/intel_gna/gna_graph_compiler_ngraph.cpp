// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_graph_compiler_ngraph.hpp"

#include "ngraph/node.hpp"
#include "ngraph/op/matmul.hpp"
#include "ngraph/op/constant.hpp"

#include "backend/gna_limitations.hpp"

namespace ngraph {
    class GNAFullyConnected : public ngraph::op::MatMul {

};
}

namespace GNAPluginNS {
    constexpr uint32_t DefaultGnaPtrAllignment = 64;
    void* GetGna2TensorPtr(ov::Input<ov::Node>& input) {
 //   auto index = input.get_index();
 //  
 //   if (index == 1) {
 //       return &comp.op.affine.ptr_weights;
 //   }
 //  
        return nullptr; //TODO
    }

    std::shared_ptr<GNAPluginNS::gna_memory_type> gnamem;

    void connectInput(ov::Input<ov::Node>& input) {
        gnalog() << "connectInput " << input.get_node()->get_friendly_name();

        auto constNode = dynamic_cast<ngraph::op::Constant*>(input.get_source_output().get_node());
        // if source is const blob to be connected
        if (constNode != nullptr) {
            gnamem->readonly().push_ptr(input.get_node(),
                                        GetGna2TensorPtr(input),
                                        constNode->get_data_ptr(),
                                        constNode->get_byte_size(),
                                        DefaultGnaPtrAllignment);
        }
    }

    void connectOutput(ov::Output<ov::Node>& output) {
        gnalog() << "connectOutput " << output.get_node()->get_friendly_name();
    }

void Gna2ModelBuilder::Append(ngraph::GNAFullyConnected& layer) {
    auto& weightable = layer;
    //auto& weightable = dynamic_cast<WeightableLayer&>(*layer.get());

    constexpr bool quantized = true;
    //auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    // layer is already validater so no need for this
    // IE_ASSERT(!layer->insData.empty());
    // IE_ASSERT(!layer->outData.empty());

    auto inputs = layer.inputs();
    auto outputs = layer.outputs();

    const auto inputPrecision = inputs.front().get_element_type();
    uint32_t noOfInputsDivisor = inputPrecision == ov::element::i16 ? GNALimitations::noOfInputsDivisor
                                                                    : GNALimitations::noOfInputsLowPrecDivisor;
    //auto input_data = HasTo2DReshapeData(layer)
    //                      ? Get2DReshapedData(inputs, GNALimitations::GetMinBatchToFitInBuffer(inputs), 8)
    //                      : inputs;
    //auto in_dims = input_data->getDims();
    //auto batch_size = (in_dims.size() == 1) ? 1 : in_dims.front();
    //uint32_t num_rows_in = InferenceEngine::details::product(in_dims) / batch_size;
    //uint32_t num_columns_in = batch_size;
    //uint32_t num_rows_out = isDiag ? num_rows_in : GetDataDimSize(outputs, 1);
    //uint32_t num_padding = ALIGN(num_rows_in, noOfInputsDivisor) - num_rows_in;
    //uint32_t num_padding_out = isDiag ? num_padding : 0;

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    void* ptr_weights = nullptr;
    void* ptr_biases = nullptr;

   // // TODO: questionable why for biases that are not in IR we inventing precision
   // auto biasPrecisionSize = weightable._biases ? weightable._biases->getTensorDesc().getPrecision().size()
   //                                             : (gnaFlags->input_low_precision ? 1 : 4);

    // layer without biases might be connected to functional layer without activations
    auto prevLayer = layer.input(0).get_source_output();

  //  bool useBiasConnection = false;
  //  if (LayerInfo(prevLayer).has32BOutput()) {
  //      if (weightable._biases) {
  //          THROW_GNA_EXCEPTION << "Layer: " << layer->name
  //                              << ", cannot be connected to its parent: " << prevLayer->name
  //                              << " due to precision mismatch";
  //      }
  //      gnalog() << "Connection " << prevLayer->name << " to " << layer->name << " is using BIAS as input"
  //               << std::endl;
  //      useBiasConnection = true;
  //  }

    auto& currentComponent = dnnComponents.addComponent(layer.get_name(), layer.get_type_name());


    auto inputShape = layer.input(0).get_shape();
    auto weightShape = layer.input(1).get_shape();
    auto weightType = layer.input(1).get_element_type();
    auto biasType = layer.input(2).get_element_type();
    auto outputType = layer.output(0).get_element_type();

    constexpr auto notUsedSF = 1.0;

    dnn->InitAffineComponent(currentComponent,
                             inputShape[0], // num_rows_in + num_padding,
                             inputShape[1], //num_columns_in,
                             weightShape[0], //num_rows_out + num_padding_out,
                             inputPrecision.size(),
                             outputType.size(), //outputs->getPrecision().size(),
                             weightType.size(),
                             biasType.size(),
                             notUsedSF, //getScaleFactor(layer, QuantizedDataType::weights),
                             notUsedSF,  // getScaleFactor(layer, QuantizedDataType::output),
                             ptr_inputs,
                             ptr_outputs,
                             ptr_weights,
                             ptr_biases,
                             false);

    size_t num_data_bytes_out = inputShape[1] * weightShape[0] * outputType.size();

    size_t num_data_bytes_in = inputShape[1] * inputShape[0] * inputPrecision.size();

    connectInput(layer.input(0));
    connectInput(layer.input(1));
    connectInput(layer.input(2));

    connectOutput(layer.output(0));

  //  auto transpose = false;
  //  auto transposedRows = 0;
  //  auto transposedCols = 0;

  /// if (0 && connectionInfo.needTransposeWeights) {
  ///     // direct order is 0, 1, 2, 3, supported order is only 0,3,2,1 where dim 2 is usually equals to 1
  ///     auto permuteOrder = connectionInfo.permute->GetParamAsInts("order");
  ///     if (permuteOrder != vector<int>({0, 3, 2, 1})) {
  ///         IE_THROW() << "[GNA plugin] Unsupported permute order: was " << layer->GetParamAsString("order")
  ///                    << ", but only support 0, 3, 2, 1";
  ///     }
  ///
  ///     /**
  ///      * TODO: weights transpose happened after quantisation might result in poor quality for in 8 - move this to
  ///      * passes
  ///      */
  ///     if (weightable._weights->getTensorDesc().getPrecision() == Precision::I8) {
  ///         IE_THROW() << "[GNA plugin] Unsupported permute operation for 8 bit weights for layer: " << layer->name;
  ///     }
  ///
  ///     // this affine connected to convolution via pool or activation
  ///     gnalog() << "Transposing weights for layer: " << layer->name << "\n";
  ///
  ///     transpose = !isDiag;
  ///     transposedRows = connectionInfo.permute->input()->getDims()[3];
  ///     transposedCols = connectionInfo.permute->input()->getDims()[1];
  /// }

  //   auto wpSize = weightable.precision.size();
  //  const auto weightsBuffer = weightable._weights->cbuffer().as<const uint8_t*>();

  // if (num_padding == 0) {
  //     if (!transpose) {
   //         gnamem->readonly().push_ptr(layer,
   //                                     ptr_weights,
   //                                     weightable._weights->cbuffer().as<const void*>(),
   //                                     weightable._weights->byteSize(),
   //                                     64);
    //    } else {
    //        gnamem->readonly().push_initializer(
    //            layer,
    //            ptr_weights,
    //            weightable._weights->byteSize(),
    //            [isDiag,
    //             num_rows_in,
    //             num_rows_out,
    //             num_padding,
    //             transposedRows,
    //             transposedCols,
    //             weightsBuffer,
    //             wpSize](void* data, size_t size) {
    //                for (uint32_t k = 0; k < (isDiag ? 1 : num_rows_out); k++) {
    //                    auto rowOffset = k * transposedRows * transposedCols * wpSize;
    //                    auto cbuffer = weightsBuffer + rowOffset;
    //                    auto u8Data = reinterpret_cast<uint8_t*>(data) + rowOffset;
    //                    for (int j = 0; j < transposedCols; j++) {
    //                        for (int i = 0; i < transposedRows; i++) {
    //                            auto offsetWrite = (transposedRows * j + i) * wpSize;
    //                            auto offsetRead = (i * transposedCols + j) * wpSize;
    //                            if (size < rowOffset + offsetWrite) {
    //                                // zero out dest if error detected
    //                                memset(data, 0, size);
    //                                THROW_GNA_EXCEPTION << "Size error";
    //                            }
    //                            ie_memcpy(u8Data + offsetWrite,
    //                                      size - rowOffset - offsetWrite,
    //                                      cbuffer + offsetRead,
    //                                      wpSize);
    //                        }
    //                    }
    //                }
    //            },
    //            64);
    //    }
    //} else {
    //    if (transpose) {
    //        THROW_GNA_EXCEPTION << "transposed weights with non zero padding not yet supported";
    //    }
    //    auto elementsIn = (num_rows_in + num_padding) * num_columns_in;
    //    auto paddedWeights = isDiag ? elementsIn : elementsIn * num_rows_out;
    //    auto paddedWeightsSize = paddedWeights * weightable.precision.size();
    //
    //    gnamem->readonly().push_initializer(
    //        layer,
    //        ptr_weights,
    //        paddedWeightsSize,
    //        [isDiag, num_rows_in, num_rows_out, num_padding, weightsBuffer, wpSize](void* data, size_t size) {
    //            for (uint32_t i = 0; i < (isDiag ? 1 : num_rows_out); i++) {
    //                ie_memcpy(data, size, weightsBuffer + num_rows_in * i * wpSize, num_rows_in * wpSize);
    //                data = reinterpret_cast<uint8_t*>(data) + (num_rows_in + num_padding) * wpSize;
    //            }
    //        },
    //        64);
    //}

  //  if (weightable._biases) {
  //      gnamem->readonly().push_ptr(layer,
  //                                  ptr_biases,
  //                                  weightable._biases->cbuffer().as<const void*>(),
  //                                  weightable._biases->byteSize(),
  //                                  64);
  //  } else {
  //      // in that case input from previous layer goes into biases, so we have to initialize input pointer by zero
  //      if (useBiasConnection) {
  //          gnamem->readonly().push_value(layer, ptr_inputs, 0.0f, num_rows_in + num_padding, 64);
  //      } else {
  //          gnamem->readonly().push_value(layer, ptr_biases, 0.0f, num_rows_out + num_padding_out, 64);
  //      }
  //  }
}

}
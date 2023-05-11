// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/custom_shape_inference/reshape.hpp"
#include "utils/custom_shape_inference/gather.hpp"
#include "utils/custom_shape_inference/transpose.hpp"
#include "utils/custom_shape_inference/color_convert.hpp"
#include "utils/custom_shape_inference/eltwise.hpp"
#include "utils/custom_shape_inference/adaptive_pooling.hpp"
#include "utils/custom_shape_inference/fullyconnected.hpp"
#include "utils/custom_shape_inference/matmul.hpp"
#include "utils/custom_shape_inference/ngram.hpp"
#include "utils/custom_shape_inference/one_hot.hpp"
#include "utils/custom_shape_inference/priorbox.hpp"
#include "utils/custom_shape_inference/priorbox_clustered.hpp"
#include "utils/custom_shape_inference/shapeof.hpp"
#include "utils/custom_shape_inference/strided_slice.hpp"
// #include "utils/custom_shape_inference/deconv.hpp"
// #include "utils/custom_shape_inference/subgraph.hpp"
#include "ie_ngraph_utils.hpp"
#include "custom_shape_infer.hpp"
#include <gtest/gtest.h>
namespace ov {
namespace intel_cpu {
namespace unit_test {
#define INTEL_CPU_CUSTOM_SHAPE_INFER(__prim, __type) \
    registerNodeIfRequired(intel_cpu, __prim, __type, __prim)

class EltwiseShapeInferTestFactory : public node::EltwiseShapeInferFactory {
public:
    EltwiseShapeInferTestFactory(std::shared_ptr<ov::Node> op) : EltwiseShapeInferFactory() {}
};

class ShapeOfShapeInferTestFactory : public node::ShapeOfShapeInferFactory {
public:
    ShapeOfShapeInferTestFactory(std::shared_ptr<ov::Node> op) : ShapeOfShapeInferFactory() {}
};

CustomShapeInferFF::CustomShapeInferFF():Factory("CpuCustomShapeInferTestFactory") {
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Generic, Type::Generic);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(CumSum, Type::CumSum);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Convolution, Type::Convolution);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(BinaryConvolution, Type::BinaryConvolution);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(SpaceToBatch, Type::SpaceToBatch);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Lrn, Type::Lrn);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(BatchToSpace, Type::BatchToSpace);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(DepthToSpace, Type::DepthToSpace);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(SpaceToDepth, Type::SpaceToDepth);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(If, Type::If);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Broadcast, Type::Broadcast);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ExperimentalDetectronTopKROIs, Type::ExperimentalDetectronTopKROIs);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Reorder, Type::Reorder);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(MatrixNms, Type::MatrixNms);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::AdaptivePoolingShapeInferFactory, Type::AdaptivePooling);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Pooling, Type::Pooling);
    INTEL_CPU_CUSTOM_SHAPE_INFER(EltwiseShapeInferTestFactory, Type::Eltwise);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(SoftMax, Type::Softmax);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(EmbeddingBagPackedSum, Type::EmbeddingBagPackedSum);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Input, Type::Input);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Input, Type::Output);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(MemoryInput, Type::MemoryInput);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(MemoryOutput, Type::MemoryOutput);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Tile, Type::Tile);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(GatherTree, Type::GatherTree);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::FCShapeInferFactory, Type::FullyConnected);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(CTCGreedyDecoder, Type::CTCGreedyDecoder);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::TransposeShapeInferFactory, Type::Transpose);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ReorgYolo, Type::ReorgYolo);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(EmbeddingSegmentsSum, Type::EmbeddingSegmentsSum);
    /* INTEL_CPU_CUSTOM_SHAPE_INFER(ShapeOfShapeInferTestFactory, Type::ShapeOf); */
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ExperimentalDetectronGenerateProposalsSingleImage, Type::ExperimentalDetectronGenerateProposalsSingleImage);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(GenerateProposals, Type::GenerateProposals);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ReverseSequence, Type::ReverseSequence);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ExperimentalDetectronPriorGridGenerator, Type::ExperimentalDetectronPriorGridGenerator);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(GatherND, Type::GatherND);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(LogSoftmax, Type::LogSoftmax);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(PSROIPooling, Type::PSROIPooling);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(RNN, Type::RNNCell);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(RNN, Type::RNNSeq);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(CTCLoss, Type::CTCLoss);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Split, Type::Split);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(DetectionOutput, Type::DetectionOutput);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(GatherElements, Type::GatherElements);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(CTCGreedyDecoderSeqLen, Type::CTCGreedyDecoderSeqLen);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Bucketize, Type::Bucketize);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ExperimentalDetectronROIFeatureExtractor, Type::ExperimentalDetectronROIFeatureExtractor);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Math, Type::Math);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(MultiClassNms, Type::MulticlassNms);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Convert, Type::Convert);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::ColorConvertShapeInferFactory, Type::ColorConvert);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(EmbeddingBagOffsetSum, Type::EmbeddingBagOffsetsSum);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Roll, Type::Roll);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Pad, Type::Pad);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::ReshapeShapeInferFactory, Type::Reshape);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(MVN, Type::MVN);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(node::MMShapeInferFactory, Type::MatMul);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ScatterUpdate, Type::ScatterUpdate);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ScatterUpdate, Type::ScatterElementsUpdate);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ScatterUpdate, Type::ScatterNDUpdate);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ShuffleChannels, Type::ShuffleChannels);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(TensorIterator, Type::TensorIterator);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Concat, Type::Concatenation);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::OneHotShapeInferFactory, Type::OneHot);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ExperimentalDetectronDetectionOutput, Type::ExperimentalDetectronDetectionOutput);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(node::DeconvolutionShapeInferFactory, Type::Deconvolution);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(DeformableConvolution, Type::DeformableConvolution);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Range, Type::Range);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(node::StridedSliceShapeInferFactory, Type::StridedSlice);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(GRN, Type::GRN);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(NonZero, Type::NonZero);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(NormalizeL2, Type::NormalizeL2);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::PriorBoxShapeInferFactory, Type::PriorBox);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::PriorBoxClusteredShapeInferFactory, Type::PriorBoxClustered);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Eye, Type::Eye);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Unique, Type::Unique);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::NgramShapeInferFactory, Type::Ngram);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Interpolate, Type::Interpolate);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Reduce, Type::Reduce);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::GatherShapeInferFactory, Type::Gather);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(NonMaxSuppression, Type::NonMaxSuppression);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ROIPooling, Type::ROIPooling);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ROIAlign, Type::ROIAlign);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(TopK, Type::TopK);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Proposal, Type::Proposal);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(RegionYolo, Type::RegionYolo);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(DFT, Type::DFT);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(RDFT, Type::RDFT);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(ExtractImagePatches, Type::ExtractImagePatches);
#if defined(OPENVINO_ARCH_X86_64)
    // INTEL_CPU_CUSTOM_SHAPE_INFER(FakeQuantize, Type::FakeQuantize);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(GridSample, Type::GridSample);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(Interaction, Type::Interaction);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(MHA, Type::MHA);
    // INTEL_CPU_CUSTOM_SHAPE_INFER(node::SnippetShapeInferFactory, Type::Subgraph);
#endif
#undef INTEL_CPU_CUSTOM_SHAPE_INFER
}


ShapeInferFactory* CustomShapeInferFF::create(const std::shared_ptr<ov::Node>& op) {
        ShapeInferFactory* newShapeInferFactory = nullptr;
        std::unique_ptr<ShapeInferFactory> ol(createNodeIfRegistered(intel_cpu, TypeFromName(op->get_type_name()), op));
        if (ol != nullptr) {
             newShapeInferFactory = ol.release();
        }
        return newShapeInferFactory;
}

static void compare_result(std::vector<StaticShape> ref, std::vector<VectorDims> cus) {
    std::cout << "=====custom_shape_infer compile result======" << std::endl;
    std::cout << "===========" << "ref.size()" << ref.size() << std::endl;
    std::cout << "===========" << "cus.size()" << cus.size() << std::endl;
    ASSERT_TRUE(ref.size() == cus.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_TRUE(ref[i].size() == cus[i].size());
        for (size_t y = 0; y < ref[i].size(); y++) {
            ASSERT_TRUE(ref[i][y].get_length() == cus[i][y]);
        }
    }
}

void custom_shape_inference(ov::Node* op,
                     const std::vector<StaticShape>& input_shapes,
                     std::vector<StaticShape>& output_shapes,
                     const std::map<size_t, HostTensorPtr>& constant_data) {
    static std::shared_ptr<CustomShapeInferFF> cusFactory = std::make_shared<CustomShapeInferFF>();
    std::cout << "=====custom_shape_infer test======" << "op" << op->get_type_name() << std::endl;
    if (auto shapeInferFactory = cusFactory->create(op->shared_from_this())) {
        if (TypeFromName(op->get_type_name()) == Type::AdaptivePooling && op->get_output_size() == 0) {
            return;
        }
        std::cout << "=====custom_shape_infer test factory======" << "op" << op->get_type_name() << std::endl;
        auto cusShapeInfer =  shapeInferFactory->makeShapeInfer();
        std::vector<std::reference_wrapper<const VectorDims>> cusInputShapes;
        std::vector<VectorDims> tmpInputShapes;
        cusInputShapes.reserve(input_shapes.size());
        tmpInputShapes.reserve(input_shapes.size());
        for (size_t port = 0; port < input_shapes.size(); ++port) {
            VectorDims dims;
            for (size_t i =0; i < input_shapes[port].size(); ++i) {
                dims.emplace_back(input_shapes[port][i].get_length());
            }
            tmpInputShapes.emplace_back(dims);
            cusInputShapes.emplace_back(std::ref(tmpInputShapes[port]));
        }

        std::unordered_map<size_t, MemoryPtr> cusInputValues;
        auto input_value_port_mask = cusShapeInfer->get_port_mask();
        dnnl::engine eng;
        if (input_value_port_mask) {
            for (size_t port = 0; port < input_shapes.size(); ++port) {
                if (input_value_port_mask & (1 << port)) {
                    const auto tensorIter = constant_data.find(port);
                    const void* data = nullptr;
                    ov::element::Type elementType;
                    if (tensorIter != constant_data.end()) {
                        const auto tensor = tensorIter->second;
                        data = tensor->get_data_ptr();
                        elementType = tensor->get_element_type();
                    } else {
                        const auto input_op = op->input_value(port).get_node_shared_ptr();
                        const auto const_op = ov::as_type_ptr<const ov::op::v0::Constant>(input_op);
                        ASSERT_TRUE(const_op != nullptr);
                        data = const_op->get_data_ptr();
                        elementType = const_op->get_element_type();
                    }
                    CpuBlockedMemoryDesc desc(
                            InferenceEngine::details::convertPrecision(elementType),
                            ov::intel_cpu::Shape(tmpInputShapes[port]));
                    MemoryPtr memoryPtr = std::make_shared<Memory>(eng);
                    memoryPtr->Create(desc, data, true);
                    cusInputValues[port] = memoryPtr;
                }
            }
        }
        auto result = cusShapeInfer->infer(cusInputShapes, cusInputValues);
        compare_result(output_shapes, result.dims);
    }
}
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov

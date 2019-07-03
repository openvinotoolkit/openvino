// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impl_register.hpp"
#include "ie_built_in_holder.hpp"
#include "ie_conv_shape_infer.hpp"
#include "ie_deconv_shape_infer.hpp"
#include "ie_pool_shape_infer.hpp"
#include "ie_crop_shape_infer.hpp"
#include "ie_tile_shape_infer.hpp"
#include "ie_split_shape_infer.hpp"
#include "ie_equal_shape_infer.hpp"
#include "ie_concat_shape_infer.hpp"
#include "ie_eltwise_shape_infer.hpp"
#include "ie_permute_shape_infer.hpp"
#include "ie_reshape_shape_infer.hpp"
#include "ie_flatten_shape_infer.hpp"
#include "ie_proposal_shape_infer.hpp"
#include "ie_priorbox_shape_infer.hpp"
#include "ie_upsampling_shape_infer.hpp"
#include "ie_reorg_yolo_shape_infer.hpp"
#include "ie_region_yolo_shape_infer.hpp"
#include "ie_simpler_nms_shape_infer.hpp"
#include "ie_roi_pooling_shape_infer.hpp"
#include "ie_psroi_pooling_shape_infer.hpp"
#include "ie_detection_output_shape_infer.hpp"
#include "ie_priorbox_clustered_shape_infer.hpp"
#include "ie_ctc_greedy_decoder_shape_infer.hpp"
#include "ie_inner_product_shape_infer.hpp"
#include "ie_resample_shape_infer.hpp"
#include "ie_interp_shape_infer.hpp"
#include "ie_argmax_shape_infer.hpp"
#include "ie_gemm_shape_infer.hpp"
#include "ie_pad_shape_infer.hpp"
#include "ie_gather_shape_infer.hpp"
#include "ie_strided_slice_shape_infer.hpp"
#include "ie_shuffle_channels_shape_infer.hpp"
#include "ie_depth_to_space_shape_infer.hpp"
#include "ie_space_to_depth_shape_infer.hpp"
#include "ie_reverse_sequence_shape_infer.hpp"
#include "ie_shape_shape_infer.hpp"
#include "ie_squeeze_shape_infer.hpp"
#include "ie_unsqueeze_shape_infer.hpp"
#include "ie_range_shape_infer.hpp"
#include "ie_fill_shape_infer.hpp"
#include "ie_expand_shape_infer.hpp"
#include "ie_rnn_shape_infer.hpp"
#include "ie_tensor_iterator_shape_infer.hpp"
#include "ie_rnn_cell_shape_infer.hpp"
#include "ie_quantize_shape_infer.hpp"
#include "ie_bin_conv_shape_infer.hpp"
#include <algorithm>
#include <memory>
#include <string>

namespace InferenceEngine {
namespace ShapeInfer {

BuiltInShapeInferHolder::ImplsHolder::Ptr BuiltInShapeInferHolder::GetImplsHolder() {
    static ImplsHolder::Ptr localHolder;
    if (localHolder == nullptr) {
        localHolder = std::make_shared<ImplsHolder>();
    }
    return localHolder;
}

void BuiltInShapeInferHolder::AddImpl(const std::string& name, const IShapeInferImpl::Ptr& impl) {
    GetImplsHolder()->list[name] = impl;
}

StatusCode BuiltInShapeInferHolder::getShapeInferTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept {
    auto& factories = GetImplsHolder()->list;
    types = new char* [factories.size()];
    size = 0;
    for (auto it = factories.begin(); it != factories.end(); it++, size++) {
        types[size] = new char[it->first.size() + 1];
        std::copy(it->first.begin(), it->first.end(), types[size]);
        types[size][it->first.size()] = '\0';
    }
    return OK;
}

StatusCode
BuiltInShapeInferHolder::getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept {
    auto& impls = BuiltInShapeInferHolder::GetImplsHolder()->list;
    if (impls.find(type) != impls.end()) {
        impl = impls[type];
        return OK;
    }
    impl.reset();
    return NOT_FOUND;
}

void BuiltInShapeInferHolder::SetLogCallback(InferenceEngine::IErrorListener& listener) noexcept {}

// Register without implementation just to protect from adding custom implementation for them
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Input);
REG_SHAPE_INFER_FOR_TYPE(DoNothingShapeProp, Output);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Memory);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Const);

// Outputs = Inputs
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Activation);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, ReLU);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, ReLU6);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, ELU);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, TanH);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Logistic);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Sigmoid);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, PReLU);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, SoftMax);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, LRN);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Norm);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Normalize);
// FIXME: Really Copy??? New MO doesn't generate this layer
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Copy);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Power);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, PowerFile);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Clamp);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, ScaleShift);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, BatchNormalization);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, GRN);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, MVN);


REG_SHAPE_INFER_FOR_TYPE(ConvShapeProp, Convolution);
REG_SHAPE_INFER_FOR_TYPE(DeconvShapeProp, Deconvolution);
REG_SHAPE_INFER_FOR_TYPE(PoolingShapeProp, Pooling);
REG_SHAPE_INFER_FOR_TYPE(InnerProductShapeProp, InnerProduct);
REG_SHAPE_INFER_FOR_TYPE(InnerProductShapeProp, FullyConnected);
REG_SHAPE_INFER_FOR_TYPE(SplitShapeProp, Split);
REG_SHAPE_INFER_FOR_TYPE(SplitShapeProp, Slice);
REG_SHAPE_INFER_FOR_TYPE(PermuteShapeProp, Permute);
REG_SHAPE_INFER_FOR_TYPE(FlattenShapeProp, Flatten);
REG_SHAPE_INFER_FOR_TYPE(ReshapeShapeProp, Reshape);
REG_SHAPE_INFER_FOR_TYPE(DetectionOutputShapeProp, DetectionOutput);
REG_SHAPE_INFER_FOR_TYPE(PriorBoxClusteredShapeProp, PriorBoxClustered);
REG_SHAPE_INFER_FOR_TYPE(PriorBoxShapeProp, PriorBox);
REG_SHAPE_INFER_FOR_TYPE(RoiPoolingShapeProp, ROIPooling);
REG_SHAPE_INFER_FOR_TYPE(PSRoiPoolingShapeProp, PSROIPooling);
REG_SHAPE_INFER_FOR_TYPE(UpsamplingShapeProp, Upsampling);
REG_SHAPE_INFER_FOR_TYPE(ResampleShapeProp, Resample);
REG_SHAPE_INFER_FOR_TYPE(InterpShapeProp, Interp);
REG_SHAPE_INFER_FOR_TYPE(SimplerNMSShapeProp, SimplerNMS);
REG_SHAPE_INFER_FOR_TYPE(TileShapeProp, Tile);
REG_SHAPE_INFER_FOR_TYPE(CropShapeProp, Crop);
REG_SHAPE_INFER_FOR_TYPE(ConcatShapeProp, Concat);
REG_SHAPE_INFER_FOR_TYPE(EltWiseShapeProp, Eltwise);
REG_SHAPE_INFER_FOR_TYPE(EltWiseShapeProp, Mul);
REG_SHAPE_INFER_FOR_TYPE(EltWiseShapeProp, Add);
REG_SHAPE_INFER_FOR_TYPE(EltWiseShapeProp, Div);
REG_SHAPE_INFER_FOR_TYPE(CTCGreedyDecoderShapeProp, CTCGreedyDecoder);
REG_SHAPE_INFER_FOR_TYPE(ProposalShapeProp, Proposal);
REG_SHAPE_INFER_FOR_TYPE(ReorgYoloShapeProp, ReorgYolo);
REG_SHAPE_INFER_FOR_TYPE(RegionYoloShapeProp, RegionYolo);
REG_SHAPE_INFER_FOR_TYPE(RNNShapeProp, RNNSequence);
REG_SHAPE_INFER_FOR_TYPE(RNNShapeProp, GRUSequence);
REG_SHAPE_INFER_FOR_TYPE(RNNShapeProp, LSTMSequence);
REG_SHAPE_INFER_FOR_TYPE(RNNCellShapeProp, RNNCell);
REG_SHAPE_INFER_FOR_TYPE(GRUCellShapeProp, GRUCell);
REG_SHAPE_INFER_FOR_TYPE(LSTMCellShapeProp, LSTMCell);
REG_SHAPE_INFER_FOR_TYPE(TensorIteratorShapeProp, TensorIterator);
REG_SHAPE_INFER_FOR_TYPE(ArgMaxShapeProp, ArgMax);
REG_SHAPE_INFER_FOR_TYPE(GemmShapeProp, Gemm);
REG_SHAPE_INFER_FOR_TYPE(PadShapeProp, Pad);
REG_SHAPE_INFER_FOR_TYPE(GatherShapeProp, Gather);
REG_SHAPE_INFER_FOR_TYPE(StridedSliceShapeProp, StridedSlice);
REG_SHAPE_INFER_FOR_TYPE(ShuffleChannelsShapeProp, ShuffleChannels);
REG_SHAPE_INFER_FOR_TYPE(DepthToSpaceShapeProp, DepthToSpace);
REG_SHAPE_INFER_FOR_TYPE(SpaceToDepthShapeProp, SpaceToDepth);
REG_SHAPE_INFER_FOR_TYPE(ReverseSequenceShapeProp, ReverseSequence);
REG_SHAPE_INFER_FOR_TYPE(SqueezeShapeProp, Squeeze);
REG_SHAPE_INFER_FOR_TYPE(UnsqueezeShapeProp, Unsqueeze);
REG_SHAPE_INFER_FOR_TYPE(RangeShapeProp, Range);
REG_SHAPE_INFER_FOR_TYPE(FillShapeProp, Fill);
REG_SHAPE_INFER_FOR_TYPE(ExpandShapeProp, Expand);
REG_SHAPE_INFER_FOR_TYPE(ShapeShapeProp, Shape);
REG_SHAPE_INFER_FOR_TYPE(QuantizeShapeProp, Quantize);
REG_SHAPE_INFER_FOR_TYPE(BinConvShapeProp, BinaryConvolution);

}  // namespace ShapeInfer
}  // namespace InferenceEngine

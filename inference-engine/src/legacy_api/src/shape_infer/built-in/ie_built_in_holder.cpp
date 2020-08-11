// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <string>

#include "legacy/shape_infer/built-in/ie_built_in_holder.hpp"
#include "ie_argmax_shape_infer.hpp"
#include "ie_bin_conv_shape_infer.hpp"
#include "ie_broadcast_shape_infer.hpp"
#include "ie_concat_shape_infer.hpp"
#include "ie_conv_shape_infer.hpp"
#include "ie_crop_shape_infer.hpp"
#include "ie_ctc_greedy_decoder_shape_infer.hpp"
#include "ie_deconv_shape_infer.hpp"
#include "ie_deformable_conv_shape_infer.hpp"
#include "ie_depth_to_space_shape_infer.hpp"
#include "ie_detection_output_shape_infer.hpp"
#include "ie_eltwise_shape_infer.hpp"
#include "ie_equal_shape_infer.hpp"
#include "ie_erf_shape_infer.hpp"
#include "ie_fill_shape_infer.hpp"
#include "ie_flatten_shape_infer.hpp"
#include "ie_gather_shape_infer.hpp"
#include "ie_gather_tree_shape_infer.hpp"
#include "ie_gemm_shape_infer.hpp"
#include "ie_inner_product_shape_infer.hpp"
#include "ie_interp_shape_infer.hpp"
#include "ie_non_max_suppression_shape_infer.hpp"
#include "ie_one_hot_shape_infer.hpp"
#include "ie_pad_shape_infer.hpp"
#include "ie_permute_shape_infer.hpp"
#include "ie_pool_shape_infer.hpp"
#include "ie_priorbox_clustered_shape_infer.hpp"
#include "ie_priorbox_shape_infer.hpp"
#include "ie_proposal_shape_infer.hpp"
#include "ie_psroi_pooling_shape_infer.hpp"
#include "ie_quantize_shape_infer.hpp"
#include "ie_range_shape_infer.hpp"
#include "ie_reduce_shape_infer.hpp"
#include "ie_region_yolo_shape_infer.hpp"
#include "ie_reorg_yolo_shape_infer.hpp"
#include "ie_resample_shape_infer.hpp"
#include "ie_reshape_shape_infer.hpp"
#include "ie_reverse_sequence_shape_infer.hpp"
#include "ie_rnn_cell_shape_infer.hpp"
#include "ie_rnn_shape_infer.hpp"
#include "ie_roi_pooling_shape_infer.hpp"
#include "ie_scatter_shape_infer.hpp"
#include "ie_select_shape_infer.hpp"
#include "ie_shape_shape_infer.hpp"
#include "ie_shuffle_channels_shape_infer.hpp"
#include "ie_simpler_nms_shape_infer.hpp"
#include "ie_space_to_depth_shape_infer.hpp"
#include "ie_sparse_fill_empty_rows_shape_infer.hpp"
#include "ie_sparse_segment_reduce_shape_infer.hpp"
#include "ie_split_shape_infer.hpp"
#include "ie_sparse_to_dense_shape_infer.hpp"
#include "ie_bucketize_shape_infer.hpp"
#include "ie_squeeze_shape_infer.hpp"
#include "ie_sparse_weighted_reduce_shape_infer.hpp"
#include "ie_strided_slice_shape_infer.hpp"
#include "ie_tensor_iterator_shape_infer.hpp"
#include "ie_tile_shape_infer.hpp"
#include "ie_topk_shape_infer.hpp"
#include "ie_unique_shape_infer.hpp"
#include "ie_unsqueeze_shape_infer.hpp"
#include "ie_upsampling_shape_infer.hpp"
#include "impl_register.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

BuiltInShapeInferHolder::ImplsHolder::Ptr BuiltInShapeInferHolder::GetImplsHolder() {
    static ImplsHolder::Ptr localHolder;
    if (localHolder == nullptr) {
        localHolder = std::make_shared<ImplsHolder>();
    }
    return localHolder;
}

IE_SUPPRESS_DEPRECATED_START

void BuiltInShapeInferHolder::AddImpl(const std::string& name, const IShapeInferImpl::Ptr& impl) {
    GetImplsHolder()->list[name] = impl;
}

StatusCode BuiltInShapeInferHolder::getShapeInferTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept {
    auto& factories = GetImplsHolder()->list;
    types = new char*[factories.size()];
    size = 0;
    for (auto it = factories.begin(); it != factories.end(); it++, size++) {
        types[size] = new char[it->first.size() + 1];
        std::copy(it->first.begin(), it->first.end(), types[size]);
        types[size][it->first.size()] = '\0';
    }
    return OK;
}

StatusCode BuiltInShapeInferHolder::getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type,
                                                      ResponseDesc* resp) noexcept {
    auto& impls = BuiltInShapeInferHolder::GetImplsHolder()->list;
    if (impls.find(type) != impls.end()) {
        impl = impls[type];
        return OK;
    }
    impl.reset();
    return NOT_FOUND;
}

IE_SUPPRESS_DEPRECATED_END

// Register without implementation just to protect from adding custom implementation for them
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Input);
REG_SHAPE_INFER_FOR_TYPE(DoNothingShapeProp, Output);
REG_SHAPE_INFER_FOR_TYPE(MemoryShapeProp, Memory);
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
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, LogSoftMax);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, LRN);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Norm);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Normalize);
REG_SHAPE_INFER_FOR_TYPE(EqualShapeProp, Convert);
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
REG_SHAPE_INFER_FOR_TYPE(DeformableConvShapeProp, DeformableConvolution);
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
REG_SHAPE_INFER_FOR_TYPE(SparseFillEmptyRowsShapeProp, SparseFillEmptyRows);
REG_SHAPE_INFER_FOR_TYPE(SparseSegmentReduceShapeProp, SparseSegmentMean);
REG_SHAPE_INFER_FOR_TYPE(SparseSegmentReduceShapeProp, SparseSegmentSqrtN);
REG_SHAPE_INFER_FOR_TYPE(SparseSegmentReduceShapeProp, SparseSegmentSum);
REG_SHAPE_INFER_FOR_TYPE(ExperimentalSparseWeightedReduceShapeProp, ExperimentalSparseWeightedSum);
REG_SHAPE_INFER_FOR_TYPE(SparseToDenseShapeProp, SparseToDense);
REG_SHAPE_INFER_FOR_TYPE(BucketizeShapeProp, Bucketize);
REG_SHAPE_INFER_FOR_TYPE(ReverseSequenceShapeProp, ReverseSequence);
REG_SHAPE_INFER_FOR_TYPE(SelectShapeProp, Select);
REG_SHAPE_INFER_FOR_TYPE(SqueezeShapeProp, Squeeze);
REG_SHAPE_INFER_FOR_TYPE(UnsqueezeShapeProp, Unsqueeze);
REG_SHAPE_INFER_FOR_TYPE(RangeShapeProp, Range);
REG_SHAPE_INFER_FOR_TYPE(FillShapeProp, Fill);
REG_SHAPE_INFER_FOR_TYPE(BroadcastShapeProp, Broadcast);
REG_SHAPE_INFER_FOR_TYPE(ShapeShapeProp, Shape);
REG_SHAPE_INFER_FOR_TYPE(OneHotShapeProp, OneHot);
REG_SHAPE_INFER_FOR_TYPE(QuantizeShapeProp, FakeQuantize);
REG_SHAPE_INFER_FOR_TYPE(BinConvShapeProp, BinaryConvolution);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Abs);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Acos);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Acosh);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Asin);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Asinh);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Atan);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Atanh);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Ceil);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Cos);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Cosh);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Erf);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Floor);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, HardSigmoid);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Log);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Exp);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Neg);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Reciprocal);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Selu);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Sign);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Sin);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Sinh);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Softplus);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Softsign);
REG_SHAPE_INFER_FOR_TYPE(MathShapeProp, Tan);
REG_SHAPE_INFER_FOR_TYPE(ReduceShapeProp, ReduceAnd);
REG_SHAPE_INFER_FOR_TYPE(ReduceShapeProp, ReduceL1);
REG_SHAPE_INFER_FOR_TYPE(ReduceShapeProp, ReduceL2);
REG_SHAPE_INFER_FOR_TYPE(ReduceShapeProp, ReduceLogSum);
REG_SHAPE_INFER_FOR_TYPE(ReduceShapeProp, ReduceLogSumExp);
REG_SHAPE_INFER_FOR_TYPE(ReduceShapeProp, ReduceMax);
REG_SHAPE_INFER_FOR_TYPE(ReduceShapeProp, ReduceMean);
REG_SHAPE_INFER_FOR_TYPE(ReduceShapeProp, ReduceMin);
REG_SHAPE_INFER_FOR_TYPE(ReduceShapeProp, ReduceOr);
REG_SHAPE_INFER_FOR_TYPE(ReduceShapeProp, ReduceProd);
REG_SHAPE_INFER_FOR_TYPE(ReduceShapeProp, ReduceSum);
REG_SHAPE_INFER_FOR_TYPE(ReduceShapeProp, ReduceSumSquare);
REG_SHAPE_INFER_FOR_TYPE(GatherTreeShapeProp, GatherTree);
REG_SHAPE_INFER_FOR_TYPE(TopKShapeProp, TopK);
REG_SHAPE_INFER_FOR_TYPE(UniqueShapeProp, Unique);
REG_SHAPE_INFER_FOR_TYPE(NMSShapeProp, NonMaxSuppression);
REG_SHAPE_INFER_FOR_TYPE(ScatterUpdateShapeProp, ScatterUpdate);
REG_SHAPE_INFER_FOR_TYPE(ScatterElementsUpdateShapeProp, ScatterElementsUpdate);

}  // namespace ShapeInfer
}  // namespace InferenceEngine

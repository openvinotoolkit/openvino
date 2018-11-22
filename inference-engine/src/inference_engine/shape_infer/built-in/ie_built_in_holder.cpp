// Copyright (C) 2018 Intel Corporation
//
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
#include "ie_spatial_transformer_shape_infer.hpp"
#include "ie_inner_product_shape_infer.hpp"
#include "ie_resample_shape_infer.hpp"
#include "ie_interp_shape_infer.hpp"
#include "ie_argmax_shape_infer.hpp"
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
REG_SHAPE_INFER_FOR_TYPE(DoNothingShapeProp, Input);
REG_SHAPE_INFER_FOR_TYPE(DoNothingShapeProp, Memory);
REG_SHAPE_INFER_FOR_TYPE(DoNothingShapeProp, Const);

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
REG_SHAPE_INFER_FOR_TYPE(ReshapeShapeProp, Flatten);
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
REG_SHAPE_INFER_FOR_TYPE(CTCGreedyDecoderShapeProp, CTCGreedyDecoder);
REG_SHAPE_INFER_FOR_TYPE(ProposalShapeProp, Proposal);
REG_SHAPE_INFER_FOR_TYPE(ReorgYoloShapeProp, ReorgYolo);
REG_SHAPE_INFER_FOR_TYPE(RegionYoloShapeProp, RegionYolo);
REG_SHAPE_INFER_FOR_TYPE(ArgMaxShapeProp, ArgMax);

}  // namespace ShapeInfer
}  // namespace InferenceEngine

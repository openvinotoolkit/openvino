// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/ops.hpp"

#include "shared_test_classes/single_layer/roi_align.hpp"
#include "shared_test_classes/single_layer/psroi_pooling.hpp"
#include "shared_test_classes/read_ir/generate_inputs.hpp"

namespace LayerTestsDefinitions {

namespace {
InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::PSROIPooling> node,
                                    const InferenceEngine::InputInfo& info) {
    const auto& inputShape = info.getInputData()->getDims();
    if (inputShape.size() == 2) {
        InferenceEngine::Blob::Ptr blob;
        blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();
        PSROIPoolingLayerTest::fillROITensor(blob->buffer(),
                                             blob->size() / 5,
                                             inputShape[0],
                                             inputShape[2],
                                             inputShape[3],
                                             node->get_group_size(),
                                             node->get_spatial_scale(),
                                             node->get_spatial_bins_x(),
                                             node->get_spatial_bins_y(),
                                             node->get_mode());
        return blob;
    }
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::ROIPooling> node,
                                    const InferenceEngine::InputInfo& info) {
    const auto& inputShape = info.getInputData()->getDims();
    if (inputShape.size() == 2) {
        InferenceEngine::Blob::Ptr blob;
        blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        CommonTestUtils::fill_data_roi(blob->buffer(),
                                       blob->size(),
                                       node->get_input_shape(0).front() - 1,
                                       inputShape[2],
                                       inputShape[3],
                                       1.0f,
                                       node->get_method() == "max");
        return blob;
    }
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v3::ROIAlign> node,
                                    const InferenceEngine::InputInfo& info) {
    const auto& inputShape = info.getInputData()->getDims();
    switch (inputShape.size()) {
        case 1: {
            std::vector<int> roiIdxVector(node->get_shape()[0]);
            ROIAlignLayerTest::fillIdxTensor(roiIdxVector, node->get_shape()[0]);
            return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), roiIdxVector.data(), roiIdxVector.size());
        }
        case 2: {
            std::vector<float> blobData(node->get_shape()[0] * 4);
            ROIAlignLayerTest::fillCoordTensor(blobData,
                                               inputShape[2],
                                               inputShape[3],
                                               node->get_spatial_scale(),
                                               node->get_sampling_ratio(),
                                               node->get_pooled_h(),
                                               node->get_pooled_w());
            return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), blobData.data(), blobData.size());
        }
        default:
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
    }
}

template<typename T>
InferenceEngine::Blob::Ptr generateInput(const std::shared_ptr<ngraph::Node> node,
                                         const InferenceEngine::InputInfo& info) {
    return generate(ngraph::as_type_ptr<T>(node), info);
}
} // namespace

InputsMap getInputMap() {
    static InputsMap inputsMap{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, generateInput<NAMESPACE::NAME>},
    #include <shared_test_classes/read_ir/opset_int_tbl.hpp>
#undef NGRAPH_OP
    };
    return inputsMap;
}

} // namespace LayerTestsDefinitions
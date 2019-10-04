// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include <utility>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class NonMaxSuppressionImpl: public ExtLayerBase {
public:
    explicit NonMaxSuppressionImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() < 2 || layer->insData.size() > 5)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges!";

            if (layer->outData.size() != 1)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of output edges!";

            if (layer->insData[NMS_BOXES].lock()->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect 'boxes' input precision. Only FP32 is supported!";
            SizeVector boxes_dims = layer->insData[NMS_BOXES].lock()->getTensorDesc().getDims();
            if (boxes_dims.size() != 3 || boxes_dims[2] != 4)
                THROW_IE_EXCEPTION << layer->name << " 'boxes' should be with shape [num_batches, spatial_dimension, 4]";

            if (layer->insData[NMS_SCORES].lock()->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect 'scores' input precision. Only FP32 is supported!";
            SizeVector scores_dims = layer->insData[NMS_SCORES].lock()->getTensorDesc().getDims();
            if (scores_dims.size() != 3)
                THROW_IE_EXCEPTION << layer->name << " 'scores' should be with shape [num_batches, num_classes, spatial_dimension]";

            if (boxes_dims[0] != scores_dims[0])
                THROW_IE_EXCEPTION << layer->name << " num_batches is different in 'boxes' and 'scores' tensors";
            if (boxes_dims[1] != scores_dims[2])
                THROW_IE_EXCEPTION << layer->name << " spatial_dimension is different in 'boxes' and 'scores' tensors";

            if (layer->insData.size() > 2) {
                if (layer->insData[NMS_MAXOUTPUTBOXESPERCLASS].lock()->getTensorDesc().getPrecision() != Precision::I32)
                    THROW_IE_EXCEPTION << layer->name << " Incorrect 'max_output_boxes_per_class' input precision. Only I32 is supported!";
                SizeVector max_output_boxes_per_class_dims = layer->insData[NMS_MAXOUTPUTBOXESPERCLASS].lock()->getTensorDesc().getDims();
                if (max_output_boxes_per_class_dims.size() != 1 || max_output_boxes_per_class_dims[0] != 1)
                    THROW_IE_EXCEPTION << layer->name << " 'max_output_boxes_per_class' should be scalar";
            }

            if (layer->insData.size() > 3) {
                if (layer->insData[NMS_IOUTHRESHOLD].lock()->getTensorDesc().getPrecision() != Precision::FP32)
                    THROW_IE_EXCEPTION << layer->name << " Incorrect 'iou_threshold' input precision. Only FP32 is supported!";
                SizeVector iou_threshold_dims = layer->insData[NMS_IOUTHRESHOLD].lock()->getTensorDesc().getDims();
                if (iou_threshold_dims.size() != 1 || iou_threshold_dims[0] != 1)
                    THROW_IE_EXCEPTION << layer->name << " 'iou_threshold' should be scalar";
            }

            if (layer->insData.size() > 4) {
                if (layer->insData[NMS_SCORETHRESHOLD].lock()->getTensorDesc().getPrecision() != Precision::FP32)
                    THROW_IE_EXCEPTION << layer->name << " Incorrect 'score_threshold' input precision. Only FP32 is supported!";
                SizeVector score_threshold_dims = layer->insData[NMS_SCORETHRESHOLD].lock()->getTensorDesc().getDims();
                if (score_threshold_dims.size() != 1 || score_threshold_dims[0] != 1)
                    THROW_IE_EXCEPTION << layer->name << " 'score_threshold' should be scalar";
            }

            if (layer->outData[0]->getTensorDesc().getPrecision() != Precision::I32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect 'selected_indices' input precision. Only I32 is supported!";
            SizeVector selected_indices_dims = layer->outData[0]->getTensorDesc().getDims();
            if (selected_indices_dims.size() != 2 || selected_indices_dims[1] != 3)
                THROW_IE_EXCEPTION << layer->name << " 'selected_indices' should be with shape [num_selected_indices, 3]";

            center_point_box = layer->GetParamAsBool("center_point_box", false);

            if (layer->insData.size() == 2) {
                addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
            } else if (layer->insData.size() == 3) {
                addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                    { DataConfigurator(ConfLayout::PLN) });
            } else if (layer->insData.size() == 4) {
                addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN),
                    DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
            } else {
                addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN),
                    DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
            }
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    static float intersectionOverUnion(float* boxesI, float* boxesJ, bool center_point_box) {
        float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
        if (center_point_box) {
            //  box format: x_center, y_center, width, height
            yminI = boxesI[1] - boxesI[3] / 2.f;
            xminI = boxesI[0] - boxesI[2] / 2.f;
            ymaxI = boxesI[1] + boxesI[3] / 2.f;
            xmaxI = boxesI[0] + boxesI[2] / 2.f;
            yminJ = boxesJ[1] - boxesJ[3] / 2.f;
            xminJ = boxesJ[0] - boxesJ[2] / 2.f;
            ymaxJ = boxesJ[1] + boxesJ[3] / 2.f;
            xmaxJ = boxesJ[0] + boxesJ[2] / 2.f;
        } else {
            //  box format: y1, x1, y2, x2
            yminI = (std::min)(boxesI[0], boxesI[2]);
            xminI = (std::min)(boxesI[1], boxesI[3]);
            ymaxI = (std::max)(boxesI[0], boxesI[2]);
            xmaxI = (std::max)(boxesI[1], boxesI[3]);
            yminJ = (std::min)(boxesJ[0], boxesJ[2]);
            xminJ = (std::min)(boxesJ[1], boxesJ[3]);
            ymaxJ = (std::max)(boxesJ[0], boxesJ[2]);
            xmaxJ = (std::max)(boxesJ[1], boxesJ[3]);
        }

        float areaI = (ymaxI - yminI) * (xmaxI - xminI);
        float areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
        if (areaI <= 0.f || areaJ <= 0.f)
            return 0.f;

        float intersection_area =
            (std::max)((std::min)(ymaxI, ymaxJ) - (std::max)(yminI, yminJ), 0.f) *
            (std::max)((std::min)(xmaxI, xmaxJ) - (std::max)(xminI, xminJ), 0.f);
        return intersection_area / (areaI + areaJ - intersection_area);
    }

    typedef struct {
        float score;
        int batch_index;
        int class_index;
        int box_index;
    } filteredBoxes;

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        float *boxes = inputs[NMS_BOXES]->cbuffer().as<float *>() +
            inputs[NMS_BOXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float *scores = inputs[NMS_SCORES]->cbuffer().as<float *>() +
            inputs[NMS_SCORES]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        SizeVector scores_dims = inputs[NMS_SCORES]->getTensorDesc().getDims();
        int num_boxes = static_cast<int>(scores_dims[2]);
        int max_output_boxes_per_class = num_boxes;
        if (inputs.size() > 2)
            max_output_boxes_per_class = (std::min)(max_output_boxes_per_class,
                (inputs[NMS_MAXOUTPUTBOXESPERCLASS]->cbuffer().as<int *>() +
                inputs[NMS_MAXOUTPUTBOXESPERCLASS]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0]);

        float iou_threshold = 1.f;  //  Value range [0, 1]
        if (inputs.size() > 3)
            iou_threshold = (std::min)(iou_threshold, (inputs[NMS_IOUTHRESHOLD]->cbuffer().as<float *>() +
                inputs[NMS_IOUTHRESHOLD]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0]);

        float score_threshold = 0.f;
        if (inputs.size() > 4)
            score_threshold = (inputs[NMS_SCORETHRESHOLD]->cbuffer().as<float *>() +
                inputs[NMS_SCORETHRESHOLD]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];
        int* selected_indices = outputs[0]->cbuffer().as<int *>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        SizeVector selected_indices_dims = outputs[0]->getTensorDesc().getDims();

        SizeVector boxesStrides = inputs[NMS_BOXES]->getTensorDesc().getBlockingDesc().getStrides();
        SizeVector scoresStrides = inputs[NMS_SCORES]->getTensorDesc().getBlockingDesc().getStrides();

        // boxes shape: {num_batches, num_boxes, 4}
        // scores shape: {num_batches, num_classes, num_boxes}
        int num_batches = static_cast<int>(scores_dims[0]);
        int num_classes = static_cast<int>(scores_dims[1]);
        std::vector<filteredBoxes> fb;

        for (int batch = 0; batch < num_batches; batch++) {
            float *boxesPtr = boxes + batch * boxesStrides[0];
            for (int class_idx = 0; class_idx < num_classes; class_idx++) {
                float *scoresPtr = scores + batch * scoresStrides[0] + class_idx * scoresStrides[1];
                std::vector<std::pair<float, int> > scores_vector;
                for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
                    if (scoresPtr[box_idx] > score_threshold)
                        scores_vector.push_back(std::make_pair(scoresPtr[box_idx], box_idx));
                }

                if (scores_vector.size()) {
                    parallel_sort(scores_vector.begin(), scores_vector.end(),
                        [](const std::pair<float, int>& l, const std::pair<float, int>& r) { return l.first > r.first; });

                    int io_selection_size = 1;
                    fb.push_back({ scores_vector[0].first, batch, class_idx, scores_vector[0].second });
                    for (int box_idx = 1; (box_idx < static_cast<int>(scores_vector.size()) && io_selection_size < max_output_boxes_per_class); box_idx++) {
                        bool box_is_selected = true;
                        for (int idx = io_selection_size - 1; idx >= 0; idx--) {
                            float iou = intersectionOverUnion(&boxesPtr[scores_vector[box_idx].second * 4],
                                             &boxesPtr[scores_vector[idx].second * 4], center_point_box);
                            if (iou > iou_threshold) {
                                box_is_selected = false;
                                break;
                            }
                        }

                        if (box_is_selected) {
                            scores_vector[io_selection_size] = scores_vector[box_idx];
                            io_selection_size++;
                            fb.push_back({ scores_vector[box_idx].first, batch, class_idx, scores_vector[box_idx].second });
                        }
                    }
                }
            }
        }

        parallel_sort(fb.begin(), fb.end(), [](const filteredBoxes& l, const filteredBoxes& r) { return l.score > r.score; });
        int selected_indicesStride = outputs[0]->getTensorDesc().getBlockingDesc().getStrides()[0];
        int* selected_indicesPtr = selected_indices;
        size_t idx;
        for (idx = 0; idx < (std::min)(selected_indices_dims[0], fb.size()); idx++) {
            selected_indicesPtr[0] = fb[idx].batch_index;
            selected_indicesPtr[1] = fb[idx].class_index;
            selected_indicesPtr[2] = fb[idx].box_index;
            selected_indicesPtr += selected_indicesStride;
        }
        for (; idx < selected_indices_dims[0]; idx++) {
            selected_indicesPtr[0] = -1;
            selected_indicesPtr[1] = -1;
            selected_indicesPtr[2] = -1;
            selected_indicesPtr += selected_indicesStride;
        }

        return OK;
    }

private:
    const size_t NMS_BOXES = 0;
    const size_t NMS_SCORES = 1;
    const size_t NMS_MAXOUTPUTBOXESPERCLASS = 2;
    const size_t NMS_IOUTHRESHOLD = 3;
    const size_t NMS_SCORETHRESHOLD = 4;
    bool center_point_box = false;
};

REG_FACTORY_FOR(ImplFactory<NonMaxSuppressionImpl>, NonMaxSuppression);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine

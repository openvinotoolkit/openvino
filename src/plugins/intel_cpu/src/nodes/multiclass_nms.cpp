// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multiclass_nms.hpp"
#include "ngraph_ops/multiclass_nms_ie_internal.hpp"

#include <algorithm>
#include <cmath>
#include <ie_ngraph_utils.hpp>

#include "ie_parallel.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

//
// boxes:   [num_batches, num_boxes, 4]           // The boxes are shared by all classes.
// scores:  [num_batches, num_classes, num_boxes]
//
class MultiClassNms::InputsFormatFirst : public MultiClassNms::Inputs {
public:
    Format format() const override { return Format::first; }

    void prepareParams(const Node& node, const std::string& err_str) override {
        const auto& boxes_dims = node.getParentEdgeAt(NMS_BOXES)->getMemory().getStaticDims();
        const auto& scores_dims = node.getParentEdgeAt(NMS_SCORES)->getMemory().getStaticDims();
        if (boxes_dims[0] != scores_dims[0] || boxes_dims[1] != scores_dims[2])
            IE_THROW() << err_str << "has incompatible 'boxes' and 'scores' shape " << PartialShape(boxes_dims) << " v.s. " << PartialShape(scores_dims);
        num_batches_ = boxes_dims[0];
        num_boxes_ = boxes_dims[1];
        num_classes_ = scores_dims[1];
    }

    const float* boxes(const int batch_idx, const int, const float* boxes, const int*) const override {
        return boxes + batch_idx * num_boxes_ * 4;
    }

    const float* scores(const int batch_idx, const int class_idx, const float* scores, const int*) const override {
        return scores + batch_idx * num_classes_ * num_boxes_ + class_idx * num_boxes_;
    }

    int num_input_boxes(const int batch_idx, const int* roisnum) const override {
        return num_boxes_;
    }

    void locate(const Box& box, const int* roisnum, int& input_box_index, int& input_box_offset) override {
        const int index = box.batch_idx * num_boxes_ + box.box_idx;
        input_box_index = index;
        input_box_offset = index * 4;
    }
};

//
// boxes:   [num_classes, num_boxes, 4] // where `num_boxes` is the sum of boxes from all batches
// scores:  [num_classes, num_boxes]    // where `num_boxes` is the sum of boxes from all batches
// roisnum: [num_batches]               // The sum of all elements is `num_boxes`
//
class MultiClassNms::InputsFormatSecond : public MultiClassNms::Inputs {
public:
    Format format() const override { return Format::second; }

    void prepareParams(const Node& node, const std::string& err_str) override {
        const auto& boxes_dims = node.getParentEdgeAt(NMS_BOXES)->getMemory().getStaticDims();
        const auto& scores_dims = node.getParentEdgeAt(NMS_SCORES)->getMemory().getStaticDims();
        if (boxes_dims[0] != scores_dims[0] || boxes_dims[1] != scores_dims[1])
            IE_THROW() << err_str << "has incompatible 'boxes' and 'scores' shape " << PartialShape(boxes_dims) << " v.s. " << PartialShape(scores_dims);
        const auto& roisnum_dims = node.getParentEdgeAt(NMS_ROISNUM)->getMemory().getStaticDims();
        if (roisnum_dims.size() != 1)
            IE_THROW() << err_str << "has unsupported 'roisnum' input rank: " << roisnum_dims.size();
        num_batches_ = roisnum_dims[0];
        num_boxes_ = boxes_dims[1];
        num_classes_ = scores_dims[0];
    }

    const float* boxes(const int batch_idx, const int class_idx, const float* boxes, const int* roisnum) const override {
        const int batch_offset = std::accumulate(roisnum, roisnum + batch_idx, int{});
        return boxes + (class_idx * num_boxes_ + batch_offset) * 4;
    }

    const float* scores(const int batch_idx, const int class_idx, const float* scores, const int* roisnum) const override {
        const int batch_offset = std::accumulate(roisnum, roisnum + batch_idx, int{});
        return scores + class_idx * num_boxes_ + batch_offset;
    }

    int num_input_boxes(const int batch_idx, const int* roisnum) const override {
        return roisnum[batch_idx];
    }

    void locate(const Box& box, const int* roisnum, int& input_box_index, int& input_box_offset) override {
        const int batch_offset = std::accumulate(roisnum, roisnum + box.batch_idx, int{});
        input_box_index = (batch_offset + box.box_idx) * num_classes_ + box.class_idx; // TODO: bug in reference impl?
        input_box_offset = (box.class_idx * num_boxes_ + batch_offset + box.box_idx) * 4;
    }
};

MultiClassNms::Workbuffers::Workbuffers(size_t num_batches, size_t num_classes, size_t num_boxes)
    : boxes(num_batches * num_classes * num_boxes),
      num_boxes_per_batch_and_class(num_batches),
      num_boxes_per_batch(num_batches, 0),
      num_batches_{num_batches}, num_classes_{num_classes}, num_boxes_{num_boxes} {
    for (auto &class_dim : num_boxes_per_batch_and_class) {
        class_dim.resize(num_classes, 0);
    }
}

MultiClassNms::Buffer MultiClassNms::Workbuffers::flatten_all() {
    size_t dst_offset = 0;
    for (size_t batch_idx = 0; batch_idx < num_batches_; ++batch_idx) {
        size_t num_boxes_in_batch = 0;
        for (size_t class_idx = 0; class_idx < num_classes_; ++class_idx) {
            const size_t num_filt_boxes = num_boxes_per_batch_and_class[batch_idx][class_idx];
            const Box* const src_buffer = thread_workspace_for(batch_idx, class_idx);
            for (size_t i = 0; i < num_filt_boxes; i++) {
                boxes[dst_offset + i] = src_buffer[i];
            }
            dst_offset += num_filt_boxes;
            num_boxes_in_batch += num_filt_boxes;
        }
        num_boxes_per_batch[batch_idx] = static_cast<int>(num_boxes_in_batch);
    }
    return Buffer {boxes.data(), dst_offset};
}

MultiClassNms::Buffer MultiClassNms::Workbuffers::flatten_within_batch(int batch_idx) {
    const size_t batch_offset = batch_idx * num_classes_ * num_boxes_;
    size_t dst_offset = batch_offset;
    for (size_t class_idx = 0; class_idx < num_classes_; ++class_idx) {
        const size_t num_filt_boxes = num_boxes_per_batch_and_class[batch_idx][class_idx];
        if (class_idx > 0) {
            const size_t src_offset = batch_offset + class_idx * num_boxes_;
            for (size_t i = 0; i < num_filt_boxes; i++) {
                boxes[dst_offset + i] = boxes[src_offset + i];
            }
        }
        dst_offset += num_filt_boxes;
    }
    return Buffer {boxes.data() + batch_offset, dst_offset - batch_offset};
}

MultiClassNms::Buffer MultiClassNms::Workbuffers::flatten_batches() {
    size_t dst_offset = 0;
    for (size_t batch_idx = 0; batch_idx < num_batches_; ++batch_idx) {
        const size_t batch_offset = batch_idx * num_classes_ * num_boxes_;
        const size_t num_boxes_in_batch = num_boxes_per_batch[batch_idx];
        if (batch_idx > 0) {
            for (size_t i = 0; i < num_boxes_in_batch; i++) {
                boxes[dst_offset + i] = boxes[batch_offset + i];
            }
        }
        dst_offset += num_boxes_in_batch;
    }
    return Buffer {boxes.data(), dst_offset};
}

void MultiClassNms::OutputsLayout::prepareParams(size_t num_classes, size_t num_boxes,
                                                 int nms_top_k, int keep_top_k, int background_class) {
    const int max_boxes_per_class = (nms_top_k < 0) ? num_boxes : std::min(nms_top_k, static_cast<int>(num_boxes));
    boxes_per_batch_ = max_boxes_per_class * num_classes;

    if (background_class >= 0 && background_class < num_classes)
        boxes_per_batch_ -= max_boxes_per_class;

    if (keep_top_k >= 0)
        boxes_per_batch_ = std::min(boxes_per_batch_, keep_top_k);
}

MultiClassNms::MultiClassNms(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr& cache)
    : Node(op, eng, cache),
      err_str_prefix_{"MultiClassNms layer with name '" + getName() + "' "}
{
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    const auto& boxes_dims = getInputShapeAtPort(NMS_BOXES).getDims();
    const auto& scores_dims = getInputShapeAtPort(NMS_SCORES).getDims();
    auto boxes_ps = PartialShape(boxes_dims);
    auto scores_ps = PartialShape(scores_dims);
    if (boxes_dims.size() != 3)
        IE_THROW() << err_str_prefix_ << "has unsupported 'boxes' input rank: " << boxes_dims.size();
    if (boxes_dims[2] != 4)
        IE_THROW() << err_str_prefix_ << "has unsupported 'boxes' input 3rd dimension size: " << boxes_dims[2];

    switch (getOriginalInputsNumber()) {
    case 2:
        // boxes [N, M, 4], scores [N, C, M] opset8/9
        inputs_.reset(new InputsFormatFirst());
        if (scores_dims.size() != 3)
            IE_THROW() << err_str_prefix_ << "has unsupported 'scores' input rank: " << scores_dims.size();
        if (!boxes_ps[0].compatible(scores_ps[0]) || !boxes_ps[1].compatible(scores_ps[2]))
            IE_THROW() << err_str_prefix_ << "has incompatible 'boxes' and 'scores' shape " << boxes_ps << " v.s. " << scores_ps;
        break;
    case 3:
        // boxes [C, M, 4], scores [C, M], roisnum [N] opset9
        inputs_.reset(new InputsFormatSecond());
        if (op->get_type_info() == ov::op::v8::MulticlassNms::get_type_info_static())
            IE_THROW() << err_str_prefix_ << "has input format which is not supported in opset8";
        if (scores_dims.size() != 2)
            IE_THROW() << err_str_prefix_ << "has unsupported 'scores' input rank: " << scores_dims.size();
        if (!boxes_ps[0].compatible(scores_ps[0]) || !boxes_ps[1].compatible(scores_ps[1]))
            IE_THROW() << err_str_prefix_ << "has incompatible 'boxes' and 'scores' shape " << boxes_ps << " v.s. " << scores_ps;
        break;
    default:
        IE_THROW() << err_str_prefix_ << "has incorrect number of input edges: " << getOriginalInputsNumber();
    }

    if (getOriginalOutputsNumber() != 3)
        IE_THROW() << err_str_prefix_ << "has incorrect number of output edges: " << getOriginalOutputsNumber();

    auto nmsBase = std::dynamic_pointer_cast<ov::op::util::MulticlassNmsBase>(op);
    if (nmsBase == nullptr)
        IE_THROW() << err_str_prefix_ << " is not an instance of MulticlassNmsBase.";

    auto& atrri = nmsBase->get_attrs();
    attr_sort_result_across_batch_ = atrri.sort_result_across_batch;
    attr_nms_top_k_ = atrri.nms_top_k;
    attr_iou_threshold_ = atrri.iou_threshold;
    attr_score_threshold_ = atrri.score_threshold;
    attr_background_class_ = atrri.background_class;
    attr_keep_top_k_ = atrri.keep_top_k;

    using OpSortResultType = ov::op::util::MulticlassNmsBase::SortResultType;
    if (atrri.sort_result_type == OpSortResultType::CLASSID)
        attr_sort_result_ = SortResultType::CLASSID;
    else if (atrri.sort_result_type == OpSortResultType::SCORE)
        attr_sort_result_ = SortResultType::SCORE;
    else if (atrri.sort_result_type == OpSortResultType::NONE)
        attr_sort_result_ = SortResultType::NONE;

    attr_nms_eta_ = atrri.nms_eta;
    if ((attr_nms_eta_ < 0.0f) || (attr_nms_eta_ > 1.0f)) {
        IE_THROW() << err_str_prefix_ << "has invalid 'nms_eta' attribute value: " << attr_nms_eta_;
    }
    attr_normalized_ = atrri.normalized;
}

bool MultiClassNms::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ov::op::v9::MulticlassNms::get_type_info_static(),
                ov::op::v8::MulticlassNms::get_type_info_static(),
                ngraph::op::internal::MulticlassNmsIEInternal::get_type_info_static())) {
            errorMessage = "Node is not an instance of MulticlassNms from opset v8 or v9.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void MultiClassNms::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const std::vector<Precision> supportedFloatPrecision = {Precision::FP32, Precision::BF16};
    const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32, Precision::I64};

    const char* inTypeStr = "input";
    const char* outTypeStr = "output";

    auto checkPrecision = [&](const Precision prec, const std::vector<Precision> precList, const std::string name, const char* type) {
        if (std::find(precList.begin(), precList.end(), prec) == precList.end())
            IE_THROW() << err_str_prefix_ << "has unsupported '" << name << "' " << type << " precision: " << prec;
    };

    checkPrecision(getOriginalInputPrecisionAtPort(NMS_BOXES), supportedFloatPrecision, "boxes", inTypeStr);
    checkPrecision(getOriginalInputPrecisionAtPort(NMS_SCORES), supportedFloatPrecision, "scores", inTypeStr);

    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTEDINDICES), supportedIntOutputPrecision, "selected_indices", outTypeStr);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTEDOUTPUTS), supportedFloatPrecision, "selected_outputs", outTypeStr);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTEDNUM), supportedIntOutputPrecision, "selected_num", outTypeStr);

    if (inputs_->format() == Inputs::Format::second) {
        checkPrecision(getOriginalInputPrecisionAtPort(NMS_ROISNUM), supportedIntOutputPrecision, "roisnum", inTypeStr);
        addSupportedPrimDesc({{LayoutType::ncsp, Precision::FP32},
                            {LayoutType::ncsp, Precision::FP32},
                            {LayoutType::ncsp, Precision::I32}},
                            {{LayoutType::ncsp, Precision::FP32},
                            {LayoutType::ncsp, Precision::I32},
                            {LayoutType::ncsp, Precision::I32}},
                            impl_desc_type::ref_any);
    } else {
        addSupportedPrimDesc({{LayoutType::ncsp, Precision::FP32},
                            {LayoutType::ncsp, Precision::FP32}},
                            {{LayoutType::ncsp, Precision::FP32},
                            {LayoutType::ncsp, Precision::I32},
                            {LayoutType::ncsp, Precision::I32}},
                            impl_desc_type::ref_any);
    }
}

bool MultiClassNms::created() const {
    return getType() == Type::MulticlassNms;
}

bool MultiClassNms::isExecutable() const {
    return isDynamicNode() || Node::isExecutable();
}

void MultiClassNms::prepareParams() {
    inputs_->prepareParams(*this, err_str_prefix_);
    const size_t num_batches = inputs_->num_batches();
    const size_t num_classes = inputs_->num_classes();
    const size_t num_boxes = inputs_->num_boxes();

    workbuffers_.reset(new Workbuffers{num_batches, num_classes, num_boxes});
    output_layout_.prepareParams(num_classes, num_boxes, attr_nms_top_k_, attr_keep_top_k_, attr_background_class_);
}

void MultiClassNms::executeDynamicImpl(dnnl::stream strm) {
    if (hasEmptyInputTensors()) {
        redefineOutputMemory({{0, 6}, {0, 1}, {0}});
        return;
    }
    execute(strm);
}

void MultiClassNms::execute(dnnl::stream strm) {
    const float* boxes = reinterpret_cast<const float*>(getParentEdgeAt(NMS_BOXES)->getMemory().GetPtr());
    const float* scores = reinterpret_cast<const float*>(getParentEdgeAt(NMS_SCORES)->getMemory().GetPtr());
    const int* roisnum = (inputs_->format() == Inputs::Format::second) ?
        reinterpret_cast<int*>(getParentEdgeAt(NMS_ROISNUM)->getMemory().GetPtr()) : nullptr;

    multiclass_nms(boxes, scores, roisnum);

    Buffer rois {};
    if (attr_keep_top_k_ < 0) {
        rois = workbuffers_->flatten_all();
    } else {
        parallel_for(workbuffers_->num_boxes_per_batch_and_class.size(), [&](int batch_idx) {
            auto batch_buffer = workbuffers_->flatten_within_batch(batch_idx);
            const int num_boxes_per_batch = std::min<int>(batch_buffer.size, attr_keep_top_k_);
            partial_sort_by_score(batch_buffer, num_boxes_per_batch);
            workbuffers_->num_boxes_per_batch[batch_idx] = num_boxes_per_batch;
        });
        rois = workbuffers_->flatten_batches();
    }

    if (attr_sort_result_across_batch_ && (attr_sort_result_ != SortResultType::NONE)) {
        if (attr_sort_result_ == SortResultType::SCORE) {
            parallel_sort_by_score_across_batch(rois);
        } else if (attr_sort_result_ == SortResultType::CLASSID) {
            parallel_sort_by_class_across_batch(rois);
        }
    } else {
        if (attr_sort_result_ == SortResultType::SCORE || attr_sort_result_ == SortResultType::NONE) {
            if (attr_keep_top_k_ < 0) {
                parallel_sort_by_score(rois);
            } else {
                // already sorted
            }
        } else if (attr_sort_result_ == SortResultType::CLASSID) {
            parallel_sort_by_class(rois);
        }
    }

    fill_outputs(rois, boxes, roisnum);
}

float MultiClassNms::intersectionOverUnion(const float* boxesI, const float* boxesJ, const bool normalized) {
    float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
    const float norm = static_cast<float>(normalized == false);

    // to align with reference
    yminI = boxesI[0];
    xminI = boxesI[1];
    ymaxI = boxesI[2];
    xmaxI = boxesI[3];
    yminJ = boxesJ[0];
    xminJ = boxesJ[1];
    ymaxJ = boxesJ[2];
    xmaxJ = boxesJ[3];

    float areaI = (ymaxI - yminI + norm) * (xmaxI - xminI + norm);
    float areaJ = (ymaxJ - yminJ + norm) * (xmaxJ - xminJ + norm);
    if (areaI <= 0.f || areaJ <= 0.f)
        return 0.f;

    float intersection_area = (std::max)((std::min)(ymaxI, ymaxJ) - (std::max)(yminI, yminJ) + norm, 0.f) *
                              (std::max)((std::min)(xmaxI, xmaxJ) - (std::max)(xminI, xminJ) + norm, 0.f);
    return intersection_area / (areaI + areaJ - intersection_area);
}

void MultiClassNms::multiclass_nms(const float* in_boxes, const float* in_scores, const int* in_roisnum) {
    parallel_for2d(inputs_->num_batches(), inputs_->num_classes(), [&](int batch_idx, int class_idx) {
        const int num_input_boxes = inputs_->num_input_boxes(batch_idx, in_roisnum);
        if ((num_input_boxes == 0) || (class_idx == attr_background_class_)) {
            workbuffers_->num_boxes_per_batch_and_class[batch_idx][class_idx] = 0;
            return;
        }

        const float* input_boxes = inputs_->boxes(batch_idx, class_idx, in_boxes, in_roisnum);
        const float* input_scores = inputs_->scores(batch_idx, class_idx, in_scores, in_roisnum);

        Box* const boxes = workbuffers_->thread_workspace_for(batch_idx, class_idx);

        int num_boxes_selected = 0;
        for (int i = 0; i < num_input_boxes; ++i) {
            const float score = input_scores[i];
            if (score >= attr_score_threshold_) {
                Box& selected_box = boxes[num_boxes_selected++];
                selected_box.score = score;
                selected_box.batch_idx = batch_idx;
                selected_box.class_idx = class_idx;
                selected_box.box_idx = i;
            }
        }

        // empirically derived given the differences in implementation of sort(quick_sort) and partial_sort(heap_sort)
        static constexpr int sort_type_threshold = 5;

        const int num_boxes_per_class = (attr_nms_top_k_ < 0) ? num_boxes_selected : std::min(attr_nms_top_k_, num_boxes_selected);
        if (num_boxes_per_class == 0) {
            workbuffers_->num_boxes_per_batch_and_class[batch_idx][class_idx] = 0;
            return;
        } else if (num_boxes_selected / num_boxes_per_class < sort_type_threshold) {
            parallel_sort(boxes, boxes + num_boxes_selected, [](const Box& l, const Box& r) {
                return (l.score > r.score || ((l.score == r.score) && (l.box_idx < r.box_idx)));
            });
        } else {
            std::partial_sort(boxes, boxes + num_boxes_per_class, boxes + num_boxes_selected,
                [](const Box& l, const Box& r) {
                    return (l.score > r.score || ((l.score == r.score) && (l.box_idx < r.box_idx)));
                });
        }

        num_boxes_selected = 0;
        float iou_threshold = attr_iou_threshold_;
        for (size_t i = 0; i < num_boxes_per_class; i++) {
            bool box_is_selected = true;
            for (int j = num_boxes_selected - 1; j >= 0; j--) {
                const float iou = intersectionOverUnion(&input_boxes[boxes[i].box_idx * 4],
                                                        &input_boxes[boxes[j].box_idx * 4],
                                                        attr_normalized_);
                if (iou >= iou_threshold) {
                    box_is_selected = false;
                    break;
                }
                if (boxes[i].score == attr_score_threshold_)    // TODO: bug in reference impl?
                    break;
            }
            if (box_is_selected) {
                if (iou_threshold > 0.5f) {
                    iou_threshold *= attr_nms_eta_;
                }
                boxes[num_boxes_selected++] = boxes[i];
            }
        }

        workbuffers_->num_boxes_per_batch_and_class[batch_idx][class_idx] = num_boxes_selected;
    });
}

void MultiClassNms::partial_sort_by_score(Buffer b, int mid_size) {
    std::partial_sort(b.boxes, b.boxes + mid_size, b.boxes + b.size, [](const Box& l, const Box& r) {
        if (l.score > r.score) {
            return true;
        } else if (std::fabs(l.score - r.score) < 1e-6f) {
            if (l.class_idx < r.class_idx)
                return true;
            else if (l.class_idx == r.class_idx)
                return l.box_idx < r.box_idx;
        }
        return false;
    });
}

void MultiClassNms::parallel_sort_by_score(Buffer b) {
    parallel_sort(b.boxes, b.boxes + b.size, [](const Box& l, const Box& r) {
        if (l.batch_idx < r.batch_idx) {
            return true;
        } else if (l.batch_idx == r.batch_idx) {
            if (l.score > r.score) {
                return true;
            } else if (std::fabs(l.score - r.score) < 1e-6f) {
                if (l.class_idx < r.class_idx)
                    return true;
                else if (l.class_idx == r.class_idx)
                    return l.box_idx < r.box_idx;
            }
        }
        return false;
    });
}

void MultiClassNms::parallel_sort_by_score_across_batch(Buffer b) {
    parallel_sort(b.boxes, b.boxes + b.size, [](const Box& l, const Box& r) {
        if (l.score > r.score) {
            return true;
        } else if (l.score == r.score) {
            if (l.batch_idx < r.batch_idx) {
                return true;
            } else if (l.batch_idx == r.batch_idx) {
                if (l.class_idx < r.class_idx)
                    return true;
                else if (l.class_idx == r.class_idx)
                    return l.box_idx < r.box_idx;
            }
        }
        return false;
    });
}

void MultiClassNms::parallel_sort_by_class(Buffer b) {
    parallel_sort(b.boxes, b.boxes + b.size, [](const Box& l, const Box& r) {
        if (l.batch_idx < r.batch_idx) {
            return true;
        } else if (l.batch_idx == r.batch_idx) {
            if (l.class_idx < r.class_idx) {
                return true;
            } else if (l.class_idx == r.class_idx) {
                if (l.score > r.score)
                    return true;
                else if (std::fabs(l.score - r.score) <= 1e-6f)
                    return l.box_idx < r.box_idx;
            }
        }
        return false;
    });
}

void MultiClassNms::parallel_sort_by_class_across_batch(Buffer b) {
    parallel_sort(b.boxes, b.boxes + b.size, [](const Box& l, const Box& r) {
        if (l.class_idx < r.class_idx) {
            return true;
        } else if (l.class_idx == r.class_idx) {
            if (l.batch_idx < r.batch_idx) {
                return true;
            } else if (l.batch_idx == r.batch_idx) {
                if (l.score > r.score)
                    return true;
                else if (l.score == r.score)
                    return l.box_idx < r.box_idx;
            }
        }
        return false;
    });
}

//
// selected_outputs    [number of selected boxes, 6]    // {class_id, box_score, xmin, ymin, xmax, ymax}
// selected_indices    [number of selected boxes, 1]    // selected indices in the flattened `boxes`, which are absolute values cross batches
// selected_num        [num_batches]                    // number of selected boxes for each batch element
//
void MultiClassNms::fill_outputs(Buffer rois,
                                 const float* input_boxes,
                                 const int* input_roisnum) {
    // TODO [DS NMS]: remove when nodes from models where nms is not last node in model supports DS
    const bool shapes_are_dynamic = isDynamicNode();
    if (shapes_are_dynamic) {
        redefineOutputMemory({{rois.size, 6}, {rois.size, 1}, {inputs_->num_batches()}});
    }

    float* const selected_outputs = reinterpret_cast<float*>(getChildEdgesAtPort(NMS_SELECTEDOUTPUTS)[0]->getMemoryPtr()->GetPtr());
    int* const selected_indices = reinterpret_cast<int*>(getChildEdgesAtPort(NMS_SELECTEDINDICES)[0]->getMemoryPtr()->GetPtr());
    int* const selected_num = reinterpret_cast<int*>(getChildEdgesAtPort(NMS_SELECTEDNUM)[0]->getMemoryPtr()->GetPtr());

    int64_t dst_batch_offset = 0;
    int64_t src_batch_offset = 0;

    for (size_t batch_idx = 0; batch_idx < inputs_->num_batches(); ++batch_idx) {
        const int num_boxes_in_batch = workbuffers_->num_boxes_per_batch[batch_idx];
        selected_num[batch_idx] = num_boxes_in_batch;

        for (size_t box_idx = 0; box_idx < num_boxes_in_batch; ++box_idx) {
            const auto &box = rois.boxes[src_batch_offset + box_idx];
            const int dst_box_idx = dst_batch_offset + box_idx;

            float* const tuple = selected_outputs + dst_box_idx * 6;
            tuple[0] = box.class_idx;
            tuple[1] = box.score;

            int input_box_offset {};
            inputs_->locate(box, input_roisnum, selected_indices[dst_box_idx], input_box_offset);
            tuple[2] = input_boxes[input_box_offset];
            tuple[3] = input_boxes[input_box_offset + 1];
            tuple[4] = input_boxes[input_box_offset + 2];
            tuple[5] = input_boxes[input_box_offset + 3];
        }

        // TODO [DS NMS]: remove when nodes from models where nms is not last node in model supports DS
        if (shapes_are_dynamic) {
            dst_batch_offset += num_boxes_in_batch;
            src_batch_offset += num_boxes_in_batch;
        } else {
            std::fill_n(selected_outputs + (dst_batch_offset + num_boxes_in_batch) * 6,
                        (output_layout_.boxes_per_batch() - num_boxes_in_batch) * 6,
                        -1.f);
            std::fill_n(selected_indices + (dst_batch_offset + num_boxes_in_batch),
                        output_layout_.boxes_per_batch() - num_boxes_in_batch,
                        -1);
            dst_batch_offset += output_layout_.boxes_per_batch();
            src_batch_offset += num_boxes_in_batch;
        }
    }
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

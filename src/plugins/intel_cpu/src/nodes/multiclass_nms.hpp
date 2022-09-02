// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>

namespace ov {
namespace intel_cpu {
namespace node {

class MultiClassNms : public Node {
public:
    MultiClassNms(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr& cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    bool isExecutable() const override;
    void executeDynamicImpl(dnnl::stream strm) override;

    bool needShapeInfer() const override { return false; }
    void prepareParams() override;

private:
    // input (port Num)
    constexpr static size_t NMS_BOXES = 0;
    constexpr static size_t NMS_SCORES = 1;
    constexpr static size_t NMS_ROISNUM = 2;

    // output (port Num)
    constexpr static size_t NMS_SELECTEDOUTPUTS = 0;
    constexpr static size_t NMS_SELECTEDINDICES = 1;
    constexpr static size_t NMS_SELECTEDNUM = 2;

    std::string err_str_prefix_;

private:
    // Attributes
    int attr_nms_top_k_ = 0;
    float attr_iou_threshold_ = 0.0f;
    float attr_score_threshold_ = 0.0f;
    int attr_background_class_ = 0;
    int attr_keep_top_k_ = 0;
    float attr_nms_eta_ = 0.0f;
    bool attr_normalized_ = true;
    bool attr_sort_result_across_batch_ = false;
    enum class SortResultType { CLASSID, SCORE, NONE };
    SortResultType attr_sort_result_ = SortResultType::NONE;

    struct Box {
        float score;
        int batch_idx;
        int class_idx;
        int box_idx;
    };

    struct Buffer {
        Box* boxes;
        size_t size;
    };

    class Inputs {
    public:
        virtual ~Inputs() = default;

        enum class Format { first, second };
        virtual Format format() const = 0;

        virtual void prepareParams(const Node& node, const std::string& err_str) = 0;

        virtual const float* boxes(const int batch_idx, const int class_idx, const float* boxes, const int* roisnum) const = 0;
        virtual const float* scores(const int batch_idx, const int class_idx, const float* scores, const int* roisnum) const = 0;
        virtual int num_input_boxes(const int batch_idx, const int* roisnum) const = 0;

        virtual void locate(const Box& box, const int* roisnum, int& input_box_index, int& input_box_offset) = 0;

        size_t num_batches() const { return num_batches_; }
        size_t num_boxes() const { return num_boxes_; }
        size_t num_classes() const { return num_classes_; }

    protected:
        size_t num_batches_ = 0;
        size_t num_boxes_ = 0;
        size_t num_classes_ = 0;
    };
    class InputsFormatFirst;
    class InputsFormatSecond;
    std::unique_ptr<Inputs> inputs_;

    class Workbuffers {
    public:
        Workbuffers(size_t num_batches, size_t num_classes, size_t num_boxes);

        std::vector<Box> boxes; // rois after nms for each class in each image
        std::vector<std::vector<size_t>> num_boxes_per_batch_and_class; // number of rois after nms for each class in each image
        std::vector<int> num_boxes_per_batch;

        Box* thread_workspace_for(int batch_idx, int class_idx) {
            return boxes.data() + batch_idx * num_classes_ * num_boxes_ + class_idx * num_boxes_;
        }

        Buffer flatten_all();
        Buffer flatten_within_batch(int batch_idx);
        Buffer flatten_batches();

    private:
        const size_t num_batches_;
        const size_t num_classes_;
        const size_t num_boxes_;
    };
    std::unique_ptr<Workbuffers> workbuffers_;

    class OutputsLayout {
    public:
        void prepareParams(size_t num_classes, size_t num_boxes, int nms_top_k, int keep_top_k, int background_class);
        int boxes_per_batch() const { return boxes_per_batch_; }
    private:
        int boxes_per_batch_ = 0;
    };
    OutputsLayout output_layout_;

    float intersectionOverUnion(const float* boxesI, const float* boxesJ, const bool normalized);
    void multiclass_nms(const float* boxes, const float* scores, const int* roisnum);

    static void partial_sort_by_score(Buffer buffer, int mid_size);
    static void parallel_sort_by_score(Buffer buffer);
    static void parallel_sort_by_score_across_batch(Buffer buffer);
    static void parallel_sort_by_class(Buffer buffer);
    static void parallel_sort_by_class_across_batch(Buffer buffer);

    void fill_outputs(Buffer rois, const float* input_boxes, const int* input_roisnum);
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

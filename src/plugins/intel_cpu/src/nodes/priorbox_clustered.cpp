// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "priorbox_clustered.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <ie_parallel.hpp>
#include <dnnl_types.h>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

namespace {
/**
 * Implements Prior Box Clustered shape inference algorithm. The output shape is [2,  4 * height * width * number_of_priors].
 * `number_of_priors` is an attribute of the operation. heigh and width are in the the first input parameter.
 *  
 */
class PriorBoxClusteredShapeInfer : public ShapeInferEmptyPads {
public:
    explicit PriorBoxClusteredShapeInfer(size_t number_of_priors) : m_number_of_priors(number_of_priors) {}
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const int* in_data = reinterpret_cast<const int*>(data_dependency.at(0)->GetPtr());
        const int H = in_data[0];
        const int W = in_data[1];
        const auto output = static_cast<size_t>(4 * H * W * m_number_of_priors);
        return {{{2, output}}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return PortMask(0);
    }

private:
    size_t m_number_of_priors = 0;
};

class PriorBoxClusteredShapeInferFactory : public ShapeInferFactory {
public:
    explicit PriorBoxClusteredShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        auto priorBox = ov::as_type_ptr<const ngraph::opset1::PriorBoxClustered>(m_op);
        if (!priorBox) {
            IE_THROW() << "Unexpected op type in PriorBoxClustered shape inference factory: " << m_op->get_type_name();
        }
        const auto& attrs = priorBox->get_attrs();
        auto number_of_priors = attrs.widths.size();
        return std::make_shared<PriorBoxClusteredShapeInfer>(number_of_priors);
    }

private:
    std::shared_ptr<ov::Node> m_op;
};

} // namespace

bool PriorBoxClustered::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto priorBox = std::dynamic_pointer_cast<const ngraph::opset1::PriorBoxClustered>(op);
        if (!priorBox) {
            errorMessage = "Only opset1 PriorBoxClustered operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

PriorBoxClustered::PriorBoxClustered(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, PriorBoxClusteredShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    const auto priorBox = std::dynamic_pointer_cast<const ngraph::opset1::PriorBoxClustered>(op);
    const ngraph::opset1::PriorBoxClustered::Attributes& attrs = priorBox->get_attrs();

    widths = attrs.widths;
    heights = attrs.heights;
    clip = attrs.clip;
    variances = attrs.variances;
    step = attrs.step;
    step_heights = attrs.step_heights;
    step_widths = attrs.step_widths;
    offset = attrs.offset;

    number_of_priors = widths.size();

    if (variances.empty()) {
        variances.push_back(0.1f);
    }
}

bool PriorBoxClustered::needShapeInfer() const {
    auto& memory = getChildEdgeAt(0)->getMemoryPtr();
    if (memory->GetShape().isDynamic()) {
        return true;
    }

    const auto& outputShape = memory->GetShape().getStaticDims();
    const int* in_data = reinterpret_cast<int*>(memory->GetPtr());
    const int h = in_data[0];
    const int w = in_data[1];
    const auto output = static_cast<size_t>(4 * h * w * number_of_priors);

    return outputShape[1] != output;
}

bool PriorBoxClustered::needPrepareParams() const {
    return false;
}

void PriorBoxClustered::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    addSupportedPrimDesc(
            {{LayoutType::ncsp, Precision::I32}, {LayoutType::ncsp, Precision::I32}},
            {{LayoutType::ncsp, Precision::FP32}},
            impl_desc_type::ref_any);
}

void PriorBoxClustered::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void PriorBoxClustered::execute(dnnl::stream strm) {
    const int* in_data = reinterpret_cast<int*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    const int layer_height = in_data[0];
    const int layer_width = in_data[1];

    const int* in_image = reinterpret_cast<int*>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
    int img_height = in_image[0];
    int img_width = in_image[1];

    float step_w = step_widths == 0 ? step : step_widths;
    float step_h = step_heights == 0 ? step : step_heights;
    if (step_w == 0 && step_h == 0) {
        step_w = static_cast<float>(img_width) / layer_width;
        step_h = static_cast<float>(img_height) / layer_height;
    }

    float* dst_data = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    const auto& out_shape = getChildEdgeAt(0)->getMemory().GetShape().getStaticDims();

    size_t var_size = variances.size();
    parallel_for2d(layer_height, layer_width, [&](int64_t h, int64_t w) {
        float center_x = (w + offset) * step_w;
        float center_y = (h + offset) * step_h;

        for (size_t s = 0; s < number_of_priors; ++s) {
            float box_width = widths[s];
            float box_height = heights[s];

            float xmin = (center_x - box_width / 2.0f) / img_width;
            float ymin = (center_y - box_height / 2.0f) / img_height;
            float xmax = (center_x + box_width / 2.0f) / img_width;
            float ymax = (center_y + box_height / 2.0f) / img_height;

            if (clip) {
                xmin = (std::min)((std::max)(xmin, 0.0f), 1.0f);
                ymin = (std::min)((std::max)(ymin, 0.0f), 1.0f);
                xmax = (std::min)((std::max)(xmax, 0.0f), 1.0f);
                ymax = (std::min)((std::max)(ymax, 0.0f), 1.0f);
            }

            const uint64_t idx = h * layer_width * number_of_priors * 4 + w * number_of_priors * 4 + s * 4;
            dst_data[idx + 0] = xmin;
            dst_data[idx + 1] = ymin;
            dst_data[idx + 2] = xmax;
            dst_data[idx + 3] = ymax;

            // At this point we have either:
            // 1. A single variance value (to be repeated 4 times for each prior)
            // 2. 4 variance values
            if (var_size == 1) {
                for (size_t j = 0; j < 4; j++)
                    dst_data[idx + j + out_shape[1]] = variances[0];
            } else {
                for (size_t j = 0; j < var_size; j++)
                    dst_data[idx + j + out_shape[1]] = variances[j];
            }
        }
    });
}

bool PriorBoxClustered::created() const {
    return getType() == Type::PriorBoxClustered;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

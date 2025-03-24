// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "priorbox.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "dnnl_types.h"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset1.hpp"
#include "shape_inference/custom/priorbox.hpp"

namespace ov::intel_cpu::node {
namespace {
float clip_great(float x, float threshold) {
    return x < threshold ? x : threshold;
}

float clip_less(float x, float threshold) {
    return x > threshold ? x : threshold;
}

}  // namespace

bool PriorBox::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto priorBox = ov::as_type_ptr<const ov::opset1::PriorBox>(op);
        if (!priorBox) {
            errorMessage = "Only opset1 PriorBox operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

PriorBox::PriorBox(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PriorBoxShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto priorBox = ov::as_type_ptr<const ov::opset1::PriorBox>(op);
    const ov::opset1::PriorBox::Attributes& attrs = priorBox->get_attrs();
    offset = attrs.offset;
    step = attrs.step;
    min_size = attrs.min_size;
    max_size = attrs.max_size;
    flip = attrs.flip;
    clip = attrs.clip;
    scale_all_sizes = attrs.scale_all_sizes;
    fixed_size = attrs.fixed_size;
    fixed_ratio = attrs.fixed_ratio;
    density = attrs.density;

    bool exist;
    aspect_ratio.push_back(1.0f);
    for (float aspect_ratio_item : attrs.aspect_ratio) {
        exist = false;

        if (std::fabs(aspect_ratio_item) < std::numeric_limits<float>::epsilon()) {
            THROW_CPU_NODE_ERR("has aspect_ratio param can't be equal to zero");
        }

        for (float _aspect_ratio : aspect_ratio) {
            if (std::fabs(aspect_ratio_item - _aspect_ratio) < 1e-6) {
                exist = true;
                break;
            }
        }

        if (exist) {
            continue;
        }

        aspect_ratio.push_back(aspect_ratio_item);
        if (flip) {
            aspect_ratio.push_back(1.0f / aspect_ratio_item);
        }
    }

    number_of_priors = ov::opset1::PriorBox::number_of_priors(attrs);

    if (attrs.variance.size() == 1 || attrs.variance.size() == 4) {
        for (float i : attrs.variance) {
            if (i < 0) {
                THROW_CPU_NODE_ERR("variance must be > 0.");
            }

            variance.push_back(i);
        }
    } else if (attrs.variance.empty()) {
        variance.push_back(0.1f);
    } else {
        THROW_CPU_NODE_ERR("has wrong number of variance values. Not less than 1 and more than 4 variance values.");
    }
}

bool PriorBox::needShapeInfer() const {
    auto memory = getDstMemoryAtPort(0);
    if (memory->getShape().isDynamic()) {
        return true;
    }

    const auto& outputShape = memory->getShape().getStaticDims();
    const int* in_data = memory->getDataAs<int>();
    const int h = in_data[0];
    const int w = in_data[1];
    const auto output = static_cast<size_t>(4) * h * w * number_of_priors;

    return outputShape[1] != output;
}

bool PriorBox::needPrepareParams() const {
    return false;
}

void PriorBox::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::i32}, {LayoutType::ncsp, ov::element::i32}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref_any);
}

void PriorBox::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams()) {
            prepareParams();
        }
        updateLastInputDims();
    }
}

void PriorBox::execute(const dnnl::stream& strm) {
    const int* in_data = getSrcDataAtPortAs<int>(0);
    const int H = in_data[0];
    const int W = in_data[1];

    const int* in_image = getSrcDataAtPortAs<int>(1);
    const int IH = in_image[0];
    const int IW = in_image[1];

    const int OH = 4 * H * W * number_of_priors;
    const int OW = 1;

    auto* dst_data = getDstDataAtPortAs<float>(0);

    float step_ = step;
    auto min_size_ = min_size;
    if (!scale_all_sizes) {
        // mxnet-like PriorBox
        if (step_ == -1) {
            step_ = 1.f * IH / H;
        } else {
            step_ *= IH;
        }
        for (auto& size : min_size_) {
            size *= IH;
        }
    }

    int64_t idx = 0;
    float center_x, center_y, box_width, box_height, step_x, step_y;
    float IWI = 1.0f / static_cast<float>(IW);
    float IHI = 1.0f / static_cast<float>(IH);

    if (step_ == 0) {
        step_x = static_cast<float>(IW) / W;
        step_y = static_cast<float>(IH) / H;
    } else {
        step_x = step_;
        step_y = step_;
    }

    auto calculate_data =
        [&dst_data, &IWI, &IHI, &idx](float center_x, float center_y, float box_width, float box_height, bool clip) {
            if (clip) {
                // order: xmin, ymin, xmax, ymax
                dst_data[idx++] = clip_less((center_x - box_width) * IWI, 0);
                dst_data[idx++] = clip_less((center_y - box_height) * IHI, 0);
                dst_data[idx++] = clip_great((center_x + box_width) * IWI, 1);
                dst_data[idx++] = clip_great((center_y + box_height) * IHI, 1);
            } else {
                dst_data[idx++] = (center_x - box_width) * IWI;
                dst_data[idx++] = (center_y - box_height) * IHI;
                dst_data[idx++] = (center_x + box_width) * IWI;
                dst_data[idx++] = (center_y + box_height) * IHI;
            }
        };

    for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
            if (step_ == 0) {
                center_x = (w + 0.5f) * step_x;
                center_y = (h + 0.5f) * step_y;
            } else {
                center_x = (offset + w) * step_;
                center_y = (offset + h) * step_;
            }

            for (size_t s = 0; s < fixed_size.size(); ++s) {
                auto fixed_size_ = static_cast<size_t>(fixed_size[s]);
                box_width = box_height = fixed_size_ * 0.5f;

                if (!fixed_ratio.empty()) {
                    for (float ar : fixed_ratio) {
                        auto density_ = static_cast<int64_t>(density[s]);
                        auto shift = static_cast<int64_t>(fixed_size[s] / density_);
                        ar = std::sqrt(ar);
                        float box_width_ratio = fixed_size[s] * 0.5f * ar;
                        float box_height_ratio = fixed_size[s] * 0.5f / ar;
                        for (int64_t r = 0; r < density_; ++r) {
                            for (int64_t c = 0; c < density_; ++c) {
                                float center_x_temp = center_x - fixed_size_ / 2.f + shift / 2.f + c * shift;
                                float center_y_temp = center_y - fixed_size_ / 2.f + shift / 2.f + r * shift;
                                calculate_data(center_x_temp, center_y_temp, box_width_ratio, box_height_ratio, true);
                            }
                        }
                    }
                } else {
                    if (!density.empty()) {
                        auto density_ = static_cast<int64_t>(density[s]);
                        auto shift = static_cast<int64_t>(fixed_size[s] / density_);
                        for (int64_t r = 0; r < density_; ++r) {
                            for (int64_t c = 0; c < density_; ++c) {
                                float center_x_temp = center_x - fixed_size_ / 2.f + shift / 2.f + c * shift;
                                float center_y_temp = center_y - fixed_size_ / 2.f + shift / 2.f + r * shift;
                                calculate_data(center_x_temp, center_y_temp, box_width, box_height, true);
                            }
                        }
                    }
                    //  Rest of priors
                    for (float ar : aspect_ratio) {
                        if (fabs(ar - 1.) < 1e-6) {
                            continue;
                        }

                        auto density_ = static_cast<int64_t>(density[s]);
                        auto shift = static_cast<int64_t>(fixed_size[s] / density_);
                        ar = std::sqrt(ar);
                        float box_width_ratio = fixed_size[s] * 0.5f * ar;
                        float box_height_ratio = fixed_size[s] * 0.5f / ar;
                        for (int64_t r = 0; r < density_; ++r) {
                            for (int64_t c = 0; c < density_; ++c) {
                                float center_x_temp = center_x - fixed_size_ / 2.f + shift / 2.f + c * shift;
                                float center_y_temp = center_y - fixed_size_ / 2.f + shift / 2.f + r * shift;
                                calculate_data(center_x_temp, center_y_temp, box_width_ratio, box_height_ratio, true);
                            }
                        }
                    }
                }
            }

            for (size_t ms_idx = 0; ms_idx < min_size_.size(); ms_idx++) {
                box_width = min_size_[ms_idx] * 0.5f;
                box_height = min_size_[ms_idx] * 0.5f;
                calculate_data(center_x, center_y, box_width, box_height, false);

                if (max_size.size() > ms_idx) {
                    box_width = box_height = std::sqrt(min_size_[ms_idx] * max_size[ms_idx]) * 0.5f;
                    calculate_data(center_x, center_y, box_width, box_height, false);
                }

                if (scale_all_sizes || (!scale_all_sizes && (ms_idx == min_size_.size() - 1))) {
                    size_t s_idx = scale_all_sizes ? ms_idx : 0;
                    for (float ar : aspect_ratio) {
                        if (std::fabs(ar - 1.0f) < 1e-6) {
                            continue;
                        }

                        ar = std::sqrt(ar);
                        box_width = min_size_[s_idx] * 0.5f * ar;
                        box_height = min_size_[s_idx] * 0.5f / ar;
                        calculate_data(center_x, center_y, box_width, box_height, false);
                    }
                }
            }
        }
    }

    if (clip) {
        parallel_for((H * W * number_of_priors * 4), [&](size_t i) {
            dst_data[i] = (std::min)((std::max)(dst_data[i], 0.0f), 1.0f);
        });
    }

    uint64_t channel_size = OH * OW;
    if (variance.size() == 1) {
        parallel_for(channel_size, [&](size_t i) {
            dst_data[i + channel_size] = variance[0];
        });
    } else {
        parallel_for(H * W * number_of_priors, [&](size_t i) {
            for (size_t j = 0; j < 4; ++j) {
                dst_data[i * 4 + j + channel_size] = variance[j];
            }
        });
    }
}

bool PriorBox::created() const {
    return getType() == Type::PriorBox;
}

}  // namespace ov::intel_cpu::node

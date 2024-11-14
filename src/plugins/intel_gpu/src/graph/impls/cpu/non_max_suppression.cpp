// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "non_max_suppression_inst.h"
#include "primitive_inst.h"
#include "register.hpp"
#include "cpu_impl_helpers.hpp"
#include "impls/registry/implementation_map.hpp"

#include <vector>
#include <queue>
#include <algorithm>
#include <tuple>

namespace cldnn {
namespace cpu {

using namespace cldnn::cpu;
namespace {
struct result_indices {
    float score;
    int batch_index;
    int class_index;
    int box_index;
};

struct boxInfo {
    float score;
    int idx;
    int suppress_begin_index;
};

std::vector<result_indices> run_nms(
    const vector2D<bounding_box>& boxes,
    const vector3D<float>& scores,
    int num_select_per_class,
    float score_threshold,
    float iou_threshold,
    float soft_nms_sigma,
    bool sort_result_descending
) {
    auto less = [](const boxInfo& l, const boxInfo& r) {
        return l.score < r.score || ((l.score == r.score) && (l.idx > r.idx));
    };
    float scale = 0.0f;
    bool soft_nms = false;
    if (soft_nms_sigma > 0.0f) {
        scale = -0.5f / soft_nms_sigma;
        soft_nms = true;
    }

    auto coeff = [&](float iou) {
        const float weight = std::exp(scale * iou * iou);
        return (iou <= iou_threshold || soft_nms) ? weight : 0.0f;
    };
    std::vector<result_indices> result;

    for (size_t bi = 0; bi < boxes.size(); ++bi) {
        for (size_t ci = 0; ci < scores[bi].size(); ++ci) {
            std::vector<result_indices> fb;

            std::priority_queue<boxInfo, std::vector<boxInfo>, decltype(less)> sorted_boxes(less);
            for (size_t bbi = 0; bbi < boxes[bi].size(); ++bbi) {
                if (scores[bi][ci][bbi] > score_threshold)
                    sorted_boxes.emplace(boxInfo({scores[bi][ci][bbi], static_cast<int>(bbi), 0}));
            }
            fb.reserve(sorted_boxes.size());

            while (static_cast<int>(fb.size()) < num_select_per_class && !sorted_boxes.empty()) {
                boxInfo currBox = sorted_boxes.top();
                float origScore = currBox.score;
                sorted_boxes.pop();

                bool box_is_selected = true;
                for (int idx = static_cast<int>(fb.size()) - 1; idx >= currBox.suppress_begin_index; idx--) {
                    float iou_boxes = iou(boxes[bi][currBox.idx], boxes[bi][fb[idx].box_index]);

                    currBox.score *= coeff(iou_boxes);
                    if (iou_boxes >= iou_threshold && !soft_nms) {
                        box_is_selected = false;
                        break;
                    }
                    if (currBox.score <= score_threshold)
                        break;
                }
                currBox.suppress_begin_index = static_cast<int>(fb.size());
                if (box_is_selected) {
                    if (currBox.score == origScore) {
                        fb.push_back(result_indices{ currBox.score, static_cast<int>(bi), static_cast<int>(ci), currBox.idx });
                        continue;
                    }
                    if (currBox.score > score_threshold) {
                        sorted_boxes.push(currBox);
                    }
                }
            }
            std::move(fb.begin(), fb.end(), std::back_inserter(result));
        }
    }

    if (sort_result_descending) {
        std::sort(result.begin(), result.end(), [](const result_indices& l, const result_indices& r) {
            return (l.score > r.score) || (l.score == r.score && l.batch_index < r.batch_index) ||
                   (l.score == r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                   (l.score == r.score && l.batch_index == r.batch_index && l.class_index == r.class_index &&
                    l.box_index < r.box_index);
        });
    }
    return result;
}

template <typename T>
vector2D<bounding_box> load_boxes_impl(stream& stream, memory::ptr mem, bool center_point) {
    vector2D<bounding_box> result;
    auto lay = mem->get_layout();
    auto batch_size = lay.batch();
    auto boxes_num = lay.feature();
    result.resize(batch_size);

    mem_lock<T, mem_lock_type::read> boxes_lock(mem, stream);
    auto ptr = boxes_lock.data();

    for (int bi = 0; bi < batch_size; ++bi) {
        result[bi].reserve(boxes_num);
        for (int bxi = 0; bxi < boxes_num; ++bxi) {
            int offset = bi * boxes_num * 4 + bxi * 4;
            if (center_point) {
                result[bi].emplace_back(static_cast<float>(ptr[offset + 0]),
                                        static_cast<float>(ptr[offset + 1]),
                                        static_cast<float>(ptr[offset + 2]),
                                        static_cast<float>(ptr[offset + 3]),
                                        bounding_box::center_point_construct_tag());
            } else {
                result[bi].emplace_back(
                    static_cast<float>(ptr[offset + 1]),
                    static_cast<float>(ptr[offset + 0]),
                    static_cast<float>(ptr[offset + 3]),
                    static_cast<float>(ptr[offset + 2]),
                    bounding_box::two_corners_construct_tag());
            }
        }
    }

    return result;
}

vector2D<bounding_box> load_boxes(stream& stream, memory::ptr mem, bool center_point) {
    auto data_type = mem->get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::f16:
        return load_boxes_impl<ov::element_type_traits<data_types::f16>::value_type>(stream, mem, center_point);
    case cldnn::data_types::f32:
        return load_boxes_impl<ov::element_type_traits<data_types::f32>::value_type>(stream, mem, center_point);
    default:
        throw std::runtime_error("Non max suppression - unsupported boxes data type");
    }
}

template <typename T>
vector3D<float> load_scores_impl(stream& stream, memory::ptr mem) {
    auto lay = mem->get_layout();
    auto batch_size = lay.batch();
    auto classes_num = lay.feature();
    auto boxes_num = lay.spatial(1);

    vector3D<float> result(batch_size, vector2D<float>(classes_num));

    mem_lock<T, mem_lock_type::read> lock(mem, stream);
    auto ptr = lock.data();

    for (int bi = 0; bi < batch_size; ++bi) {
        for (int ci = 0; ci < classes_num; ++ci) {
            result[bi][ci].reserve(boxes_num);
            for (int bxi = 0; bxi < boxes_num; ++bxi) {
                auto offset = bi * boxes_num * classes_num + ci * boxes_num + bxi;
                result[bi][ci].emplace_back(static_cast<float>(ptr[offset]));
            }
        }
    }

    return result;
}

vector3D<float> load_scores(stream& stream, memory::ptr mem) {
    auto data_type = mem->get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::f16:
        return load_scores_impl<ov::element_type_traits<data_types::f16>::value_type>(stream, mem);
    case cldnn::data_types::f32:
        return load_scores_impl<ov::element_type_traits<data_types::f32>::value_type>(stream, mem);
    default:
        throw std::runtime_error("Non max suppression - unsupported scores data type");
    }
}

template <typename T, typename MemT>
T load_scalar_impl(stream& stream, memory::ptr mem) {
    mem_lock<MemT, mem_lock_type::read> lock(mem, stream);
    auto ptr = lock.data();

    return static_cast<T>(ptr[0]);
}

template <typename T>
T load_scalar(stream& stream, memory::ptr mem) {
    auto data_type = mem->get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::i32:
        return load_scalar_impl<T, ov::element_type_traits<data_types::i32>::value_type>(stream, mem);
    case cldnn::data_types::f16:
        return load_scalar_impl<T, ov::element_type_traits<data_types::f16>::value_type>(stream, mem);
    case cldnn::data_types::f32:
        return load_scalar_impl<T, ov::element_type_traits<data_types::f32>::value_type>(stream, mem);
    default:
        throw std::runtime_error("Non max suppression - unsupported data type");
    }
}

template <typename T>
void store_result_impl(stream& stream, memory::ptr mem, const std::vector<result_indices>& result, size_t output_size) {
    mem_lock<T, mem_lock_type::write> lock(mem, stream);
    auto ptr = lock.data();
    auto results_size = result.size();

    size_t si = 0;
    for (; si < std::min(output_size, results_size); ++si) {
        auto offset = si * 3;
        ptr[offset + 0] = static_cast<T>(result[si].batch_index);
        ptr[offset + 1] = static_cast<T>(result[si].class_index);
        ptr[offset + 2] = static_cast<T>(result[si].box_index);
    }
    for (; si < output_size; ++si) {
        auto offset = si * 3;
        ptr[offset + 0] = static_cast<T>(-1);
        ptr[offset + 1] = static_cast<T>(-1);
        ptr[offset + 2] = static_cast<T>(-1);
    }
}

void store_result(stream& stream, memory::ptr mem, const std::vector<result_indices>& result, size_t output_size) {
    auto data_type = mem->get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::i32:
        store_result_impl<ov::element_type_traits<data_types::i32>::value_type>(stream, mem, result, output_size);
        break;
    case cldnn::data_types::f16:
        store_result_impl<ov::element_type_traits<data_types::f16>::value_type>(stream, mem, result, output_size);
        break;
    case cldnn::data_types::f32:
        store_result_impl<ov::element_type_traits<data_types::f32>::value_type>(stream, mem, result, output_size);
        break;
    default:
        throw std::runtime_error("Non max suppression - unsupported output data type");
    }
}

void store_first_output(stream& stream, memory::ptr mem, const std::vector<result_indices>& result, size_t output_size) {
    auto data_type = mem->get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::i32:
        store_result_impl<ov::element_type_traits<data_types::i32>::value_type>(stream, mem, result, output_size);
        break;
    case cldnn::data_types::i64:
        store_result_impl<ov::element_type_traits<data_types::i32>::value_type>(stream, mem, result, output_size);
        break;
    default:
        throw std::runtime_error("Non max suppression - unsupported output data type");
    }
}

template <typename T>
void store_second_output_impl(stream& stream, memory::ptr mem, const std::vector<result_indices>& result, size_t output_size) {
    mem_lock<T, mem_lock_type::write> lock(mem, stream);
    auto ptr = lock.data();
    auto results_size = result.size();

    size_t si = 0;
    for (; si < std::min(output_size, results_size); ++si) {
        auto offset = si * 3;
        ptr[offset + 0] = static_cast<T>(result[si].batch_index);
        ptr[offset + 1] = static_cast<T>(result[si].class_index);
        ptr[offset + 2] = static_cast<T>(result[si].score);
    }
    for (; si < output_size; ++si) {
        auto offset = si * 3;
        ptr[offset + 0] = static_cast<T>(-1);
        ptr[offset + 1] = static_cast<T>(-1);
        ptr[offset + 2] = static_cast<T>(-1);
    }
}

void store_second_output(stream& stream, memory::ptr mem, const std::vector<result_indices>& result, size_t output_size) {
    auto data_type = mem->get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::f16:
        store_second_output_impl<ov::element_type_traits<data_types::f16>::value_type>(stream, mem, result, output_size);
        break;
    case cldnn::data_types::f32:
        store_second_output_impl<ov::element_type_traits<data_types::f32>::value_type>(stream, mem, result, output_size);
        break;
    default:
        throw std::runtime_error("Non max suppression - unsupported second output data type");
    }
}

template <typename T>
void store_third_output_impl(stream& stream, const memory::ptr& mem, const std::vector<result_indices>& result) {
    mem_lock<T, mem_lock_type::write> lock(mem, stream);
    auto ptr = lock.data();
    ptr[0] = static_cast<T>(result.size());
}

void store_third_output(stream& stream, memory::ptr mem, const std::vector<result_indices>& result) {
    auto data_type = mem->get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::i32:
        store_third_output_impl<ov::element_type_traits<data_types::i32>::value_type>(stream, mem, result);
        break;
    case cldnn::data_types::i64:
        store_third_output_impl<ov::element_type_traits<data_types::i32>::value_type>(stream, mem, result);
        break;
    default:
        throw std::runtime_error("Non max suppression - unsupported third output data type");
    }
}

void run(non_max_suppression_inst& instance) {
    auto prim = instance.get_typed_desc<non_max_suppression>();
    auto& stream = instance.get_network().get_stream();

    auto boxes = load_boxes(stream, instance.input_boxes_mem(), prim->center_point_box);
    auto scores = load_scores(stream, instance.input_scores_mem());

    int num_select_per_class = 0;
    float iou_threshold = 1.f;
    float score_threshold = 0.f;
    float soft_nms_sigma = 0.f;

    if (instance.has_num_select_per_class()) {
        num_select_per_class = load_scalar<int>(stream, instance.num_select_per_class_mem());
    }

    if (instance.has_iou_threshold()) {
        iou_threshold = load_scalar<float>(stream, instance.iou_threshold_mem());
    }

    if (instance.has_score_threshold()) {
        score_threshold = load_scalar<float>(stream, instance.score_threshold_mem());
    }

    if (instance.has_soft_nms_sigma()) {
        soft_nms_sigma = load_scalar<float>(stream, instance.soft_nms_sigma_mem());
    }

    auto result = run_nms(boxes,
                          scores,
                          num_select_per_class,
                          score_threshold,
                          iou_threshold,
                          soft_nms_sigma,
                          prim->sort_result_descending);

    size_t output_size = instance.get_impl_params()->output_layouts[0].batch();

    // Legacy APIs using mutable inputs for multiple outputs
    if (instance.has_third_output()) {
        store_third_output(stream, instance.third_output_mem(), result);
    }

    if (instance.has_second_output()) {
        store_second_output(stream, instance.second_output_mem(), result, output_size);
        store_first_output(stream, instance.output_memory_ptr(), result, output_size);
        return;
    }

    // New API for mutiple outputs support
    if (instance.outputs_memory_count() == 3)
        store_third_output(stream, instance.output_memory_ptr(2), result);

    if (instance.outputs_memory_count() >= 2) {
        store_second_output(stream, instance.output_memory_ptr(1), result, output_size);
        store_first_output(stream, instance.output_memory_ptr(), result, output_size);
        return;
    }

    store_result(stream, instance.output_memory_ptr(), result, output_size);
}

}  // namespace

struct non_max_suppression_impl : typed_primitive_impl<non_max_suppression> {
    using parent = typed_primitive_impl<non_max_suppression>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::non_max_suppression_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<non_max_suppression_impl>(*this);
    }

    non_max_suppression_impl() : parent("non_max_suppression_impl") {}

    event::ptr execute_impl(const std::vector<event::ptr>& events, typed_primitive_inst<non_max_suppression>& instance) override {
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }

        run(instance);

        if (pass_through_events) {
            if (events.size() > 1) {
                return stream.group_events(events);
            } else if (events.size() == 1) {
                return events[0];
            }
        }

        return stream.create_user_event(true);
    }

    static std::unique_ptr<primitive_impl> create(const non_max_suppression_node&, const kernel_impl_params&) {
        return make_unique<non_max_suppression_impl>();
    }
    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}
};
namespace detail {

attach_non_max_suppression_impl::attach_non_max_suppression_impl() {
    implementation_map<non_max_suppression>::add(impl_types::cpu, non_max_suppression_impl::create, {
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::bfyx),
    });
}

}  // namespace detail

struct non_max_suppression_gather_impl : typed_primitive_impl<non_max_suppression_gather> {
    using parent = typed_primitive_impl<non_max_suppression_gather>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::non_max_suppression_gather_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<non_max_suppression_gather_impl>(*this);
    }

    non_max_suppression_gather_impl() : parent("non_max_suppression_gather_impl") {}

    event::ptr execute_impl(const std::vector<event::ptr>& events, typed_primitive_inst<non_max_suppression_gather>& instance) override {
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }

        if (pass_through_events) {
            if (events.size() > 1) {
                return stream.group_events(events);
            } else if (events.size() == 1) {
                return events[0];
            }
        }

        return stream.create_user_event(true);
    }

    static std::unique_ptr<primitive_impl> create(const non_max_suppression_gather_node&, const kernel_impl_params&) {
        return make_unique<non_max_suppression_gather_impl>();
    }
    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}
};

namespace detail {

attach_non_max_suppression_gather_impl::attach_non_max_suppression_gather_impl() {
    implementation_map<non_max_suppression_gather>::add(impl_types::cpu, non_max_suppression_gather_impl::create, {
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::bfyx),
    });
}

}  // namespace detail

}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::non_max_suppression_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::non_max_suppression)

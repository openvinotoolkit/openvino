// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deferred_weights.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "weights.hpp"

namespace ov::frontend::gguf {
namespace {

using Components = std::array<DeferredWeightTensor, 3>;

std::array<ov::Tensor*, 3> tensor_components(WeightTensors& tensors) {
    return {&tensors.weight, &tensors.scales, &tensors.zero_point};
}

std::array<const ov::Tensor*, 3> tensor_components(const WeightTensors& tensors) {
    return {&tensors.weight, &tensors.scales, &tensors.zero_point};
}

size_t checked_size(const ov::Shape& shape) {
    if (std::find(shape.begin(), shape.end(), 0) != shape.end())
        return 0;
    size_t size = 1;
    for (const auto dim : shape) {
        OPENVINO_ASSERT(dim <= std::numeric_limits<size_t>::max() / size, "Deferred tensor size overflows");
        size *= dim;
    }
    return size;
}

size_t checked_bytes(const ov::element::Type& type, const ov::Shape& shape) {
    OPENVINO_ASSERT(type.is_static() && type != ov::element::string, "Unsupported deferred tensor type ", type);
    checked_size(shape);
    const auto bytes = ov::util::get_memory_size_safe(type, shape);
    OPENVINO_ASSERT(bytes, "Deferred tensor byte size overflows");
    return *bytes;
}

void copy_tensors(const WeightTensors& source, WeightTensors& destination) {
    const auto src = tensor_components(source);
    const auto dst = tensor_components(destination);
    for (size_t i = 0; i < src.size(); ++i) {
        if (!*src[i])
            continue;
        OPENVINO_ASSERT(*dst[i] && src[i]->get_element_type() == dst[i]->get_element_type() &&
                        src[i]->get_shape() == dst[i]->get_shape());
        const auto bytes = src[i]->get_byte_size();
        if (bytes == 0)
            continue;
        if (src[i]->is_continuous()) {
            std::memcpy(dst[i]->data(), src[i]->data(), bytes);
        } else {
            src[i]->copy_to(*dst[i]);
        }
    }
}

WeightTensors row_views(WeightTensors& destination, const Components& components, size_t row_offset) {
    WeightTensors views;
    const auto dst = tensor_components(destination);
    const auto out = tensor_components(views);
    for (size_t i = 0; i < out.size(); ++i) {
        if (components[i].type.is_dynamic())
            continue;
        auto row_shape = components[i].shape;
        row_shape[0] = 1;
        const auto row_bytes = checked_bytes(components[i].type, row_shape);
        auto* ptr = static_cast<uint8_t*>(dst[i]->data());
        *out[i] = ov::Tensor(components[i].type, components[i].shape, ptr ? ptr + row_offset * row_bytes : nullptr);
    }
    return views;
}

class DeferredGroup {
public:
    DeferredGroup(Components components, std::function<void(WeightTensors&)> repack)
        : m_components(std::move(components)),
          m_repack(std::move(repack)) {}

    const Components& components() const {
        return m_components;
    }

    ov::Tensor get(size_t index) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_value.weight) {
            WeightTensors value;
            const auto tensors = tensor_components(value);
            for (size_t i = 0; i < tensors.size(); ++i) {
                if (m_components[i].type.is_static())
                    *tensors[i] = ov::Tensor(m_components[i].type, m_components[i].shape);
            }
            m_repack(value);
            m_value = std::move(value);
            m_repack = {};
            ++m_materializations;
        }
        return *tensor_components(m_value)[index];
    }

    void fill_into(WeightTensors& destination) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_value.weight) {
            copy_tensors(m_value, destination);
        } else {
            m_repack(destination);
        }
    }

    size_t materializations() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_materializations;
    }

private:
    Components m_components;
    std::function<void(WeightTensors&)> m_repack;
    WeightTensors m_value;
    mutable std::mutex m_mutex;
    size_t m_materializations = 0;
};

class DeferredTensor final : public ov::ITensor {
public:
    DeferredTensor(ov::element::Type type, ov::Shape shape, std::function<ov::Tensor()> materialize)
        : m_type(std::move(type)),
          m_shape(std::move(shape)),
          m_initial_shape(m_shape),
          m_materialize(std::move(materialize)) {
        update_metadata();
    }

    DeferredTensor(const std::shared_ptr<DeferredGroup>& group, size_t index)
        : DeferredTensor(group->components()[index].type, group->components()[index].shape, [group, index] {
              return group->get(index);
          }) {
        m_group = group;
        m_component = index;
    }

    void set_shape(ov::Shape shape) override {
        const auto bytes = checked_bytes(m_type, shape);
        const auto size = checked_size(shape);
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        if (shape == m_shape)
            return;
        if (size != m_size) {
            materialize();
            if (bytes > m_value.get_byte_size()) {
                m_value = ov::Tensor(m_type, shape);
                m_group.reset();
            }
        }
        m_shape = std::move(shape);
        update_metadata();
    }

    const ov::element::Type& get_element_type() const override {
        return m_type;
    }

    const ov::Shape& get_shape() const override {
        return m_shape;
    }

    size_t get_size() const override {
        return m_size;
    }

    size_t get_byte_size() const override {
        return m_bytes;
    }

    const ov::Strides& get_strides() const override {
        OPENVINO_ASSERT(m_type.bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less than 8 bit. Tensor type: ",
                        m_type);
        return m_strides;
    }

    void* data() override {
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        materialize();
        return ov::get_tensor_impl(m_value)->data();
    }

    const void* data() const override {
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        materialize();
        return std::as_const(m_value).data();
    }

    void* data_rw() override {
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        materialize();
        return m_value.data();
    }

    void* data(const ov::element::Type& type) override {
        check_type(type);
        return data();
    }

    const void* data(const ov::element::Type& type) const override {
        check_type(type);
        return data();
    }

    void* data_rw(const ov::element::Type& type) override {
        check_type(type);
        return data_rw();
    }

    bool materialized() const {
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        return static_cast<bool>(m_value) || (m_group && m_group->materializations() != 0);
    }

    size_t materializations() const {
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        return m_group ? m_group->materializations() : m_materializations;
    }

    std::shared_ptr<DeferredGroup> group(size_t index) const {
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        return m_group && index == m_component && m_shape == m_initial_shape ? m_group : nullptr;
    }

private:
    void materialize() const {
        if (m_value)
            return;
        OPENVINO_ASSERT(!m_initializing, "Recursive deferred tensor materialization");
        m_initializing = true;
        try {
            auto value = m_materialize();
            OPENVINO_ASSERT(value && ov::get_tensor_impl(value)._ptr.get() != this &&
                                value.get_element_type() == m_type && value.get_shape() == m_initial_shape,
                            "Deferred tensor materialization changed its type or shape");
            OPENVINO_ASSERT(value.is_continuous(), "Deferred tensor materialization must be continuous");
            static_cast<void>(std::as_const(value).data());
            m_value = std::move(value);
            m_materialize = {};
            ++m_materializations;
            m_initializing = false;
        } catch (...) {
            m_initializing = false;
            throw;
        }
    }

    void check_type(const ov::element::Type& type) const {
        OPENVINO_ASSERT(type.is_dynamic() || (type != ov::element::string && type.bitwidth() == m_type.bitwidth() &&
                                              type.is_real() == m_type.is_real()),
                        "Tensor data with element type ",
                        m_type,
                        ", is not representable as pointer to ",
                        type);
    }

    void update_metadata() {
        m_size = checked_size(m_shape);
        m_bytes = checked_bytes(m_type, m_shape);
        m_strides.clear();
        if (m_type.bitwidth() >= 8 && !m_shape.empty()) {
            m_strides.resize(m_shape.size());
            m_strides.back() = m_shape.back() == 0 ? 0 : m_type.size();
            for (size_t i = m_shape.size() - 1; i > 0; --i) {
                OPENVINO_ASSERT(m_shape[i] == 0 || m_strides[i] <= std::numeric_limits<size_t>::max() / m_shape[i],
                                "Deferred tensor stride overflows");
                m_strides[i - 1] = m_strides[i] * m_shape[i];
            }
        }
    }

    ov::element::Type m_type;
    ov::Shape m_shape;
    ov::Shape m_initial_shape;
    ov::Strides m_strides;
    size_t m_size = 0;
    size_t m_bytes = 0;
    mutable std::function<ov::Tensor()> m_materialize;
    mutable ov::Tensor m_value;
    mutable size_t m_materializations = 0;
    mutable bool m_initializing = false;
    mutable std::recursive_mutex m_mutex;
    std::shared_ptr<DeferredGroup> m_group;
    size_t m_component = 0;
};

std::shared_ptr<DeferredTensor> deferred_impl(const ov::Tensor& tensor) {
    return tensor ? std::dynamic_pointer_cast<DeferredTensor>(ov::get_tensor_impl(tensor)._ptr) : nullptr;
}

std::shared_ptr<DeferredGroup> deferred_group(const WeightTensors& tensors) {
    const auto components = tensor_components(tensors);
    std::shared_ptr<DeferredGroup> group;
    for (size_t i = 0; i < components.size(); ++i) {
        if (!*components[i]) {
            if (group && group->components()[i].type.is_static())
                return nullptr;
            continue;
        }
        const auto impl = deferred_impl(*components[i]);
        const auto candidate = impl ? impl->group(i) : nullptr;
        if (!candidate || (group && group != candidate))
            return nullptr;
        group = candidate;
    }
    return group;
}

WeightTensors group_tensors(const std::shared_ptr<DeferredGroup>& group) {
    WeightTensors result;
    const auto tensors = tensor_components(result);
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (group->components()[i].type.is_static())
            *tensors[i] = ov::make_tensor(std::make_shared<DeferredTensor>(group, i));
    }
    return result;
}

}  // namespace

bool is_deferred(const ov::Tensor& tensor) {
    return static_cast<bool>(deferred_impl(tensor));
}

bool is_deferred(const WeightTensors& tensors) {
    return is_deferred(tensors.weight) || is_deferred(tensors.scales) || is_deferred(tensors.zero_point);
}

bool is_materialized(const ov::Tensor& tensor) {
    const auto impl = deferred_impl(tensor);
    return impl ? impl->materialized() : static_cast<bool>(tensor);
}

size_t get_materialization_count(const ov::Tensor& tensor) {
    const auto impl = deferred_impl(tensor);
    return impl ? impl->materializations() : 0;
}

ov::Tensor defer_tensor(const ov::element::Type& type,
                        const ov::Shape& shape,
                        std::function<ov::Tensor()> materialize) {
    OPENVINO_ASSERT(materialize, "Deferred tensor has no materializer");
    return ov::make_tensor(std::make_shared<DeferredTensor>(type, shape, std::move(materialize)));
}

WeightTensors defer_weight_tensors(const Components& components, std::function<void(WeightTensors&)> repack) {
    OPENVINO_ASSERT(components[0].type.is_static() && repack, "Deferred weight group has no weights or repacker");
    return group_tensors(std::make_shared<DeferredGroup>(components, std::move(repack)));
}

std::optional<WeightTensors> concat_deferred_rows(const WeightTensors& lhs, const WeightTensors& rhs) {
    if (!lhs.weight || !rhs.weight || (!is_deferred(lhs) && !is_deferred(rhs)))
        return std::nullopt;
    const auto left = tensor_components(lhs);
    const auto right = tensor_components(rhs);
    Components lhs_info;
    Components rhs_info;
    Components combined;
    for (size_t i = 0; i < combined.size(); ++i) {
        if (static_cast<bool>(*left[i]) != static_cast<bool>(*right[i]))
            return std::nullopt;
        if (!*left[i])
            continue;
        const auto& ls = left[i]->get_shape();
        const auto& rs = right[i]->get_shape();
        const auto& type = left[i]->get_element_type();
        if (type != right[i]->get_element_type() || ls.empty() || ls.size() != rs.size() ||
            !std::equal(ls.begin() + 1, ls.end(), rs.begin() + 1) || ls[0] != left[0]->get_shape()[0] ||
            rs[0] != right[0]->get_shape()[0] || rs[0] > std::numeric_limits<size_t>::max() - ls[0])
            return std::nullopt;
        auto row_shape = ls;
        row_shape[0] = 1;
        const auto row_size = checked_size(row_shape);
        if (row_size > std::numeric_limits<size_t>::max() / type.bitwidth() || (row_size * type.bitwidth()) % 8 != 0)
            return std::nullopt;
        lhs_info[i] = {type, ls};
        rhs_info[i] = {type, rs};
        combined[i] = {type, ls};
        combined[i].shape[0] += rs[0];
        checked_bytes(type, combined[i].shape);
    }
    const auto left_group = deferred_group(lhs);
    const auto right_group = deferred_group(rhs);
    const auto left_copy = left_group ? WeightTensors{} : lhs;
    const auto right_copy = right_group ? WeightTensors{} : rhs;
    return defer_weight_tensors(
        combined,
        [left_group, right_group, left_copy, right_copy, lhs_info, rhs_info, rows = lhs.weight.get_shape()[0]](
            WeightTensors& destination) {
            auto left_views = row_views(destination, lhs_info, 0);
            auto right_views = row_views(destination, rhs_info, rows);
            if (left_group)
                left_group->fill_into(left_views);
            else
                copy_tensors(left_copy, left_views);
            if (right_group)
                right_group->fill_into(right_views);
            else
                copy_tensors(right_copy, right_views);
        });
}

}  // namespace ov::frontend::gguf

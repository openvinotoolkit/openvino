// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lazy_tensor.hpp"

#include <tuple>
#include <type_traits>
#include <variant>

#include "logging.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "util.hpp"

using ov::npuw::weights::LazyTensor;

namespace ov {
namespace npuw {
namespace weights {
namespace op {
Const::Const(const std::shared_ptr<ov::op::v0::Constant>& n) : m_node(n) {
    m_cached_type = m_node->get_element_type();
    m_cached_shape = m_node->get_shape();
    m_cached_ptr = m_node->get_data_ptr();
    m_byte_size = m_node->get_byte_size();

    auto rt_info = m_node->get_rt_info();
    auto weightless_cache_attr = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
    if (weightless_cache_attr != rt_info.end()) {
        m_offset = weightless_cache_attr->second.as<ov::WeightlessCacheAttribute>().bin_offset;
    } else {
        // See the comment in serialize() for more details
        LOG_WARN("Some pattern introduced a new Constant node not present in the original weights file. We need to "
                 "keep it in case export occurs. This will increase memory consumption.");
        m_copied_if_not_in_model = ov::npuw::util::copy_tensor_from_const(m_node);
    }
}

std::size_t Const::hash() const {
    std::size_t seed = std::hash<const void*>()(m_cached_ptr) + 0x9e3779b9;
    seed ^= m_cached_type.hash() + 0x9e3779b9;
    for (const auto& dim : m_cached_shape) {
        seed ^= std::hash<std::size_t>()(dim) + 0x9e3779b9;
    }
    return seed;
}

bool Const::operator==(const Const& other) const {
    return (m_cached_type == other.m_cached_type && m_cached_shape == other.m_cached_shape &&
            m_cached_ptr == other.m_cached_ptr);
}

ov::Tensor Const::eval() const {
    if (m_node) {
        return ov::npuw::util::copy_tensor_from_const(m_node);
    }

    // Weightless import case. Mmmap CPU weight on demand to avoid allocating all weights at once.
    if (!m_weights_path.empty()) {
        NPUW_ASSERT(!m_read_from_bin &&
                    "Trying to read weight from weights file, but the weight has been deserialized already!");
        auto mapped_memory = ov::load_mmap_object(m_weights_path);
        m_mmaped_weights =
            std::make_shared<ov::npuw::s11n::Weights>(mapped_memory->data(), mapped_memory->size(), mapped_memory);
        return ov::Tensor(m_cached_type, m_cached_shape, m_mmaped_weights->get_ptr(m_offset));
    }

    NPUW_ASSERT(m_read_from_bin && "Underlying data should have been read first! Or the tensor is already detached.");
    return m_read_from_bin;
}

LazyTensor::Meta Const::eval_meta() const {
    if (m_node) {
        return {m_node->get_shape(), m_node->get_element_type()};
    }

    // Weightless import case
    if (!m_weights_path.empty()) {
        return {m_cached_shape, m_cached_type};
    }

    NPUW_ASSERT(m_read_from_bin && "Underlying data should have been read first!");
    return {m_read_from_bin.get_shape(), m_read_from_bin.get_element_type()};
}

void Const::read_weight(const ov::npuw::s11n::WeightsContext& ctx) {
    NPUW_ASSERT(!m_node &&
                "LazyTensor can only read weight when it's being deserialized and not created from a Constant!");
    if (m_read_from_bin) {
        // already deserialized, see the comment in serialize() for more details
        return;
    }
    if (ctx.weights) {
        if (ctx.bf16_consts.find({m_offset, m_byte_size}) != ctx.bf16_consts.end()) {
            NPUW_ASSERT(m_cached_type == ov::element::f16);
            // Read original bf16 weight
            auto bf16_tensor = ov::Tensor(ov::element::bf16, m_cached_shape);
            NPUW_ASSERT(bf16_tensor.get_byte_size() == m_byte_size);
            std::memcpy(bf16_tensor.data(), ctx.weights->get_ptr(m_offset), m_byte_size);

            m_read_from_bin = ov::Tensor(m_cached_type, m_cached_shape);
            NPUW_ASSERT(bf16_tensor.get_size() == m_read_from_bin.get_size());
            // Transform bf16 to f16 tensor
            using dst_type = typename element_type_traits<ov::element::Type_t::f16>::value_type;
            auto src_data = bf16_tensor.data<ov::bfloat16>();
            auto dst_data = m_read_from_bin.data<dst_type>();
            ov::reference::convert_from_bf16_to_f16_with_clamp(src_data, dst_data, m_read_from_bin.get_size());
        } else {
            // Each LazyTensor will mmap the whole weights file on demand (in eval()).
            // It doesn't introduce extra allocation, however it allows to gradually 1 by 1
            // read mmaped CPU weights and allocate them on device without loading all the weights first.
            // Thus the memory consumption during import is greatly reduced but at the slight cost of performance.
            NPUW_ASSERT(!ctx.weights_path.empty());
            // Just save weights_path for the eval() to call the actual mmap.
            m_weights_path = ctx.weights_path;
        }
    } else {
        auto it = ctx.consts_cache.find({m_offset, m_byte_size});
        NPUW_ASSERT(it != ctx.consts_cache.end() && "Couldn't find Constant in cache!");
        m_read_from_bin = ov::npuw::util::tensor_from_const(it->second);
        NPUW_ASSERT(m_read_from_bin.get_byte_size() == m_byte_size && m_read_from_bin.get_shape() == m_cached_shape &&
                    m_read_from_bin.get_element_type() == m_cached_type);
    }
}

void Const::detach() {
    m_node.reset();
    m_read_from_bin = ov::Tensor();
    m_mmaped_weights.reset();
}

void Const::serialize(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    write(stream, m_cached_type.to_string());
    write(stream, m_cached_shape);
    write(stream, m_offset);
    write(stream, m_byte_size);

    // FIXME: handle a special case:
    // 1) We added a Constant to the model before compilation (e.g. int RoPE patterns)
    // 2) This Constant became a parameter during folding
    // 3) Thus it became a LazyTensor, but there is no original data in weights file,
    // so it will fail in read_weight() during weightless deserialization.
    // In this case we need to include Constant's data into the blob.
    if (m_copied_if_not_in_model) {
        LOG_WARN("Some pattern introduced a new Constant node not present in the original weights file. This will "
                 "increase the blob size.");
        write(stream, true);
        write(stream, m_copied_if_not_in_model);
        // detach the tensor
        m_copied_if_not_in_model = ov::Tensor();
    } else {
        write(stream, false);
    }
}

Const Const::deserialize(std::istream& stream) {
    using namespace ov::npuw::s11n;
    Const c;
    std::string type_str;
    read(stream, type_str);
    c.m_cached_type = ov::element::Type(type_str);
    read(stream, c.m_cached_shape);
    read(stream, c.m_offset);
    read(stream, c.m_byte_size);

    bool contains_weight = false;
    read(stream, contains_weight);
    if (contains_weight) {
        read(stream, c.m_read_from_bin);
    }

    return c;
}

std::size_t Concat::hash() const {
    std::size_t seed = std::hash<std::size_t>()(axis) + 0x9e3779b9;
    for (auto& lt : tensors) {
        seed ^= lt.get_hash() + 0x9e3779b9;
    }
    return seed;
}

bool Concat::operator==(const Concat& other) const {
    return (axis == other.axis && tensors == other.tensors);
}

ov::Tensor Concat::eval() const {
    std::vector<ov::Tensor> to_concat;
    for (const auto& lt : tensors) {
        to_concat.push_back(lt.eval());
    }
    return ov::npuw::util::concat(to_concat, axis);
}

LazyTensor::Meta Concat::eval_meta() const {
    auto meta = tensors[0].eval_meta();
    ov::Shape shape = meta.shape;
    for (std::size_t i = 1; i < tensors.size(); ++i) {
        shape[axis] += tensors[i].eval_meta().shape[axis];
    }
    return {shape, meta.type};
}

void Concat::read_weight(const ov::npuw::s11n::WeightsContext& ctx) {
    for (auto& lt : tensors) {
        lt.read_weight(ctx);
    }
}

void Concat::detach() {
    for (auto&& lt : tensors) {
        lt.detach();
    }
}

void Concat::serialize(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    write(stream, axis);
    write(stream, tensors);
}

Concat Concat::deserialize(std::istream& stream) {
    using namespace ov::npuw::s11n;
    Concat c;
    read(stream, c.axis);
    read(stream, c.tensors);
    return c;
}

std::size_t Unpack::hash() const {
    std::size_t seed = w.get_hash() + 0x9e3779b9;
    seed ^= z.get_hash() + 0x9e3779b9;
    seed ^= s.get_hash() + 0x9e3779b9;
    seed ^= type.hash() + 0x9e3779b9;
    for (const auto& dim : shape) {
        seed ^= std::hash<std::size_t>()(dim) + 0x9e3779b9;
    }
    return seed;
}

bool Unpack::operator==(const Unpack& other) const {
    return (type == other.type && shape == other.shape && w == other.w && z == other.z && s == other.s);
}

ov::Tensor Unpack::eval() const {
    const auto& gti = ov::get_tensor_impl;
    const auto& tw = w.eval();
    const auto& tz = z.eval();
    const auto& ts = s.eval();
    NPUW_ASSERT(tw);
    ov::Tensor dst(type, shape);
    if (tw && tz && ts) {
        ov::npuw::util::unpack(gti(tw), gti(tz), gti(ts), gti(dst));
    } else if (tw && ts) {
        ov::npuw::util::unpack(gti(tw), gti(ts), gti(dst));
    } else {
        NPUW_ASSERT(false && "Unsupported combination");
    }
    return dst;
}

LazyTensor::Meta Unpack::eval_meta() const {
    return {shape, type};
}

void Unpack::read_weight(const ov::npuw::s11n::WeightsContext& ctx) {
    w.read_weight(ctx);
    if (z) {  // could be empty
        z.read_weight(ctx);
    }
    s.read_weight(ctx);
}

void Unpack::detach() {
    w.detach();
    z.detach();
    s.detach();
}

void Unpack::serialize(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    write(stream, type.to_string());
    write(stream, shape);
    write(stream, w);
    write(stream, z);
    write(stream, s);
}

Unpack Unpack::deserialize(std::istream& stream) {
    using namespace ov::npuw::s11n;
    Unpack u;
    std::string type_str;
    read(stream, type_str);
    u.type = ov::element::Type(type_str);
    read(stream, u.shape);
    read(stream, u.w);
    read(stream, u.z);
    read(stream, u.s);
    return u;
}

std::size_t Permute::hash() const {
    std::size_t seed = tensor.get_hash() + 0x9e3779b9;
    for (const auto& axis : axes) {
        seed ^= std::hash<std::size_t>()(axis) + 0x9e3779b9;
    }
    return seed;
}

bool Permute::operator==(const Permute& other) const {
    return (axes == other.axes && tensor == other.tensor);
}

ov::Tensor Permute::eval() const {
    return ov::npuw::util::permute(tensor.eval(), axes);
}

LazyTensor::Meta Permute::eval_meta() const {
    auto meta = tensor.eval_meta();
    auto shape = meta.shape;
    ov::Shape new_shape;
    std::transform(axes.begin(), axes.end(), std::back_inserter(new_shape), [&](std::size_t i) {
        return shape[i];
    });
    return {new_shape, meta.type};
}

void Permute::read_weight(const ov::npuw::s11n::WeightsContext& ctx) {
    tensor.read_weight(ctx);
}

void Permute::detach() {
    tensor.detach();
}

void Permute::serialize(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    write(stream, axes);
    write(stream, tensor);
}

Permute Permute::deserialize(std::istream& stream) {
    using namespace ov::npuw::s11n;
    Permute p;
    read(stream, p.axes);
    read(stream, p.tensor);
    return p;
}

std::size_t Convert::hash() const {
    std::size_t seed = type.hash() + 0x9e3779b9;
    seed ^= tensor.get_hash() + 0x9e3779b9;
    return seed;
}

bool Convert::operator==(const Convert& other) const {
    return (type == other.type && tensor == other.tensor);
}

ov::Tensor Convert::eval() const {
    NPUW_ASSERT(ov::element::f16 == type);
    return ov::npuw::util::to_f16(tensor.eval());
}

LazyTensor::Meta Convert::eval_meta() const {
    return {tensor.eval_meta().shape, type};
}

void Convert::read_weight(const ov::npuw::s11n::WeightsContext& ctx) {
    tensor.read_weight(ctx);
}

void Convert::detach() {
    tensor.detach();
}

void Convert::serialize(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    write(stream, type.to_string());
    write(stream, tensor);
}

Convert Convert::deserialize(std::istream& stream) {
    using namespace ov::npuw::s11n;
    Convert c;
    std::string type_str;
    read(stream, type_str);
    c.type = ov::element::Type(type_str);
    read(stream, c.tensor);
    return c;
}

std::size_t Gather::hash() const {
    std::size_t seed = w.get_hash() + 0x9e3779b9;
    seed ^= t.get_element_type().hash() + 0x9e3779b9;
    for (const auto& dim : t.get_shape()) {
        seed ^= std::hash<std::size_t>()(dim) + 0x9e3779b9;
    }
    auto ttype = t.get_element_type();
    NPUW_ASSERT(ttype == ov::element::f8e4m3 || ttype == ov::element::f8e5m2 || ttype == ov::element::f8e8m0);
    std::vector<uint8_t> t_data(t.get_size());
    std::memcpy(t_data.data(), static_cast<uint8_t*>(t.data()), t.get_size());
    seed ^= t_data.size();
    for (const auto& el : t_data) {
        seed ^= std::hash<uint8_t>()(el) + 0x9e3779b9;
    }
    seed ^= dst_type.hash() + 0x9e3779b9;
    for (const auto& dim : dst_shape) {
        seed ^= std::hash<std::size_t>()(dim) + 0x9e3779b9;
    }
    return seed;
}

bool Gather::operator==(const Gather& other) const {
    auto ttype = t.get_element_type();
    NPUW_ASSERT(ttype == ov::element::f8e4m3 || ttype == ov::element::f8e5m2 || ttype == ov::element::f8e8m0);
    std::vector<uint8_t> t_data(t.get_size());
    std::memcpy(t_data.data(), static_cast<uint8_t*>(t.data()), t.get_size());

    auto ttype_other = other.t.get_element_type();
    NPUW_ASSERT(ttype_other == ov::element::f8e4m3 || ttype_other == ov::element::f8e5m2 ||
                ttype_other == ov::element::f8e8m0);
    std::vector<uint8_t> t_other_data(other.t.get_size());
    std::memcpy(t_other_data.data(), static_cast<uint8_t*>(other.t.data()), other.t.get_size());

    return (w == other.w && t.get_element_type() == other.t.get_element_type() &&
            t.get_shape() == other.t.get_shape() && t_data == t_other_data && dst_type == other.dst_type &&
            dst_shape == other.dst_shape);
}

ov::Tensor Gather::eval() const {
    auto ttype = t.get_element_type();
    NPUW_ASSERT(ttype == ov::element::f8e4m3 || ttype == ov::element::f8e5m2 || ttype == ov::element::f8e8m0);
    ov::Tensor dst(dst_type, dst_shape);
    const auto& gti = ov::get_tensor_impl;
    ov::npuw::util::gather_cb4(gti(t), gti(w.eval()), gti(dst));
    return dst;
}

LazyTensor::Meta Gather::eval_meta() const {
    return {dst_shape, dst_type};
}

void Gather::read_weight(const ov::npuw::s11n::WeightsContext& ctx) {
    w.read_weight(ctx);
}

void Gather::detach() {
    w.detach();
}

void Gather::serialize(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    write(stream, dst_type.to_string());
    write(stream, dst_shape);
    write(stream, w);
    write(stream, t);
}

Gather Gather::deserialize(std::istream& stream) {
    using namespace ov::npuw::s11n;
    Gather g;
    std::string type_str;
    read(stream, type_str);
    g.dst_type = ov::element::Type(type_str);
    read(stream, g.dst_shape);
    read(stream, g.w);
    read(stream, g.t);
    return g;
}
}  // namespace op

enum class TransformType : int { CONST = 0, CONCAT, UNPACK, PERMUTE, CONVERT, GATHER };

struct LazyTensorImpl {
    LazyTensorImpl() = default;
    explicit LazyTensorImpl(LazyTensor::Transform&& t);
    bool operator==(const LazyTensorImpl& other) const;

    ov::Tensor eval() const;
    LazyTensor::Meta eval_meta() const;
    std::size_t get_hash() const;
    void get_transformations(std::vector<LazyTensor::Transform>& vec) const;

    void detach();

    void serialize(std::ostream& stream) const;
    static std::shared_ptr<LazyTensorImpl> deserialize(std::istream& stream);
    void read_weight(const ov::npuw::s11n::WeightsContext& ctx);

    LazyTensor::Transform m_transform;
    std::size_t m_hash = 0;
};

}  // namespace weights
}  // namespace npuw
}  // namespace ov

using namespace ov::npuw::weights::op;
using ov::npuw::weights::LazyTensorImpl;

// std::visit helper
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

LazyTensorImpl::LazyTensorImpl(LazyTensor::Transform&& t)
    : m_transform(std::move(t)),
      m_hash(std::visit(overloaded{[](const auto& op) {
                            return op.hash();
                        }},
                        m_transform)) {}

bool LazyTensorImpl::operator==(const LazyTensorImpl& other) const {
    return m_hash == other.m_hash && m_transform == other.m_transform;
}

ov::Tensor LazyTensorImpl::eval() const {
    /* FIXME:
    Consider case:
        model1: concat->permute->f16
        model2: permute->f16
    Due to different history of transformation new tensors will be allocated for model2.
    However, we could avoid it by introducing a proper slicing on top of known axes and
    some kind of indicator that the only difference is concat and we should look for an existing ov::Tensor.
    Perhaps it should be done after model compilation and not handled here.
    */
    return std::visit(overloaded{[](const auto& op) {
                          return op.eval();
                      }},
                      m_transform);
}

LazyTensor::Meta LazyTensorImpl::eval_meta() const {
    return std::visit(overloaded{[](const auto& op) {
                          return op.eval_meta();
                      }},
                      m_transform);
}

void LazyTensorImpl::read_weight(const ov::npuw::s11n::WeightsContext& ctx) {
    std::visit(overloaded{[&ctx](auto& op) {
                   return op.read_weight(ctx);
               }},
               m_transform);
}

std::size_t LazyTensorImpl::get_hash() const {
    return m_hash;
}

void LazyTensorImpl::get_transformations(std::vector<LazyTensor::Transform>& vec) const {
    vec.push_back(m_transform);
    std::visit(overloaded{
                   [&vec](const op::Concat& op) {
                       for (const auto& lt : op.tensors) {
                           auto next_tr = lt.get_transformations();
                           vec.insert(vec.end(), next_tr.begin(), next_tr.end());
                       }
                   },
                   [](const op::Const& op) {
                       // do nothing
                   },
                   [&vec](const op::Convert& op) {
                       auto next_tr = op.tensor.get_transformations();
                       vec.insert(vec.end(), next_tr.begin(), next_tr.end());
                   },
                   [&vec](const op::Permute& op) {
                       auto next_tr = op.tensor.get_transformations();
                       vec.insert(vec.end(), next_tr.begin(), next_tr.end());
                   },
                   [&vec](const op::Unpack& op) {
                       auto next_w_tr = op.w.get_transformations();
                       vec.insert(vec.end(), next_w_tr.begin(), next_w_tr.end());
                       auto next_z_tr = op.z.get_transformations();
                       vec.insert(vec.end(), next_z_tr.begin(), next_z_tr.end());
                       auto next_s_tr = op.s.get_transformations();
                       vec.insert(vec.end(), next_s_tr.begin(), next_s_tr.end());
                   },
                   [&vec](const op::Gather& op) {
                       auto next_tr = op.w.get_transformations();
                       vec.insert(vec.end(), next_tr.begin(), next_tr.end());
                   },
               },
               m_transform);
}

void LazyTensorImpl::detach() {
    std::visit(overloaded{[](auto& op) {
                   op.detach();
               }},
               m_transform);
}

void LazyTensorImpl::serialize(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    write(stream, m_hash);
    // FIXME: create proper op identificators instead of int
    std::visit(overloaded{
                   [&stream](const op::Concat& op) {
                       write(stream, static_cast<int>(TransformType::CONCAT));
                       op.serialize(stream);
                   },
                   [&stream](const op::Const& op) {
                       write(stream, static_cast<int>(TransformType::CONST));
                       op.serialize(stream);
                   },
                   [&stream](const op::Convert& op) {
                       write(stream, static_cast<int>(TransformType::CONVERT));
                       op.serialize(stream);
                   },
                   [&stream](const op::Permute& op) {
                       write(stream, static_cast<int>(TransformType::PERMUTE));
                       op.serialize(stream);
                   },
                   [&stream](const op::Unpack& op) {
                       write(stream, static_cast<int>(TransformType::UNPACK));
                       op.serialize(stream);
                   },
                   [&stream](const op::Gather& op) {
                       write(stream, static_cast<int>(TransformType::GATHER));
                       op.serialize(stream);
                   },
               },
               m_transform);
}

std::shared_ptr<LazyTensorImpl> LazyTensorImpl::deserialize(std::istream& stream) {
    using namespace ov::npuw::s11n;
    auto lt_impl = std::make_shared<LazyTensorImpl>();
    read(stream, lt_impl->m_hash);
    int op_type;
    read(stream, op_type);
    switch (TransformType(op_type)) {
    case TransformType::CONCAT:
        lt_impl->m_transform = op::Concat::deserialize(stream);
        break;
    case TransformType::CONST:
        lt_impl->m_transform = op::Const::deserialize(stream);
        break;
    case TransformType::CONVERT:
        lt_impl->m_transform = op::Convert::deserialize(stream);
        break;
    case TransformType::PERMUTE:
        lt_impl->m_transform = op::Permute::deserialize(stream);
        break;
    case TransformType::UNPACK:
        lt_impl->m_transform = op::Unpack::deserialize(stream);
        break;
    case TransformType::GATHER:
        lt_impl->m_transform = op::Gather::deserialize(stream);
        break;
    default:
        NPUW_ASSERT(false && "Unsupported type");
        break;
    }
    return lt_impl;
}

LazyTensor::LazyTensor(const std::shared_ptr<ov::op::v0::Constant>& const_ptr)
    : m_impl(std::make_shared<LazyTensorImpl>(op::Const(const_ptr))) {}
LazyTensor::LazyTensor(const std::vector<LazyTensor>& to_concat, const std::size_t axis)
    : m_impl(std::make_shared<LazyTensorImpl>(op::Concat(to_concat, axis))) {}
LazyTensor::LazyTensor(const LazyTensor& cw,
                       const LazyTensor& cz,
                       const LazyTensor& cs,
                       const ov::element::Type& type,
                       const ov::Shape& shape)
    : m_impl(std::make_shared<LazyTensorImpl>(op::Unpack(cw, cz, cs, type, shape))) {}
LazyTensor::LazyTensor(const LazyTensor& cw,
                       const ov::Tensor& t,
                       const ov::element::Type& dst_type,
                       const ov::Shape& dst_shape)
    : m_impl(std::make_shared<LazyTensorImpl>(op::Gather(cw, t, dst_type, dst_shape))) {}

LazyTensor LazyTensor::permute(const std::vector<std::size_t>& axes) {
    LazyTensor new_lt;
    new_lt.m_impl = std::make_shared<LazyTensorImpl>(op::Permute(*this, axes));
    return new_lt;
}

LazyTensor LazyTensor::convert(const ov::element::Type& type) {
    LazyTensor new_lt;
    new_lt.m_impl = std::make_shared<LazyTensorImpl>(op::Convert(*this, type));
    return new_lt;
}

bool LazyTensor::operator==(const LazyTensor& other) const {
    if (!m_impl && !other.m_impl) {
        return true;
    }
    if ((!m_impl && other.m_impl) || (m_impl && !other.m_impl)) {
        return false;
    }
    return *m_impl.get() == *other.m_impl.get();
}

bool LazyTensor::operator!=(const LazyTensor& other) const {
    return !(*this == other);
}

ov::Tensor LazyTensor::eval() const {
    if (!m_impl) {
        return ov::Tensor();
    }
    return m_impl->eval();
}

LazyTensor::Meta LazyTensor::eval_meta() const {
    if (!m_impl) {
        return {};
    }
    return m_impl->eval_meta();
}

void LazyTensor::read_weight(const ov::npuw::s11n::WeightsContext& ctx) {
    NPUW_ASSERT(m_impl && "Trying to read weights into uninitialized tensor!");
    m_impl->read_weight(ctx);
}

LazyTensor::operator bool() const {
    return m_impl != nullptr;
}

std::size_t LazyTensor::get_hash() const {
    if (!m_impl) {
        return 0;
    }
    return m_impl->get_hash();
}

std::vector<LazyTensor::Transform> LazyTensor::get_transformations() const {
    if (!m_impl) {
        return {};
    }

    std::vector<LazyTensor::Transform> transformations;
    m_impl->get_transformations(transformations);
    return transformations;
}

void LazyTensor::detach() {
    if (m_impl) {
        m_impl->detach();
    }
}

void LazyTensor::serialize(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    if (!m_impl) {
        write(stream, false);
        return;
    }
    write(stream, true);
    m_impl->serialize(stream);
}

LazyTensor LazyTensor::deserialize(std::istream& stream) {
    using namespace ov::npuw::s11n;
    bool is_initialized;
    read(stream, is_initialized);
    LazyTensor lt;
    if (!is_initialized) {
        return lt;
    }
    lt.m_impl = LazyTensorImpl::deserialize(stream);
    return lt;
}

std::size_t LazyTensor::Hash::operator()(const LazyTensor& lt) const {
    return lt.get_hash();
}

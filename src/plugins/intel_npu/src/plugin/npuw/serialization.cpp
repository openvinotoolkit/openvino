// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "serialization.hpp"

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "lazy_tensor.hpp"
#include "logging.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "spatial.hpp"
#include "util.hpp"

// NOTE: This construtor should only be used when exporting blobs
ov::npuw::s11n::WeightsContext::WeightsContext(bool _is_weightless,
                                               const std::unordered_map<const void*, std::size_t>& _const_to_offset)
    : is_weightless(_is_weightless),
      const_to_offset(_const_to_offset) {}

// NOTE: This construtor can and should only be used when importing blobs
ov::npuw::s11n::WeightsContext::WeightsContext(const ov::npuw::s11n::WeightsPtr& _weights,
                                               const std::string& _weights_path,
                                               const s11n::WeightsContext::ConstsCache& _consts_cache,
                                               const BF16Cache& _bf16_consts)
    : weights(_weights),
      weights_path(_weights_path),
      consts_cache(_consts_cache),
      bf16_consts(_bf16_consts) {
    is_weightless = _weights || !_consts_cache.empty();
}

ov::npuw::s11n::BF16Cache ov::npuw::s11n::get_bf16_consts(const std::shared_ptr<ov::Model>& model) {
    ov::npuw::s11n::BF16Cache bf16_cache;
    for (auto&& node_ptr : model->get_ordered_ops()) {
        if (const auto c = ov::as_type_ptr<ov::op::v0::Constant>(node_ptr)) {
            if (c->get_element_type() != ov::element::bf16) {
                continue;
            }
            auto rt_info = c->get_rt_info();
            auto weightless_cache_attr = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
            if (weightless_cache_attr == rt_info.end()) {
                continue;
            }
            std::size_t offset = weightless_cache_attr->second.as<ov::WeightlessCacheAttribute>().bin_offset;
            bf16_cache.insert({offset, c->get_byte_size()});
        }
    }
    return bf16_cache;
}

void ov::npuw::s11n::write(std::ostream& stream, const std::streampos& var) {
    stream.write(reinterpret_cast<const char*>(&var), sizeof var);
}

void ov::npuw::s11n::write(std::ostream& stream, const std::string& var) {
    auto var_size = var.size();
    stream.write(reinterpret_cast<const char*>(&var_size), sizeof var_size);
    stream.write(&var[0], var.size());
}

void ov::npuw::s11n::write(std::ostream& stream, const bool& var) {
    stream.write(reinterpret_cast<const char*>(&var), sizeof var);
}

void ov::npuw::s11n::write(std::ostream& stream, const float& var) {
    stream.write(reinterpret_cast<const char*>(&var), sizeof var);
}

void ov::npuw::s11n::write(std::ostream& stream, const ov::npuw::compiled::Spatial& var) {
    using ov::npuw::s11n::write;

    write(stream, var.params.size());
    for (const auto& p : var.params) {
        write(stream, p.idx);
        write(stream, p.dim);
    }
    write(stream, var.range);
    write(stream, var.nway);
    write(stream, var.out_dim);
    write(stream, var.nway_iters);
    write(stream, var.tail_size);
}

void ov::npuw::s11n::write(std::ostream& stream, const ov::Tensor& var) {
    using ov::npuw::s11n::write;

    if (!var) {
        write(stream, false);
        return;
    }
    write(stream, true);

    auto type_str = var.get_element_type().to_string();
    write(stream, type_str);
    write(stream, var.get_shape());
    write(stream, var.get_byte_size());

    ov::Tensor tensor;
    if (var.is_continuous()) {
        tensor = var;
    } else {
        // Just copy strided tensor to a non-strided one
        tensor = ov::Tensor(var.get_element_type(), var.get_shape());
        var.copy_to(tensor);
    }
    NPUW_ASSERT(tensor);
    size_t blob_size = var.get_byte_size();
    if (blob_size > static_cast<decltype(blob_size)>(std::numeric_limits<std::streamsize>::max())) {
        OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
    }
    stream.write(reinterpret_cast<const char*>(var.data()), static_cast<std::streamsize>(blob_size));
}

void ov::npuw::s11n::write(std::ostream& stream, const ::intel_npu::Config& var) {
    write(stream, var.toString());
}

void ov::npuw::s11n::write(std::ostream& stream, const ov::Output<const ov::Node>& var) {
    write(stream, var.get_element_type().to_string());
    write(stream, var.get_partial_shape().to_string());
    write(stream, var.get_names());
}

void ov::npuw::s11n::write_any(std::ostream& stream, const ov::Any& var) {
    auto str = ov::npuw::s11n::anyToString(var);
    write(stream, str);
}

void ov::npuw::s11n::write(std::ostream& stream, const ov::npuw::weights::LazyTensor& var) {
    var.serialize(stream);
}

void ov::npuw::s11n::write(std::ostream& stream, const ov::CacheMode& var) {
    stream.write(reinterpret_cast<const char*>(&var), sizeof var);
}

void ov::npuw::s11n::write(std::ostream& stream, const ov::element::Type& var) {
    stream.write(reinterpret_cast<const char*>(&var), sizeof var);
}

void ov::npuw::s11n::write(std::ostream& stream, const ov::hint::PerformanceMode& var) {
    stream.write(reinterpret_cast<const char*>(&var), sizeof var);
}

void ov::npuw::s11n::write(std::ostream& stream, const ov::AnyMap& var) {
    auto str = ov::npuw::s11n::anyMapToString(var);
    write(stream, str);
}

void ov::npuw::s11n::read(std::istream& stream, std::streampos& var) {
    stream.read(reinterpret_cast<char*>(&var), sizeof var);
}

void ov::npuw::s11n::read(std::istream& stream, std::string& var) {
    std::size_t var_size = 0;
    stream.read(reinterpret_cast<char*>(&var_size), sizeof var_size);
    var.resize(var_size);
    stream.read(&var[0], var_size);
}

void ov::npuw::s11n::read(std::istream& stream, bool& var) {
    stream.read(reinterpret_cast<char*>(&var), sizeof var);
}

void ov::npuw::s11n::read(std::istream& stream, float& var) {
    stream.read(reinterpret_cast<char*>(&var), sizeof var);
}

void ov::npuw::s11n::read(std::istream& stream, ov::npuw::compiled::Spatial& var) {
    using ov::npuw::s11n::read;

    std::size_t params_size = 0;
    read(stream, params_size);
    for (std::size_t i = 0; i < params_size; ++i) {
        ov::npuw::compiled::Spatial::Param p;
        read(stream, p.idx);
        read(stream, p.dim);
        var.params.push_back(p);
    }
    read(stream, var.range);
    read(stream, var.nway);
    read(stream, var.out_dim);
    read(stream, var.nway_iters);
    read(stream, var.tail_size);
}

void ov::npuw::s11n::read(std::istream& stream, ov::Tensor& var) {
    bool is_initialized = false;
    read(stream, is_initialized);

    if (!is_initialized) {
        return;
    }

    std::string type_str;
    read(stream, type_str);
    ov::element::Type type(type_str);

    ov::Shape shape;
    read(stream, shape);

    std::size_t byte_size = 0;
    read(stream, byte_size);

    var = ov::Tensor(type, shape);

    stream.read(reinterpret_cast<char*>(var.data()), byte_size);
}

void ov::npuw::s11n::read(std::istream& stream, ::intel_npu::Config& var) {
    std::string str;
    read(stream, str);
    var.fromString(str);
}

void ov::npuw::s11n::read(std::istream& stream, std::shared_ptr<ov::op::v0::Parameter>& var) {
    std::string elem_type_str;
    std::string part_shape_str;
    std::unordered_set<std::string> names;
    read(stream, elem_type_str);
    read(stream, part_shape_str);
    read(stream, names);
    // NOTE: the code below is taken from NPU plugin's create_dummy_model()
    var = std::make_shared<op::v0::Parameter>(ov::element::Type(elem_type_str), ov::PartialShape(part_shape_str));
    var->set_friendly_name(*names.begin());  // FIXME: any_name ?
    var->output(0).get_tensor().set_names(names);
}

void ov::npuw::s11n::read(std::istream& stream, std::shared_ptr<ov::Node>& var) {
    std::string elem_type_str;
    std::string part_shape_str;
    std::unordered_set<std::string> names;
    read(stream, elem_type_str);
    read(stream, part_shape_str);
    read(stream, names);
    // NOTE: the code below is taken from NPU plugin's create_dummy_model()
    std::shared_ptr<ov::Node> res =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type(elem_type_str), std::vector<size_t>{1});
    // FIXME: serialize names as well?
    const std::shared_ptr<ov::descriptor::Tensor>& tensor_dummy =
        std::make_shared<ov::descriptor::Tensor>(ov::element::Type(elem_type_str),
                                                 ov::PartialShape(part_shape_str),
                                                 names);
    var = std::make_shared<ov::op::v0::Result>(res);
    var->output(0).set_tensor_ptr(tensor_dummy);
    var->set_friendly_name(*names.begin());  // any_name ?
}

void ov::npuw::s11n::read_any(std::istream& stream, ov::Any& var) {
    std::string str;
    read(stream, str);
    var = ov::npuw::s11n::stringToAny(str);
}

void ov::npuw::s11n::read(std::istream& stream, ov::npuw::weights::LazyTensor& var) {
    var = ov::npuw::weights::LazyTensor::deserialize(stream);
}

void ov::npuw::s11n::read(std::istream& stream, ov::CacheMode& var) {
    stream.read(reinterpret_cast<char*>(&var), sizeof var);
}

void ov::npuw::s11n::read(std::istream& stream, ov::element::Type& var) {
    stream.read(reinterpret_cast<char*>(&var), sizeof var);
}

void ov::npuw::s11n::read(std::istream& stream, ov::hint::PerformanceMode& var) {
    stream.read(reinterpret_cast<char*>(&var), sizeof var);
}

void ov::npuw::s11n::read(std::istream& stream, ov::AnyMap& var) {
    std::string str;
    read(stream, str);
    var = ov::npuw::s11n::stringToAnyMap(str);
}

// Weightless
// FIXME: all serialization needs a good rewriting
void ov::npuw::s11n::write_weightless(std::ostream& stream,
                                      const std::vector<ov::Tensor>& var,
                                      const ov::npuw::s11n::WeightsContext& ctx) {
    write(stream, var.size());
    for (const auto& t : var) {
        if (!t) {
            write(stream, false);
            continue;
        }
        write(stream, true);
        auto data = t.data();
        auto iter = ctx.const_to_offset.find(data);
        if (iter == ctx.const_to_offset.end()) {
            write(stream, false);
            write(stream, t);
        } else {
            write(stream, true);
            write(stream, t.get_element_type().to_string());
            write(stream, t.get_shape());
            write(stream, t.get_byte_size());
            write(stream, iter->second);  // offset in weights file
        }
    }
}

void ov::npuw::s11n::read_weightless(std::istream& stream,
                                     std::vector<ov::Tensor>& var,
                                     const ov::npuw::s11n::WeightsContext& ctx) {
    var.clear();
    std::size_t size;
    read(stream, size);
    for (std::size_t i = 0; i < size; ++i) {
        bool is_initialized = false;
        read(stream, is_initialized);
        if (!is_initialized) {
            var.push_back(ov::Tensor());
            continue;
        }
        bool is_weightless = false;
        read(stream, is_weightless);
        if (is_weightless) {
            std::string type_str;
            read(stream, type_str);
            ov::element::Type type(type_str);
            ov::Shape shape;
            read(stream, shape);
            std::size_t byte_size = 0;
            read(stream, byte_size);
            std::size_t offset = 0;
            read(stream, offset);
            ov::Tensor t(type, shape);

            if (ctx.weights) {
                if (ctx.bf16_consts.find({offset, byte_size}) != ctx.bf16_consts.end()) {
                    NPUW_ASSERT(type == ov::element::f16);
                    // Read original bf16 weight
                    auto bf16_tensor = ov::Tensor(ov::element::bf16, shape);
                    NPUW_ASSERT(bf16_tensor.get_byte_size() == byte_size);
                    std::memcpy(bf16_tensor.data(), ctx.weights->get_ptr(offset), byte_size);

                    NPUW_ASSERT(bf16_tensor.get_size() == t.get_size());

                    // Transform bf16 to f16 tensor
                    using dst_type = typename element_type_traits<ov::element::Type_t::f16>::value_type;
                    auto src_data = bf16_tensor.data<ov::bfloat16>();
                    auto dst_data = t.data<dst_type>();
                    ov::reference::convert_from_bf16_to_f16_with_clamp(src_data, dst_data, t.get_size());
                } else {
                    std::memcpy(t.data(), ctx.weights->get_ptr(offset), byte_size);
                }
            } else {
                auto it = ctx.consts_cache.find({offset, byte_size});
                NPUW_ASSERT(it != ctx.consts_cache.end() && "Couldn't find Constant in cache!");
                t = ov::npuw::util::copy_tensor_from_const(it->second);
                NPUW_ASSERT(t.get_byte_size() == byte_size && t.get_shape() == shape && t.get_element_type() == type);
            }

            var.push_back(t);
        } else {
            ov::Tensor t;
            read(stream, t);
            var.push_back(t);
        }
    }
}

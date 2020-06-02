// Copyright (C) 2019 Intel Corporationconvert2OutputVector
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <vector>
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/runtime/backend_manager.hpp>
#include <ngraph/component_manager.hpp>
#include <ngraph/runtime/backend.hpp>
#include <ngraph/runtime/tensor.hpp>

namespace ngraph {
namespace helpers {

template<ngraph::element::Type_t type>
struct nGraphTypesTrait {
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::boolean> {
    using value_type = bool;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::f64> {
    using value_type = double;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::f32> {
    using value_type = float;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::f16> {
    using value_type = ngraph::float16;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::bf16> {
    using value_type = ngraph::bfloat16;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::i8> {
    using value_type = int8_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::i16> {
    using value_type = int16_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::i32> {
    using value_type = int32_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::i64> {
    using value_type = int64_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::u8> {
    using value_type = uint8_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::u16> {
    using value_type = uint16_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::u32> {
    using value_type = uint32_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::u64> {
    using value_type = uint64_t;
};
enum PoolingTypes {
    MAX,
    AVG
};
enum ActivationTypes {
    None,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
    Exp,
    Log,
    Sign,
    Abs
};

ngraph::OutputVector convert2OutputVector(const std::vector<std::shared_ptr<ngraph::Node>> &nodes);

template<class opType>
inline ngraph::NodeVector castOps2Nodes(const std::vector<std::shared_ptr<opType>> &ops) {
    ngraph::NodeVector nodes;
    for (const auto &op : ops) {
        nodes.push_back(std::dynamic_pointer_cast<ngraph::Node>(op));
    }
    return nodes;
}

template<ngraph::element::Type_t type>
inline std::vector<std::shared_ptr<typename ngraph::helpers::nGraphTypesTrait<type>::value_type *>>
inferFnWithInterp(const std::shared_ptr<ngraph::Function> &fn,
                  std::vector<const typename ngraph::helpers::nGraphTypesTrait<type>::value_type *> inData) {
    using stdType = typename ngraph::helpers::nGraphTypesTrait<type>::value_type;
    ngraph::runtime::Backend::set_backend_shared_library_search_directory("");

    ngraph_register_interpreter_backend();
    auto backend = ngraph::runtime::Backend::create("INTERPRETER");

    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> inTensors;
    auto params = fn->get_parameters();
    for (size_t i = 0; i < params.size(); i++) {
        auto t = backend->create_tensor(type, params[i]->get_shape());
        t->write(inData[i], ngraph::shape_size(params[i]->get_shape()) * sizeof(stdType));
        inTensors.push_back(t);
    }

    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> outTensors;
    for (size_t i = 0; i < fn->get_output_size(); i++) {
        auto outShape = fn->get_output_shape(i);
        outTensors.push_back(backend->create_tensor(type, outShape));
    }

    auto handle = backend->compile(fn);
    handle->call_with_validate(outTensors, inTensors);
    std::vector<std::shared_ptr<stdType *>> outData;
    for (size_t i = 0; i < fn->get_output_size(); i++) {
        size_t nOutElements = ngraph::shape_size(fn->get_output_shape(i));
        auto outBuf = std::make_shared<stdType *>(new stdType[nOutElements]);
        outTensors[i]->read(*outBuf, nOutElements * sizeof(stdType));
        outData.push_back(outBuf);
    }
    return outData;
}
}  // namespace helpers
}  // namespace ngraph

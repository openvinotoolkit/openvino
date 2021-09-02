// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/function.hpp"
#include "openvino/core/partial_layout.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace preprocess {
class PrePostProcessorInternalData;
class InContext;
class OutProcessor;

class OPENVINO_API PrePostProcessorBase {
protected:
    std::unique_ptr<PrePostProcessorInternalData> m_impl;
    PrePostProcessorBase(std::unique_ptr<PrePostProcessorInternalData>&& impl);
    PrePostProcessorBase(PrePostProcessorBase&&);

public:
    virtual ~PrePostProcessorBase();
    InContext in();
    InContext in(int port_index);
    // InContext in(std::string tensor_name);
    OutProcessor out();
    std::shared_ptr<Function> build(const std::shared_ptr<Function>& function);
};

/// \brief TBD
class OPENVINO_API PrePostProcessor : public PrePostProcessorBase {
public:
    PrePostProcessor();
};

class InTensorContext;
class PreProcessContext;
class InNetworkContext;

class OPENVINO_API InContext : public PrePostProcessorBase {
public:
    InContext(std::unique_ptr<PrePostProcessorInternalData>&& impl);
    InContext(InContext&&);
    InTensorContext tensor();
    PreProcessContext preprocess();
    InNetworkContext network();
};

class OPENVINO_API InTensorContext : public PrePostProcessorBase {
public:
    InTensorContext(std::unique_ptr<PrePostProcessorInternalData>&& impl);
    InTensorContext(InTensorContext&&);

    InTensorContext& set_element_type(ov::element::Type type);
    InTensorContext& set_layout(const PartialLayout& layout);
    PreProcessContext preprocess();
    InNetworkContext network();
};

class OPENVINO_API PreProcessContext : public PrePostProcessorBase {
public:
    PreProcessContext(std::unique_ptr<PrePostProcessorInternalData>&& impl);
    PreProcessContext(PreProcessContext&&);

    PreProcessContext& convert_element_type(ov::element::Type type);
    PreProcessContext& scale(float value);
    PreProcessContext& scale(const std::vector<float>& values);
    PreProcessContext& mean(float value);
    PreProcessContext& mean(const std::vector<float>& values);
    InNetworkContext network();
};

class OPENVINO_API InNetworkContext : public PrePostProcessorBase {
public:
    InNetworkContext(std::unique_ptr<PrePostProcessorInternalData>&& impl);
    InNetworkContext(InNetworkContext&&);

    InNetworkContext& set_layout(const PartialLayout& layout);
};

class OPENVINO_API OutProcessor : public PrePostProcessorBase {
    std::unique_ptr<PrePostProcessorInternalData> m_impl;

public:
    OutProcessor(std::unique_ptr<PrePostProcessorInternalData>&& impl);
    OutProcessor(OutProcessor&&);
};

}  // namespace preprocess
}  // namespace ov

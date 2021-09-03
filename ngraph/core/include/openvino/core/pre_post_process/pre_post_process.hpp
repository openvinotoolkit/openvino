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

class InputInfo;
class InputTensorInfo;
class PreProcessSteps;
class InputNetworkInfo;

/// \brief TBD
class OPENVINO_API PrePostProcessor {
    class PrePostProcessorImpl;
    std::unique_ptr<PrePostProcessorImpl> m_impl;

public:
    PrePostProcessor();
    ~PrePostProcessor();
    PrePostProcessor& in(InputInfo&& builder) &;
    PrePostProcessor&& in(InputInfo&& builder) &&;
    std::shared_ptr<Function> build(const std::shared_ptr<Function>& function);

    size_t inputs_size() const;
    const InputInfo& get_inputs_data(size_t index) const;
};

class OPENVINO_API InputInfo {
    class InputInfoImpl;
    std::unique_ptr<InputInfoImpl> m_impl;
    friend class PrePostProcessor;

public:
    InputInfo();
    InputInfo(InputInfo&&);
    ~InputInfo();

    InputInfo& tensor(InputTensorInfo&& builder) &;
    InputInfo&& tensor(InputTensorInfo&& builder) &&;
    InputInfo& preprocess(PreProcessSteps&& builder) &;
    InputInfo&& preprocess(PreProcessSteps&& builder) &&;
    InputInfo& network(InputNetworkInfo&& builder) &;
    InputInfo&& network(InputNetworkInfo&& builder) &&;

    const InputTensorInfo& get_tensor() const;
    const PreProcessSteps& get_preprocess() const;
    const InputNetworkInfo& get_network() const;
};

class OPENVINO_API InputTensorInfo {
    class InputTensorImpl;
    std::unique_ptr<InputTensorImpl> m_impl;

public:
    InputTensorInfo();
    InputTensorInfo(InputTensorInfo&&);
    InputTensorInfo& operator=(InputTensorInfo&&);
    ~InputTensorInfo();

    InputTensorInfo& set_element_type(ov::element::Type type) &;
    InputTensorInfo&& set_element_type(ov::element::Type type) &&;
    InputTensorInfo& set_layout(const PartialLayout& layout) &;
    InputTensorInfo&& set_layout(const PartialLayout& layout) &&;

    ov::element::Type get_element_type() const;
    const PartialLayout& get_layout() const;
};

class PreProcessContext {
    element::Type m_type;
    PartialLayout m_layout;

public:
    PreProcessContext() = default;
    void set_element_type(const element::Type& type) {
        m_type = type;
    }
    element::Type get_element_type() const {
        return m_type;
    }

    void set_layout(const PartialLayout& layout) {
        m_layout = layout;
    }
    PartialLayout get_layout() const {
        return m_layout;
    }
};

class OPENVINO_API PreProcessSteps {
    class PreProcessImpl;
    std::unique_ptr<PreProcessImpl> m_impl;
    friend class PrePostProcessor;

public:
    PreProcessSteps();
    PreProcessSteps(PreProcessSteps&&);
    PreProcessSteps& operator=(PreProcessSteps&&);
    ~PreProcessSteps();

    PreProcessSteps& convert_element_type(ov::element::Type type) &;
    PreProcessSteps&& convert_element_type(ov::element::Type type) &&;

    PreProcessSteps& scale(float value) &;
    PreProcessSteps&& scale(float value) &&;

    PreProcessSteps& scale(const std::vector<float>& values) &;
    PreProcessSteps&& scale(const std::vector<float>& values) &&;

    PreProcessSteps& mean(float value) &;
    PreProcessSteps&& mean(float value) &&;

    PreProcessSteps& mean(const std::vector<float>& values) &;
    PreProcessSteps&& mean(const std::vector<float>& values) &&;
};

class OPENVINO_API InputNetworkInfo {
    class InputNetworkInfoImpl;
    std::unique_ptr<InputNetworkInfoImpl> m_impl;
    friend class PrePostProcessor;

public:
    InputNetworkInfo();
    InputNetworkInfo(InputNetworkInfo&&);
    InputNetworkInfo& operator=(InputNetworkInfo&&);
    ~InputNetworkInfo();

    InputNetworkInfo& set_layout(const PartialLayout& layout) &;
    InputNetworkInfo&& set_layout(const PartialLayout& layout) &&;

    const PartialLayout& get_layout() const;
};

}  // namespace preprocess
}  // namespace ov

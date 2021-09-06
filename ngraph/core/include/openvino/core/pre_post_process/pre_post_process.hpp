// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/function.hpp"
#include "openvino/core/pre_post_process/input_info.hpp"

namespace ov {
namespace preprocess {

class PrePostProcessorImpl;

/// \brief Main class for adding pre- and post- processing steps to existing ov::Function
/// API has Builder-like style to allow chaining calls in client's code, like
/// \code{.cpp}
/// auto proc = PrePostProcessor().input(<for input1>).input(<input2>);
/// \endcode
class OPENVINO_API PrePostProcessor final {
    std::unique_ptr<PrePostProcessorImpl> m_impl;

public:
    /// \brief Default constructor
    PrePostProcessor();

    /// \brief Default move constructor
    PrePostProcessor(PrePostProcessor&&) noexcept;

    /// \brief Default move assignment operator
    PrePostProcessor& operator=(PrePostProcessor&&) noexcept;

    /// \brief Default destructor
    ~PrePostProcessor();

    /// \brief Adds pre-processing information and steps to input of model. This method can be used only if ov::Function
    /// passed on `build` has only one input
    ///
    /// \param builder Pre-processing data for input tensor of model.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PrePostProcessor& input(InputInfo&& builder) &;

    /// \brief Adds pre-processing information and steps to input of model. This method can be used only if ov::Function
    /// passed on `build` has only one input This version allows chaining if object represents Rvalue
    ///
    /// \param builder Pre-processing data for input tensor of model.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    PrePostProcessor&& input(InputInfo&& builder) &&;

    /// \brief Adds pre/post-processing operations to existing function
    ///
    /// \param function Existing function representing loaded model
    ///
    /// \return Function with added pre/post-processing operations
    std::shared_ptr<Function> build(const std::shared_ptr<Function>& function);
};

}  // namespace preprocess
}  // namespace ov

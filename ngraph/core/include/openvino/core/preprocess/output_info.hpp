// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/preprocess/output_network_info.hpp"
#include "openvino/core/preprocess/output_tensor_info.hpp"
#include "openvino/core/preprocess/postprocess_steps.hpp"

namespace ov {
namespace preprocess {

/// \brief Class holding postprocessing information for one output
/// From postprocessing pipeline perspective, each output can be represented as:
///    - Network's output info,  (OutputInfo::network)
///    - Postprocessing steps applied to user's input (OutputInfo::postprocess)
///    - User's desired output parameter information, which is a final one after preprocessing (OutputInfo::tensor)
///
/// API has Builder-like style to allow chaining calls in client's code, like
/// \code{.cpp}
/// auto proc = PrePostProcessor().output(InputInfo().network(...).preprocess(...).tensor(...);
/// \endcode
class OPENVINO_API OutputInfo final {
    class OutputInfoImpl;
    std::unique_ptr<OutputInfoImpl> m_impl;
    friend class PrePostProcessor;

public:
    /// \brief Empty constructor. Should be used only if network has exactly one output
    OutputInfo();

    /// \brief Constructor for particular output index of model
    ///
    /// \param output_index Index to address specified output parameter of model
    explicit OutputInfo(size_t output_index);

    /// \brief Constructor for particular output of model addressed by it's output name
    ///
    /// \param output_tensor_name Name of output tensor name
    explicit OutputInfo(const std::string& output_tensor_name);

    /// \brief Default move constructor
    OutputInfo(OutputInfo&&) noexcept;

    /// \brief Default move assignment operator
    OutputInfo& operator=(OutputInfo&&) noexcept;

    /// \brief Default destructor
    ~OutputInfo();

    /// \brief Set network's tensor information for output - Lvalue version
    ///
    /// \param builder Output network tensor information.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    OutputInfo& network(OutputNetworkInfo&& builder) &;

    /// \brief Set network's tensor information for output - Rvalue version
    ///
    /// \param builder Output network tensor information.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    OutputInfo&& network(OutputNetworkInfo&& builder) &&;

    /// \brief Set postprocessing operations for output - Lvalue version
    ///
    /// \param builder Postprocessing operations.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    OutputInfo& postprocess(PostProcessSteps&& builder) &;

    /// \brief Set postprocessing operations for output - Rvalue version
    ///
    /// \param builder Postprocessing operations.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner

    OutputInfo&& postprocess(PostProcessSteps&& builder) &&;

    /// \brief Set final output tensor information for output after postprocessing - Lvalue version
    ///
    /// \param builder Output tensor information.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    OutputInfo& tensor(OutputTensorInfo&& builder) &;

    /// \brief Set final output tensor information for output after postprocessing - Rvalue version
    ///
    /// \param builder Output tensor information.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    OutputInfo&& tensor(OutputTensorInfo&& builder) &&;
};

}  // namespace preprocess
}  // namespace ov

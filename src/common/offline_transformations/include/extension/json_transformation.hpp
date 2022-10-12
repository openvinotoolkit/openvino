// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "openvino/core/extension.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {

/// \brief Describes transformation that JsonConfigExtension can use as a target transformation specified by ID.
/// JsonTransformationExtension passes JSON parsed object and matched points in the graph to JsonTransformationExtension
/// instance, which is derived from DecoderTransformationExtension. DecoderTransformationExtension itself cannot be
/// used for this purpose because we need to pass those additional objects which are the result of JSON parsing and
/// graph matching that JsonTransformationExtension performs.
///
/// This class is left for backward compatibility only. In the future we would like to get rid off this class as well
/// as from JsonConfigExtension, and users will use DecoderTransformationExtension only to match sub-graph and modify it
/// like we do in any other transformations.
///
/// Unlike DecoderTransformationExtension, which is initialized by some ready-to-use transformation code and is not used
/// to derive new classes by regular users, this class is intended to be derived from and it doesn't have convenient
/// ctos to be initialized. So it is intended for more advanced, internal users inside such components like Model
/// Optimizer.
class JsonTransformationExtension : public DecoderTransformationExtension {
public:
    explicit JsonTransformationExtension(const std::string& id) : m_id(id) {}

    /// \brief The name of the transformation to identify it from JSON file field 'id'
    const std::string& id() const {
        return m_id;
    }

    /// \brief Modifies OV Model according to the provided rules.
    ///
    /// \param[in]  function                   The OV Model object.
    /// \param[in]  replacement_descriptions   The rules to modify the model in .json format.
    virtual bool transform(const std::shared_ptr<ov::Model>& function,
                           const std::string& replacement_descriptions) const = 0;

private:
    std::string m_id;
};
}  // namespace frontend
}  // namespace ov

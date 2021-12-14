// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "common/extensions/decoder_transformation_extension.hpp"
#include "common/frontend_defs.hpp"
#include "nlohmann/json.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {

/// \brief Describes transformation that JsonConfigExtension can use as a target transformation spedified by ID.
/// JsonTransformationExtension passes JSON parsed object and mathed points in the graph to JsonTransformationExtension
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

    virtual bool transform(std::shared_ptr<ov::Model>& function,
                           const std::string& replacement_descriptions) const = 0;

private:
    std::string m_id;
};
}  // namespace frontend
}  // namespace ov

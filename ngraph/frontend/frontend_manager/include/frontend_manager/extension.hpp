// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <fstream>
#include <type_traits>
#include "frontend_manager_defs.hpp"
#include <openvino/pass/pass.hpp>
#include <openvino/core/extension.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/manager.hpp>


namespace ngraph {
namespace frontend {


class FRONTEND_API DecoderTransformationExtension : public ov::Extension {
public:

    //DecoderTransformationExtension () {}

    // Create a custom functional pass where code of the pass is implemented as a function.
    DecoderTransformationExtension (std::function<bool(std::shared_ptr<ov::Function>)> function_pass);

    // Create a custom matcher pass where the code of matcher pass initialization is a given function.
    DecoderTransformationExtension (std::function<void(ov::pass::MatcherPass*)> matcher_pass_initializer);

    // Register existing transformation object which will be copied and kept for further registration.
    template <typename Transformation,
            typename std::enable_if<std::is_base_of<ov::pass::PassBase, Transformation>::value, bool>::type = true>
    DecoderTransformationExtension (const Transformation& transformation) :
        m_registration([transformation](ov::pass::Manager& manager) {
            manager.register_pass<Transformation>(transformation);
        }) {}

    // Register pass from this object in a given pass manager object
    void register_pass (ov::pass::Manager& manager) const;

private:

    std::function<void(ov::pass::Manager&)> m_registration;
};


// Reads MO config file and delegate transformation functionality to specified transformation ID from the config
class FRONTEND_API JsonConfigExtension : public DecoderTransformationExtension {
public:
    JsonConfigExtension (const std::string& config_path);
};


/// \brief Provides callback to report telemetry information back to Python code
class FRONTEND_API TelemetryExtension : public ov::Extension {
public:

    TelemetryExtension (std::function<void(const std::string& message)> callback) : m_callback(callback) {}
    void send (const std::string& message) { m_callback(message); }

private:

    std::function<void(const std::string& message)> m_callback;
};


class FRONTEND_API NodeContext {
public:
    NodeContext (const std::string& _op_type, OutputVector _ng_inputs) : m_op_type(_op_type), m_ng_inputs(_ng_inputs) {}
    OutputVector get_ng_inputs() const { return m_ng_inputs; }
    const std::string& op_type() const { return m_op_type; }

private:

    OutputVector m_ng_inputs;
    std::string m_op_type;
};


class FRONTEND_API ConversionExtension : public ov::Extension {
public:

    ConversionExtension (const std::string& optype, std::function<OutputVector(std::shared_ptr<NodeContext>)> converter) :
        m_optype(optype), m_converter(converter) {}

    std::string m_optype;
    std::function<OutputVector(std::shared_ptr<NodeContext>)> m_converter;
};

}  // namespace frontend

}  // namespace ngraph

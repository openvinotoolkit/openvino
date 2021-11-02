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
#include "../../../../thirdparty/nlohmann/json/json.hpp"


namespace ngraph {
namespace frontend {

/// Holds a transformation that is applied just after the original model graph is decoded.
/** This class is a holder for transformation. The transformation can be specified as
 *  FunctionPass or MathcerPass derivatives or as a function that can be used to build corresponding
 *  FunctionPass or MatcherPass object. The type of the extension is determined in the moment of creation by
 *  calling corresponding ctor.
 */
class FRONTEND_API DecoderTransformationExtension : public ov::Extension {
public:

    DecoderTransformationExtension () {}

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

protected:

    void set_registration (std::function<void(ov::pass::Manager&)> registration) {
        m_registration = registration;
    }

private:

    std::function<void(ov::pass::Manager&)> m_registration;
};


/// \brief Describes transformation that JsonConfigExtension can use as a target transformation spedified by ID.
/** JsonTransformationExtension passes JSON parsed object and mathed points in the graph to JsonTransformationExtension
 *  instance, which is derived from DecoderTransformationExtension. DecoderTransformationExtension itself cannot be
 *  used for this purpose because we need to pass those additional objects which are the result of JSON parsing and
 *  graph matching that JsonTransformationExtension performs.
 *
 *  This class is left for backward compatibility only. In the future we would like to get rid off this class as well
 *  as from JsonConfigExtension, and users will use DecoderTransformationExtension only to match sub-graph and modify it
 *  like we do in any other transformations.
 *
 *  Unlike DecoderTransformationExtension, which is initialized by some ready-to-use transformation code and is not used
 *  to derive new classes by regular users, this class is intended to be derived from and it doesn't have convenient
 *  ctos to be initialized. So it is intended for more advanced, internal users inside such components like Model Optimizer.
 */
class FRONTEND_API JsonTransformationExtension : public DecoderTransformationExtension {
public:

    JsonTransformationExtension (const std::string& id) : m_id(id) {}

    // The name of the transformation to identify it from JSON file field 'id'
    const std::string& id () const { return m_id; }

    virtual bool transform (
            std::shared_ptr<ov::Function>& function,
            const nlohmann::json& replacement_descriptions
    ) const = 0;

private:

    std::string m_id;
};


/// Reads MO config file and delegate transformation functionality to specified transformation ID
/// specified in the config.
class FRONTEND_API JsonConfigExtension : public DecoderTransformationExtension {
public:
    JsonConfigExtension (const std::string& config_path);
    ~JsonConfigExtension ();
private:
    std::vector<Extension::Ptr> m_loaded_extensions;
    std::shared_ptr<DecoderTransformationExtension> m_target_extension;
    const nlohmann::json m_replacement_descriptions;
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

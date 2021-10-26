// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <fstream>
#include "frontend_manager_defs.hpp"
#include <openvino/pass/pass.hpp>
#include "../../../../../thirdparty/nlohmann/json/json.hpp"

namespace ngraph {
namespace frontend {

class FRONTEND_API CustomFunctionPass : public ov::pass::FunctionPass {
public:

    CustomFunctionPass (std::function<bool(std::shared_ptr<ov::Function>)> pass) :
        m_pass(pass)
    {}

    bool run_on_function (std::shared_ptr<ov::Function> f) override
    {
        return m_pass(f);
    }
private:

    std::function<bool(std::shared_ptr<ov::Function>)> m_pass;
};

class FRONTEND_API DecoderTransformationExtension : public Extension {
public:
    DecoderTransformationExtension () {}
    DecoderTransformationExtension (std::function<bool(std::shared_ptr<ov::Function>)> pass) : m_pass(pass) {}
    virtual std::function<bool(std::shared_ptr<ov::Function>)> get_pass () const { return m_pass; }
protected:
    std::function<bool(std::shared_ptr<ov::Function>)> m_pass;
};

// Reads MO config file and delegate transformation functionality to specified transformation ID from the config
class FRONTEND_API JsonConfigExtension : public DecoderTransformationExtension {
public:
    JsonConfigExtension (const std::string& config_path)
    {
        nlohmann::json config_json;
        std::ifstream config_file(config_path);
        config_file >> config_json;
        std::cerr << "++++++++++++++ Read json: ++++++++++++++++\n" << config_json;

        // TODO: Implement real loading of extensions from shared library here using the same functionality
        // as add_extension uses.
        m_pass = [](std::shared_ptr<ov::Function> f){
            auto ops = f->get_ordered_ops();
            std::cerr << "HELLO! Run on function with " << ops.size() << " nodes\n";
            return true;
        };
    }
};

/// \brief Provides callback to report telemetry information back to Python code
class FRONTEND_API TelemetryExtension : public Extension {
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

class FRONTEND_API OpExtension : public Extension {
public:

    OpExtension (const std::string& optype, std::function<OutputVector(std::shared_ptr<NodeContext>)> converter) :
        m_optype(optype), m_converter(converter) {}

    std::string m_optype;
    std::function<OutputVector(std::shared_ptr<NodeContext>)> m_converter;
};

}  // namespace frontend

}  // namespace ngraph

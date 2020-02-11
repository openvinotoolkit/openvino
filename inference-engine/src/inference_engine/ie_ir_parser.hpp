// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides interface for network reader that is used to build networks from a given IR
 * @file ie_icnn_net_reader.h
 */
#pragma once

#include <ie_blob.h>
#include <ie_iextension.h>
#include <xml_parse_utils.h>

#include <algorithm>
#include <details/caseless.hpp>
#include <map>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "cnn_network_impl.hpp"
#include "ie_ngraph_utils.hpp"

namespace InferenceEngine {

class IParser {
public:
    using Ptr = std::shared_ptr<IParser>;
    virtual ~IParser() = default;
    virtual std::shared_ptr<ngraph::Function> parse(const pugi::xml_node& root, const Blob::CPtr& weights) = 0;
};

class IRParser {
public:
    explicit IRParser(size_t version);
    IRParser(size_t version, const std::vector<InferenceEngine::IExtensionPtr>& exts);
    std::shared_ptr<ngraph::Function> parse(const pugi::xml_node& root, const Blob::CPtr& weights);
    virtual ~IRParser() = default;

private:
    IParser::Ptr parser;
};

class V10Parser : public IParser {
public:
    explicit V10Parser(const std::vector<IExtensionPtr>& exts);
    std::shared_ptr<ngraph::Function> parse(const pugi::xml_node& root, const Blob::CPtr& weights) override;

private:
    std::map<std::string, ngraph::OpSet> opsets;

    struct GenericLayerParams {
        struct LayerPortData {
            size_t portId;
            ngraph::element::Type_t precision;
            SizeVector dims;
        };
        size_t layerId;
        std::string version;
        std::string name;
        std::string type;
        ngraph::element::Type_t precision;
        std::vector<LayerPortData> inputPorts;
        std::vector<LayerPortData> outputPorts;

        size_t getRealInputPortId(size_t id) {
            size_t real_id = 0;
            for (auto& it : inputPorts) {
                if (it.portId == id) {
                    return real_id;
                }
                ++real_id;
            }
            THROW_IE_EXCEPTION << "Can not find input port with id " << id << " in layer " << name;
        }

        size_t getRealOutputPortId(size_t id) {
            size_t real_id = 0;
            for (auto& it : outputPorts) {
                if (it.portId == id) {
                    return real_id;
                }
                ++real_id;
            }
            THROW_IE_EXCEPTION << "Can not find output port with id " << id << " in layer " << name;
        }
    };

    class LayerBaseCreator {
    private:
        std::string type;

    protected:
        explicit LayerBaseCreator(const std::string& type): type(type) {}
        std::string getType() {
            return type;
        }
        template <class T>
        std::vector<T> getParameters(const pugi::xml_node& node, const std::string& name) {
            std::vector<T> result;
            std::string param = XMLParseUtils::GetStrAttr(node, name.c_str());
            std::stringstream ss(param);
            std::string field;
            while (getline(ss, field, ',')) {
                std::stringstream fs(field);
                T value;
                fs >> value;
                result.emplace_back(value);
            }
            return result;
        }

        template <class T>
        std::vector<T> getParameters(const pugi::xml_node& node, const std::string& name, const std::vector<T>& def) {
            std::vector<T> result;
            std::string param = XMLParseUtils::GetStrAttr(node, name.c_str(), "");
            if (param.empty()) return def;
            std::stringstream ss(param);
            std::string field;
            while (getline(ss, field, ',')) {
                std::stringstream fs(field);
                T value;
                fs >> value;
                result.emplace_back(value);
            }
            return result;
        }

        void checkParameters(const ngraph::OutputVector& inputs, const GenericLayerParams& params, int numInputs, int numOutputs) {
            if (numInputs >= 0 && params.inputPorts.size() != numInputs) {
                THROW_IE_EXCEPTION << params.type << " layer " << params.name << " with id: " << params.layerId
                                   << " has incorrect number of input ports!";
            }
            for (size_t i = 0; i < params.inputPorts.size(); i++) {
                for (const auto& dim : params.inputPorts[i].dims) {
                    if (!dim)
                        THROW_IE_EXCEPTION << params.type << " layer " << params.name << " with id: " << params.layerId
                                           << " has incorrect dimensions in the input port" << i << "!";
                }
            }
            if (numOutputs >= 0 && params.outputPorts.size() != numOutputs) {
                THROW_IE_EXCEPTION << params.type << " layer " << params.name << " with id: " << params.layerId
                                   << " has incorrect number of output ports!";
            }
            for (size_t i = 0; i < params.outputPorts.size(); i++) {
                for (const auto& dim : params.outputPorts[i].dims) {
                    if (!dim)
                        THROW_IE_EXCEPTION << params.type << " layer " << params.name << " with id: " << params.layerId
                                           << " has incorrect dimensions in the output port" << i << "!";
                }
            }
            if (inputs.size() != params.inputPorts.size())
                THROW_IE_EXCEPTION << params.type << " layer " << params.name << " with id: " << params.layerId
                    << " has incorrect number of inputs!";
        }

    public:
        virtual ~LayerBaseCreator() {}
        virtual std::shared_ptr<ngraph::Node> createLayer(const ngraph::OutputVector& inputs,
                                                          const pugi::xml_node& node, const Blob::CPtr& weights,
                                                          const GenericLayerParams& layerParsePrms) = 0;

        bool shouldCreate(const std::string& nodeType) const;

        std::shared_ptr<ngraph::Node> createOptionalParameter(const GenericLayerParams::LayerPortData& port);
    };

    template <class T>
    class LayerCreator : public LayerBaseCreator {
    public:
        explicit LayerCreator(const std::string& type): LayerBaseCreator(type) {}
        std::shared_ptr<ngraph::Node> createLayer(const ngraph::OutputVector& inputs, const pugi::xml_node& node,
                                                  const Blob::CPtr& weights,
                                                  const GenericLayerParams& layerParsePrms) override;
    };

    std::shared_ptr<ngraph::Node> createNode(const ngraph::OutputVector& inputs, const pugi::xml_node& node,
                                             const Blob::CPtr& weights, const GenericLayerParams& params);

    GenericLayerParams parseGenericParams(const pugi::xml_node& node);

    std::map<std::string, DataPtr> portsToData;
    std::map<std::string, GenericLayerParams> layersParseInfo;

    class XmlDeserializer : public ngraph::AttributeVisitor {
    public:
        explicit XmlDeserializer(const pugi::xml_node& node): node(node) {}
        void on_attribute(const std::string& name, std::string& value) override {
            std::string val;
            if (!getStrAttribute(node.child("data"), name, val)) return;
            value = val;
        }
        void on_attribute(const std::string& name, bool& value) override {
            std::string val;
            if (!getStrAttribute(node.child("data"), name, val)) return;
            std::transform(val.begin(), val.end(), val.begin(), [](char ch) {
                return std::tolower(static_cast<unsigned char>(ch));
            });
            std::set<std::string> true_names {"true", "1"};
            std::set<std::string> false_names {"false", "0"};

            bool is_true = true_names.find(val) != true_names.end();
            bool is_false = false_names.find(val) != false_names.end();

            if (!is_true && !is_false) return;
            value = is_true;
        }
        void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
            std::string val;
            if (!getStrAttribute(node.child("data"), name, val)) return;
            if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::element::Type>>(&adapter)) {
                static_cast<ngraph::element::Type&>(*a) = details::ngraph::convertPrecision(val);
            } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::PartialShape>>(&adapter)) {
                std::vector<int64_t> shape;
                std::vector<ngraph::Dimension> dims;
                if (!getParameters<int64_t>(node.child("data"), name, shape)) return;
                for (const auto& dim : shape) dims.emplace_back(dim);
                static_cast<ngraph::PartialShape&>(*a) = ngraph::PartialShape(dims);
            } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::Shape>>(&adapter)) {
                std::vector<size_t> shape;
                if (!getParameters<size_t>(node.child("data"), name, shape)) return;
                static_cast<ngraph::Shape&>(*a) = ngraph::Shape(shape);
            } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::Strides>>(&adapter)) {
                std::vector<size_t> shape;
                if (!getParameters<size_t>(node.child("data"), name, shape)) return;
                static_cast<ngraph::Strides&>(*a) = ngraph::Strides(shape);
            }
        }
        void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
            std::string val;
            if (!getStrAttribute(node.child("data"), name, val))
                return;
            double value;
            stringToType<double>(val, value);
            adapter.set(value);
        }
        void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
            std::string val;
            if (!getStrAttribute(node.child("data"), name, val))
                return;
            int64_t value;
            stringToType<int64_t>(val, value);
            adapter.set(value);
        }

    private:
        const pugi::xml_node node;

        bool getStrAttribute(const pugi::xml_node& node, const std::string& name, std::string& value) {
            if (!node) return false;

            auto attr = node.attribute(name.c_str());
            if (attr.empty()) return false;
            value = std::string(attr.value());
            return true;
        }

        template <class T>
        bool getParameters(const pugi::xml_node& node, const std::string& name, std::vector<T>& value) {
            std::string param;
            if (!getStrAttribute(node, name, param)) return false;
            std::stringstream ss(param);
            std::string field;
            while (getline(ss, field, ',')) {
                if (field.empty())
                    THROW_IE_EXCEPTION << "Cannot get vector of parameters! \"" << param << "\" is incorrect";
                std::stringstream fs(field);
                T val;
                fs >> val;
                value.emplace_back(val);
            }
            return true;
        }

        template <class T>
        bool stringToType(const std::string& valStr, T& value) {
            std::istringstream ss(valStr);
            if (ss.eof()) return false;
            ss >> value;
            return !ss.fail();
        }
    };
};

}  // namespace InferenceEngine

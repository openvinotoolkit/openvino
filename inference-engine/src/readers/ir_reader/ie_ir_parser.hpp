// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef IR_READER_V10
# include <ngraph/node.hpp>
# include <legacy/ie_ngraph_utils.hpp>
# include <cpp/ie_cnn_network.h>
#endif  // IR_READER_V10

#include <ie_blob.h>
#include <ie_icnn_network.hpp>
#include <ie_iextension.h>
#include <xml_parse_utils.h>

#include <cctype>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace InferenceEngine {

class IParser {
public:
    using Ptr = std::shared_ptr<IParser>;
    virtual ~IParser() = default;
    virtual std::shared_ptr<ICNNNetwork> parse(const pugi::xml_node& root, std::istream& binStream) = 0;
};

class IRParser {
public:
    explicit IRParser(size_t version);
    IRParser(size_t version, const std::vector<InferenceEngine::IExtensionPtr>& exts);
    std::shared_ptr<ICNNNetwork> parse(const pugi::xml_node& root, std::istream& binStream);
    virtual ~IRParser() = default;

private:
    IParser::Ptr parser;
};

class CNNParser : public IParser {
public:
    CNNParser() = default;
    std::shared_ptr<ICNNNetwork> parse(const pugi::xml_node& root, std::istream& binStream) override;
};

#ifdef IR_READER_V10

class V10Parser : public IParser {
public:
    explicit V10Parser(const std::vector<IExtensionPtr>& exts);
    std::shared_ptr<ICNNNetwork> parse(const pugi::xml_node& root, std::istream& binStream) override;

private:
    std::map<std::string, ngraph::OpSet> opsets;

    struct GenericLayerParams {
        struct LayerPortData {
            size_t portId;
            // Precision and dimensions are needed only for GenericIE op
            ngraph::element::Type_t precision;
            SizeVector dims;
        };
        size_t layerId;
        std::string version;
        std::string name;
        std::string type;
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

        void checkParameters(const ngraph::OutputVector& inputs, const GenericLayerParams& params, int numInputs) {
            if (numInputs >= 0 && inputs.size() != numInputs) {
                THROW_IE_EXCEPTION << params.type << " layer " << params.name << " with id: " << params.layerId
                                   << " has incorrect number of inputs! Expected: " << numInputs << ", actual: " << inputs.size();
            }
        }

    public:
        virtual ~LayerBaseCreator() {}
        virtual std::shared_ptr<ngraph::Node> createLayer(const ngraph::OutputVector& inputs,
                                                          const pugi::xml_node& node, std::istream& binStream,
                                                          const GenericLayerParams& layerParsePrms) = 0;

        bool shouldCreate(const std::string& nodeType) const;
        virtual ngraph::NodeTypeInfo getNodeType() const = 0;
    };

    template <class T>
    class LayerCreator : public LayerBaseCreator {
    public:
        explicit LayerCreator(const std::string& type): LayerBaseCreator(type) {}
        std::shared_ptr<ngraph::Node> createLayer(const ngraph::OutputVector& inputs, const pugi::xml_node& node,
                                                  std::istream& binStream,
                                                  const GenericLayerParams& layerParsePrms) override;
        ngraph::NodeTypeInfo getNodeType() const override {
            return T::type_info;
        }
    };

    std::shared_ptr<ngraph::Node> createNode(const ngraph::OutputVector& inputs, const pugi::xml_node& node,
                                             std::istream& binStream, const GenericLayerParams& params);

    GenericLayerParams parseGenericParams(const pugi::xml_node& node);
    void parsePreProcess(CNNNetwork& network, const pugi::xml_node& root, std::istream& binStream);

    std::map<std::string, DataPtr> portsToData;
    std::map<std::string, GenericLayerParams> layersParseInfo;

    class XmlDeserializer : public ngraph::AttributeVisitor {
    public:
        explicit XmlDeserializer(const pugi::xml_node& node): node(node) {}
        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& value) override {
            std::string val;
            if (!getStrAttribute(node.child("data"), name, val)) return;
            value.set(val);
        }
        void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& value) override {
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
            value.set(is_true);
        }
        void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
            std::string val;
            if (!getStrAttribute(node.child("data"), name, val)) return;
            if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::element::Type>>(&adapter)) {
                static_cast<ngraph::element::Type&>(*a) = details::convertPrecision(val);
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
#ifdef __APPLE__
            } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<std::vector<size_t>>>(&adapter)) {
                std::vector<size_t> result;
                if (!getParameters<size_t>(node.child("data"), name, result)) return;
                static_cast<std::vector<size_t>&>(*a) = result;
#else
            } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<std::vector<size_t>>>(&adapter)) {
                std::vector<size_t> result;
                if (!getParameters<size_t>(node.child("data"), name, result)) return;
                a->set(result);
#endif
            } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::AxisSet>>(&adapter)) {
                std::vector<size_t> axes;
                if (!getParameters<size_t>(node.child("data"), name, axes)) return;
                static_cast<ngraph::AxisSet&>(*a) = ngraph::AxisSet(axes);
            } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::op::TopKSortType>>(&adapter)) {
                if (!getStrAttribute(node.child("data"), name, val)) return;
                static_cast<ngraph::op::TopKSortType&>(*a) = ngraph::as_enum<ngraph::op::TopKSortType>(val);
            } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::op::TopKMode>>(&adapter)) {
                if (!getStrAttribute(node.child("data"), name, val)) return;
                static_cast<ngraph::op::TopKMode&>(*a) = ngraph::as_enum<ngraph::op::TopKMode>(val);
            }  else {
                THROW_IE_EXCEPTION << "Error IR reading. Attribute adapter can not be found for " << name
                                   << " parameter";
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

        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
            std::vector<float> value;
            if (!getParameters<float>(node.child("data"), name, value)) return;
            adapter.set(value);
        }

        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
            std::vector<std::string> value;
            if (!getParameters<std::string>(node.child("data"), name, value)) return;
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

#endif  // IR_READER_V10

}  // namespace InferenceEngine

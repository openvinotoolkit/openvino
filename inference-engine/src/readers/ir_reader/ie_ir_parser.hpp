// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef IR_READER_V10
# include <ngraph/node.hpp>
# include <ngraph/op/util/sub_graph_base.hpp>
# include <ie_ngraph_utils.hpp>
#endif  // IR_READER_V10

#include <ie_blob.h>
#include <cpp/ie_cnn_network.h>
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
    virtual std::shared_ptr<ICNNNetwork> parse(const pugi::xml_node& root, const Blob::CPtr& weights) = 0;
};

class IRParser {
public:
    explicit IRParser(size_t version);
    IRParser(size_t version, const std::vector<InferenceEngine::IExtensionPtr>& exts);
    std::shared_ptr<ICNNNetwork> parse(const pugi::xml_node& root, const Blob::CPtr& weights);
    virtual ~IRParser() = default;

private:
    IParser::Ptr parser;
};

class CNNParser : public IParser {
public:
    CNNParser() = default;
    std::shared_ptr<ICNNNetwork> parse(const pugi::xml_node& root, const Blob::CPtr& weights) override;
};

#ifdef IR_READER_V10

class V10Parser : public IParser {
public:
    explicit V10Parser(const std::vector<IExtensionPtr>& exts);
    std::shared_ptr<ICNNNetwork> parse(const pugi::xml_node& root, const Blob::CPtr& weights) override;

private:
    std::map<std::string, ngraph::OpSet> opsets;
    const std::vector<IExtensionPtr> _exts;

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
        static std::shared_ptr<ngraph::Node> fillSubGraphLayer(const ngraph::OutputVector& inputs, const pugi::xml_node& node,
                                                        const Blob::CPtr& weights,
                                                        const GenericLayerParams& layerParsePrms,
                                                        std::shared_ptr<ngraph::op::util::SubGraphOp> subgraph_op);
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
                                                          const pugi::xml_node& node, const Blob::CPtr& weights,
                                                          const GenericLayerParams& layerParsePrms) = 0;

        bool shouldCreate(const std::string& nodeType) const;
        virtual ngraph::NodeTypeInfo getNodeType() const = 0;
    };

    template <class T>
    class LayerCreator : public LayerBaseCreator {
    public:
        explicit LayerCreator(const std::string& type): LayerBaseCreator(type) {}
        std::shared_ptr<ngraph::Node> createLayer(const ngraph::OutputVector& inputs, const pugi::xml_node& node,
                                                  const Blob::CPtr& weights,
                                                  const GenericLayerParams& layerParsePrms) override;
        ngraph::NodeTypeInfo getNodeType() const override {
            return T::type_info;
        }
    };

    std::shared_ptr<ngraph::Node> createNode(const ngraph::OutputVector& inputs, const pugi::xml_node& node,
                                             const Blob::CPtr& weights, const GenericLayerParams& params);

    GenericLayerParams parseGenericParams(const pugi::xml_node& node);
    void parsePreProcess(CNNNetwork& network, const pugi::xml_node& root, const Blob::CPtr& weights);

    std::map<std::string, DataPtr> portsToData;
    std::map<std::string, GenericLayerParams> layersParseInfo;

    class XmlDeserializer : public ngraph::AttributeVisitor {
    public:
        explicit XmlDeserializer(const pugi::xml_node& node, const Blob::CPtr& weights): node(node), weights(weights) {}
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
            } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::CoordinateDiff>>(&adapter)) {
                std::vector<size_t> shape;
                if (!getParameters<size_t>(node.child("data"), name, shape)) return;
                std::vector<std::ptrdiff_t> coord_diff(shape.begin(), shape.end());
                static_cast<ngraph::CoordinateDiff&>(*a) = ngraph::CoordinateDiff(coord_diff);
            } else {
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
        void on_adapter(const std::string& name, ngraph::ValueAccessor<void*>& adapter) override  {
            std::string value;
            pugi::xml_node dn = node.child("data");
            auto type = XMLParseUtils::GetStrAttr(node, "type");

            if (dn.empty())
                THROW_IE_EXCEPTION << "No attrtibutes defined for " << type << " op!";

            if (getStrAttribute(dn, name, value)) {
                auto data = static_cast<char*>(adapter.get_ptr());
                size_t length = std::min(value.size(), adapter.size());
                value.copy(data, length);
            } else if (name == "value" && type == "Const") {
                std::vector<int64_t> shape;
                std::string el_type_str;

                size_t offset = XMLParseUtils::GetUInt64Attr(dn, "offset");
                size_t size = XMLParseUtils::GetUInt64Attr(dn, "size");
                if (!getStrAttribute(dn, "element_type", el_type_str)) return;
                if (!getParameters<int64_t>(dn, "shape", shape)) return;

                ngraph::element::Type el_type = details::convertPrecision(el_type_str);

                size_t length = weights->byteSize();
                if (!length)
                    THROW_IE_EXCEPTION << "Empty weights data in bin file or bin file cannot be found!";
                if (length < offset + size)
                    THROW_IE_EXCEPTION << "Incorrect weights in bin file!";
                if (size < std::ceil(ngraph::shape_size(shape) * el_type.bitwidth() / 8.f))
                    THROW_IE_EXCEPTION << "Attribute and shape size are inconsistent for " << type << " op!";

                auto data = static_cast<char*>(adapter.get_ptr());
                char* weights_data = weights->cbuffer().as<char*>() + offset;

                std::memcpy(data, weights_data, size);
            }
        }
        void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
            std::string val;
            if (!getStrAttribute(node.child("data"), name, val))
                return;
            int64_t value;
            stringToType<int64_t>(val, value);
            adapter.set(value);
        }

        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int32_t>>& adapter) override {
            std::vector<int32_t> value;
            if (!getParameters<int32_t>(node.child("data"), name, value)) return;
            adapter.set(value);
        }

        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
            std::vector<int64_t> value;
            if (!getParameters<int64_t>(node.child("data"), name, value)) return;
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
        const Blob::CPtr& weights;

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
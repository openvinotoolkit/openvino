// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef IR_READER_V10
# include <ngraph/node.hpp>
# include <ngraph/op/util/sub_graph_base.hpp>
# include <ngraph/opsets/opset.hpp>
# include <ie_ngraph_utils.hpp>
# include <ngraph/opsets/opset5.hpp>
#endif  // IR_READER_V10

#include <ie_blob.h>
#include <cpp/ie_cnn_network.h>
#include <ie_iextension.h>
#include <xml_parse_utils.h>

#include <cctype>
#include <algorithm>
#include <map>
#include <unordered_map>
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
    std::unordered_map<std::string, ngraph::OpSet> opsets;
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

        size_t getRealInputPortId(size_t id) const {
            size_t real_id = 0;
            for (auto& it : inputPorts) {
                if (it.portId == id) {
                    return real_id;
                }
                ++real_id;
            }
            THROW_IE_EXCEPTION << "Can not find input port with id " << id << " in layer " << name;
        }

        size_t getRealOutputPortId(size_t id) const {
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

        void checkParameters(const ngraph::OutputVector& inputs, const GenericLayerParams& params, size_t numInputs) {
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

    void parsePreProcess(CNNNetwork& network, const pugi::xml_node& root, const Blob::CPtr& weights);

    std::map<std::string, DataPtr> portsToData;
    std::map<std::string, GenericLayerParams> layersParseInfo;

    class XmlDeserializer : public ngraph::AttributeVisitor {
    public:
        explicit XmlDeserializer(const pugi::xml_node& node, const Blob::CPtr& weights,
        const std::unordered_map<std::string, ngraph::OpSet>& opsets) : node(node), weights(weights), opsets(opsets) {}
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
        void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override;
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

        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>& adapter) override;

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
        const std::unordered_map<std::string, ngraph::OpSet>& opsets;
        /// \brief Traverses port_map in order to create vector of InputDescription shared_ptrs.
        /// Shall be used only for ops which have port_map attribute.
        /// \param node xml op representation
        std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>> parseInputDescription(
            const pugi::xml_node& node);
        /// \brief Traverses port_map in order to create vector of OutputDescription shared_ptrs.
        /// Shall be used only for ops which have port_map attribute.
        /// \param node xml op representation
        std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>> parseOutputDescription(
            const pugi::xml_node& node);
        /// \brief Traverses nGraph body function for specified op type and creates a map of all
        ///  op iterations. Map constains type id and assigned to it consecutive number starting from 0.
        /// \param node xml op representation
        /// \param type op type name to find
        /// \return map container
        std::map<uint64_t, uint64_t> map_type_in_function(const pugi::xml_node& node, std::string type);
        /// \brief Traverses xml node representation in order to create nGraph function for it.
        /// \param node xml node representation
        /// \param weights weights attached to current node
        /// \return shared pointer to function representing input node
        std::shared_ptr<ngraph::Function> parse_function(const pugi::xml_node& root, const Blob::CPtr& weights);
        /// \brief Traverses xml node representation in order to get the purpose attribute of inputs/outputs in the body of Loop op.
        /// \param node xml node representation
        /// \return struct with value of purpuse attribute
        ngraph::op::v5::Loop::SpecialBodyPorts parsePurposeAttribute(const pugi::xml_node& node);

        GenericLayerParams parseGenericParams(const pugi::xml_node& node);
        std::shared_ptr<ngraph::Node> createNode(const ngraph::OutputVector& inputs, const pugi::xml_node& node,
                                             const Blob::CPtr& weights, const GenericLayerParams& params);

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
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/onnx/graph_iterator.hpp"

#include <onnx/onnx_pb.h>

#include <fstream>
#include <openvino/frontend/graph_iterator.hpp>

#include "load_from.hpp"
#include "onnx_utils.hpp"
#include "utils.hpp"

using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::Version;

TEST_P(FrontEndLoadFromTest, testLoadUsingSimpleGraphIterator) {
    ov::frontend::FrontEnd::Ptr fe;

    class SimpleIterator : public ov::frontend::onnx::GraphIterator {
    public:
        size_t size() const override {
            return 0;
        }
        void reset() override {};
        void next() override {};
        bool is_end() const override {
            return true;
        };
        std::shared_ptr<ov::frontend::onnx::DecoderBase> get_decoder() const override {
            return nullptr;
        };
        size_t get_subgraph_size() const override {
            return 0;
        };

        std::shared_ptr<GraphIterator> get_subgraph(size_t idx) const override {
            return nullptr;
        };

        int64_t get_opset_version(const std::string& domain) const {
            return 1;
        }

        ~SimpleIterator() override {};
    };

    auto iter = std::make_shared<SimpleIterator>();

    {
        auto graph_iter = std::dynamic_pointer_cast<ov::frontend::onnx::GraphIterator>(iter);
        ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework("onnx"))
            << "Could not create the ONNX FE using a pointer GraphIterator";
        ASSERT_NE(m_frontEnd, nullptr);

        ASSERT_EQ(m_frontEnd->supported(graph_iter), true);

        ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(graph_iter)) << "Could not load the model";
        ASSERT_NE(m_inputModel, nullptr);
    }
    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel)) << "Could not convert the model to OV representation";
    ASSERT_NE(model, nullptr);

    ASSERT_EQ(model->get_ordered_ops().size(), 0);
}

using ::ONNX_NAMESPACE::GraphProto;
using ::ONNX_NAMESPACE::NodeProto;
using ::ONNX_NAMESPACE::OperatorSetIdProto;
using ::ONNX_NAMESPACE::TensorProto;
using ::ONNX_NAMESPACE::TensorProto_DataLocation;
using ::ONNX_NAMESPACE::TensorProto_DataType;
using ::ONNX_NAMESPACE::ValueInfoProto;

namespace test_iterator {

namespace {
// THis is copied from utils/common.hpp
const ov::element::Type& get_ov_element_type(int64_t onnx_type) {
    switch (onnx_type) {
    case TensorProto_DataType::TensorProto_DataType_BOOL:
        return ov::element::boolean;
    case TensorProto_DataType::TensorProto_DataType_DOUBLE:
        return ov::element::f64;
    case TensorProto_DataType::TensorProto_DataType_FLOAT16:
        return ov::element::f16;
    case TensorProto_DataType::TensorProto_DataType_FLOAT:
        return ov::element::f32;
    case TensorProto_DataType::TensorProto_DataType_INT4:
        return ov::element::i4;
    case TensorProto_DataType::TensorProto_DataType_INT8:
        return ov::element::i8;
    case TensorProto_DataType::TensorProto_DataType_INT16:
        return ov::element::i16;
    case TensorProto_DataType::TensorProto_DataType_INT32:
        return ov::element::i32;
    case TensorProto_DataType::TensorProto_DataType_INT64:
        return ov::element::i64;
    case TensorProto_DataType::TensorProto_DataType_UINT4:
        return ov::element::u4;
    case TensorProto_DataType::TensorProto_DataType_UINT8:
        return ov::element::u8;
    case TensorProto_DataType::TensorProto_DataType_UINT16:
        return ov::element::u16;
    case TensorProto_DataType::TensorProto_DataType_UINT32:
        return ov::element::u32;
    case TensorProto_DataType::TensorProto_DataType_UINT64:
        return ov::element::u64;
    case TensorProto_DataType::TensorProto_DataType_UNDEFINED:
        return ov::element::dynamic;
    case TensorProto_DataType::TensorProto_DataType_BFLOAT16:
        return ov::element::bf16;
    case TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN:
        return ov::element::f8e4m3;
    case TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2:
        return ov::element::f8e5m2;
    case TensorProto_DataType::TensorProto_DataType_STRING:
        return ov::element::string;
    }
    throw std::runtime_error("Unsupported type");
}

ov::frontend::onnx::TensorMetaInfo extract_tensor_meta_info(const TensorProto* tensor_info,
                                                            const ValueInfoProto* value_info,
                                                            const GraphProto* graph_def) {
    ov::frontend::onnx::TensorMetaInfo tensor_meta_info;
    if (value_info == nullptr && tensor_info->has_name()) {
        for (const auto& val : graph_def->value_info()) {
            if (val.has_name() && val.name() == tensor_info->name()) {
                value_info = &val;
                break;
            }
        }
    }
    if (value_info != nullptr && value_info->has_type()) {
        if (!value_info->type().has_tensor_type()) {
            throw std::runtime_error("Unsupported value_info type");
        }
        tensor_meta_info.m_tensor_name = value_info->has_name() ? value_info->name() : "";
        const auto& value_type = value_info->type().tensor_type();
        if (value_type.has_shape()) {
            std::vector<int64_t> dims{};
            for (const auto& dim : value_type.shape().dim()) {
                if (dim.has_dim_value()) {
                    dims.push_back(dim.dim_value());
                } else {
                    dims.push_back(-1);
                }
            }
            tensor_meta_info.m_partial_shape = ov::PartialShape{dims};
        } else {
            tensor_meta_info.m_partial_shape = ov::PartialShape::dynamic();
        }
        if (value_type.has_elem_type()) {
            tensor_meta_info.m_element_type = get_ov_element_type(value_type.elem_type());
        } else {
            tensor_meta_info.m_element_type = ov::element::dynamic;
        }
    }
    if (tensor_info != nullptr) {
        tensor_meta_info.m_tensor_name = tensor_info->has_name() ? tensor_info->name() : "";
        tensor_meta_info.m_partial_shape =
            ov::PartialShape{std::vector<int64_t>{tensor_info->dims().begin(), tensor_info->dims().end()}};
        tensor_meta_info.m_element_type =
            tensor_info->has_data_type() ? get_ov_element_type(tensor_info->data_type()) : ov::element::dynamic;
        if (tensor_info->has_data_location() &&
            tensor_info->data_location() == TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL) {
            throw std::runtime_error("Unexpected usage of method for externally stored data");
        }
        if (tensor_info->has_raw_data()) {
            tensor_meta_info.m_tensor_data =
                static_cast<const uint8_t*>(static_cast<const void*>(tensor_info->raw_data().data()));
        }
        switch (tensor_info->data_type()) {
        case TensorProto_DataType::TensorProto_DataType_FLOAT:
            tensor_meta_info.m_tensor_data =
                static_cast<const uint8_t*>(static_cast<const void*>(tensor_info->float_data().data()));
            break;
        case TensorProto_DataType::TensorProto_DataType_INT32:
            tensor_meta_info.m_tensor_data =
                static_cast<const uint8_t*>(static_cast<const void*>(tensor_info->int32_data().data()));
            break;
        case TensorProto_DataType::TensorProto_DataType_INT64:
            tensor_meta_info.m_tensor_data =
                static_cast<const uint8_t*>(static_cast<const void*>(tensor_info->int64_data().data()));
            break;
        case TensorProto_DataType::TensorProto_DataType_UINT64:
            tensor_meta_info.m_tensor_data =
                static_cast<const uint8_t*>(static_cast<const void*>(tensor_info->uint64_data().data()));
            break;
        case TensorProto_DataType::TensorProto_DataType_DOUBLE:
            tensor_meta_info.m_tensor_data =
                static_cast<const uint8_t*>(static_cast<const void*>(tensor_info->double_data().data()));
            break;
        }
    }
    return tensor_meta_info;
}
}  // namespace

class DecoderProtoTensor : public ov::frontend::onnx::DecoderBaseTensor {
public:
    DecoderProtoTensor(const TensorProto* tensor_info,
                       const GraphProto* graph_def,
                       const int64_t input_idx,
                       const int64_t output_idx)
        // Probably, we may need to force it to 0/0
        : m_input_idx(input_idx),
          m_output_idx(output_idx) {
        m_tensor_meta_info = extract_tensor_meta_info(tensor_info, nullptr, graph_def);
    }
    DecoderProtoTensor(const ValueInfoProto* value_info,
                       const GraphProto* graph_def,
                       const int64_t input_idx,
                       const int64_t output_idx)
        : m_input_idx(input_idx),
          m_output_idx(output_idx) {
        m_tensor_meta_info = extract_tensor_meta_info(nullptr, value_info, graph_def);
    }

    const ov::frontend::onnx::TensorMetaInfo& get_tensor_info() const override {
        return *const_cast<const ov::frontend::onnx::TensorMetaInfo*>(&m_tensor_meta_info);
    }

    int64_t get_input_idx() const override {
        return m_input_idx;
    }

    int64_t get_output_idx() const override {
        return m_output_idx;
    }

    ov::Any get_attribute(const std::string& name) const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_attribute");
    }

    size_t get_input_size() const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_input_size");
    }

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_input_node");
    }

    const std::string& get_op_type() const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_op_type");
    }

    const std::string& get_op_name() const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_op_name");
    }

private:
    ov::frontend::onnx::TensorMetaInfo m_tensor_meta_info;
    int64_t m_input_idx, m_output_idx;
};

namespace {
static const std::string DEFAULT_DOMAIN = "";
static const std::string EMPTY_NAME = "";
static const std::string EMPTY_OP_TYPE = "";
}  // namespace

class DecoderProto : public ov::frontend::onnx::DecoderBaseOperation {
public:
    explicit DecoderProto(const NodeProto* node_def,
                          const uint64_t opset,
                          const GraphProto* graph_def,
                          const std::vector<const ov::frontend::onnx::TensorMetaInfo*>& input_info,
                          const std::vector<const ov::frontend::onnx::TensorMetaInfo*>& output_info)
        : m_node(node_def),
          m_opset(opset),
          m_graph(graph_def),
          m_input_info(input_info),
          m_output_info(output_info) {}

    size_t get_input_size() const override;
    size_t get_output_size() const override;

    std::string get_input_tensor_name(size_t idx) const override {
        return m_input_info.at(idx)->m_tensor_name;
    }
    ov::element::Type get_input_tensor_type(size_t idx) const override {
        return m_input_info.at(idx)->m_element_type;
    }
    std::string get_output_tensor_name(size_t idx) const override {
        return m_output_info.at(idx)->m_tensor_name;
    }
    ov::element::Type get_output_tensor_type(size_t idx) const override {
        return m_output_info.at(idx)->m_element_type;
    }
    const ov::frontend::onnx::TensorMetaInfo& get_input_tensor_info(size_t idx) const override {
        return *m_input_info.at(idx);
    }
    const ov::frontend::onnx::TensorMetaInfo& get_output_tensor_info(size_t idx) const override {
        return *m_output_info.at(idx);
    }

    ov::Any get_attribute(const std::string& name) const override;

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override;

    const std::string& get_op_type() const override;

    const std::string& get_op_name() const override;

    uint64_t get_op_set() const override {
        return m_opset;
    }

    const std::string& get_domain() const override {
        return (m_node->has_domain() && m_node->domain() != "ai.onnx" ? m_node->domain() : DEFAULT_DOMAIN);
    }

    bool has_attribute(const std::string& name) const {
        for (const auto& attr : m_node->attribute()) {
            if (attr.has_name() && attr.name() == name) {
                return true;
            }
        }
        return false;
    }

    void experimental_get_internal_structures(const void** node_def) const override {
        *node_def = m_node;
    }

private:
    // std::vector<::tensorflow::AttrValue> decode_attribute_helper(const std::string& name) const;
    const NodeProto* m_node;
    uint64_t m_opset;
    // For existence of NodeDef object corresponding to the main graph node,
    // GraphDef object must live in the memory
    const GraphProto* m_graph;
    // For existence of NodeDef object corresponding to the body graph node,
    // both GraphDef and FunctionDef objects must be alive in the memory
    // const std::shared_ptr<::tensorflow::FunctionDef> m_func_def;
    std::vector<const ov::frontend::onnx::TensorMetaInfo*> m_input_info, m_output_info;
};

ov::Any DecoderProto::get_attribute(const std::string& name) const {
    return "";
    /*
    auto attrs = decode_attribute_helper(name);
    if (attrs.empty()) {
        return {};
    }

    switch (attrs[0].value_case()) {
    case ::tensorflow::AttrValue::ValueCase::kB:
        return attrs[0].b();
    case ::tensorflow::AttrValue::ValueCase::kF:
        return attrs[0].f();
    case ::tensorflow::AttrValue::ValueCase::kS:
        return attrs[0].s();
    case ::tensorflow::AttrValue::ValueCase::kI:
        return attrs[0].i();
    case ::tensorflow::AttrValue::ValueCase::kShape: {
        const auto& tf_shape = attrs[0].shape();
        if (tf_shape.unknown_rank()) {
            return ov::PartialShape::dynamic();
        }
        auto shape_rank = tf_shape.dim_size();
        std::vector<ov::Dimension> dims(shape_rank);
        for (int i = 0; i < shape_rank; ++i) {
            dims[i] = static_cast<ov::Dimension::value_type>(tf_shape.dim(i).size());
        }
        return ov::PartialShape(dims);
    }

    case ::tensorflow::AttrValue::ValueCase::kType: {
        auto atype = attrs[0].type();

        if (atype == ::tensorflow::DT_COMPLEX64) {
            return ov::Any("DT_COMPLEX64");
        } else if (atype == ::tensorflow::DT_COMPLEX128) {
            return ov::Any("DT_COMPLEX128");
        } else {
            return get_ov_type(atype);
        }
    }

    case ::tensorflow::AttrValue::ValueCase::kList: {
        const auto& list = attrs[0].list();
        if (list.i_size())
            return std::vector<int64_t>(list.i().begin(), list.i().end());

        if (list.f_size())
            return std::vector<float>(list.f().begin(), list.f().end());

        if (list.s_size())
            return std::vector<std::string>(list.s().begin(), list.s().end());

        if (list.b_size())
            return std::vector<bool>(list.b().begin(), list.b().end());

        if (list.shape_size()) {
            auto shapes_size = list.shape_size();
            std::vector<ov::PartialShape> res(shapes_size);
            for (int shape_ind = 0; shape_ind < shapes_size; ++shape_ind) {
                auto shape = list.shape(shape_ind);
                if (shape.unknown_rank()) {
                    res[shape_ind] = ov::PartialShape::dynamic();
                } else {
                    auto shape_rank = shape.dim_size();
                    std::vector<ov::Dimension> dims(shape_rank);
                    for (int dim_ind = 0; dim_ind < shape_rank; ++dim_ind) {
                        dims[dim_ind] = static_cast<ov::Dimension::value_type>(shape.dim(dim_ind).size());
                    }
                    res[shape_ind] = dims;
                }
            }
            return res;
        }

        if (list.type_size()) {
            std::vector<ov::element::Type> res;
            for (int idx = 0; idx < list.type_size(); ++idx) {
                res.emplace_back(get_ov_type(list.type(idx)));
            }
            return res;
        }

        if (list.tensor_size() || list.func_size())
            FRONT_END_GENERAL_CHECK(
                false,
                "Conversion from Tensorflow to OpenVINO data type failed: List of tensors/functions type for '",
                name,
                "' attribute is not supported.");

        // If we got to this point it must mean we have empty list attribute
        return EmptyList();
    }

    case ::tensorflow::AttrValue::ValueCase::kTensor: {
        return unpack_tensor_proto(attrs[0].tensor());
    }
    case ::tensorflow::AttrValue::ValueCase::kPlaceholder:
        FRONT_END_GENERAL_CHECK(false,
                                "Conversion from Tensorflow to OpenVINO data type failed: Placeholder type for '",
                                name,
                                "' attribute is not supported.");
    case ::tensorflow::AttrValue::ValueCase::kFunc:
        // attrs[0].func() returns NameAttrList object from which
        // we retrieve the function name
        // Further, InputModel object is created for FunctionDef with this name
        // and is converted to ov::Model object.
        return attrs[0].func().name();
    default:
        FRONT_END_GENERAL_CHECK(false, "Conversion from Tensorflow to OpenVINO data type failed.");
    }
    */
}

size_t DecoderProto::get_input_size() const {
    return m_input_info.size();
}

size_t DecoderProto::get_output_size() const {
    return m_output_info.size();
}

void parse_producer_name(const std::string& producer_port_name,
                         std::string& producer_name,
                         std::string& producer_output_port_name,
                         size_t& producer_output_port_index) {
    return;
    // Body graph nodes may have two colons `:` input names, for example,
    // `TopKV2Name:indices:0` means that producer operation name is `TopKV2Name`
    // the middle name is output port name of the producer `indices` that means
    // the second output port of TopKV2 is used.
    // The first output port of TopKV2 is described as `TopKV2Name:values:0`
    auto first_colon = producer_port_name.find_first_of(":");
    auto last_colon = producer_port_name.find_last_of(":");
    if (first_colon != std::string::npos && first_colon < last_colon) {
        // we have at least two colons producer_name:output_port_name:port_idx
        producer_name = producer_port_name.substr(0, first_colon);
        auto port_id = producer_port_name.substr(last_colon + 1);
        auto port_name = producer_port_name.substr(first_colon + 1, last_colon - first_colon - 1);
        FRONT_END_GENERAL_CHECK(!port_id.empty() && std::all_of(port_id.begin(), port_id.end(), ::isdigit),
                                "Port id is not specified or not a number. Value: ",
                                port_id);
        producer_output_port_index = std::stoi(port_id);
        producer_output_port_name = std::move(port_name);
        return;
    } else if (first_colon != std::string::npos) {
        // just one colon case
        producer_name = producer_port_name.substr(0, first_colon);
        auto port_id = producer_port_name.substr(last_colon + 1);
        FRONT_END_GENERAL_CHECK(!port_id.empty() && std::all_of(port_id.begin(), port_id.end(), ::isdigit),
                                "Port id is not specified or not a number. Value: ",
                                port_id);
        producer_output_port_index = std::stoi(port_id);
        return;
    }
    producer_name = producer_port_name;
    producer_output_port_index = 0;
}

void DecoderProto::get_input_node(size_t input_port_idx,
                                  std::string& producer_name,
                                  std::string& producer_output_port_name,
                                  size_t& producer_output_port_index) const {
    const std::string producer_port_name = m_node->input(static_cast<int>(input_port_idx));
    parse_producer_name(producer_port_name, producer_name, producer_output_port_name, producer_output_port_index);
}

const std::string& DecoderProto::get_op_type() const {
    if (m_node->has_op_type()) {
        return m_node->op_type();
    } else {
        return EMPTY_OP_TYPE;
    }
}

const std::string& DecoderProto::get_op_name() const {
    if (m_node->has_name()) {
        return m_node->name();
    } else {
        return EMPTY_NAME;
    }
}
/*
std::vector<::tensorflow::AttrValue> DecoderProto::decode_attribute_helper(const std::string& name) const {
    auto attr_map = m_node_def->attr();
    if (attr_map.contains(name)) {
        auto value = m_node_def->attr().at(name);
        return {std::move(value)};
    } else {
        return {};
    }
}
*/

class GraphIteratorProto : public ov::frontend::onnx::GraphIterator {
    size_t node_index = 0;
    std::vector<uint8_t> m_data;
    std::vector<std::shared_ptr<ov::frontend::onnx::DecoderBase>> m_decoders{};
    std::shared_ptr<ModelProto> m_model;
    const GraphProto* m_graph{};

public:
    GraphIteratorProto() = default;
    explicit GraphIteratorProto(const std::string& path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    explicit GraphIteratorProto(const std::wstring& path);
#endif

    using Ptr = std::shared_ptr<GraphIteratorProto>;

    ~GraphIteratorProto() = default;

    /// Verifies file is supported
    template <typename T>
    static bool is_supported(const std::basic_string<T>& path) {
        FRONT_END_GENERAL_CHECK(util::file_exists(path),
                                "Could not open the file: \"",
                                util::path_to_string(path),
                                '"');
        try {
            std::streamsize file_size = util::file_size(path);
            // Skip files which less than size of file identifier
            if (file_size < 1) {
                return false;
            }
#if defined(__MINGW32__) || defined(__MINGW64__)
            std::ifstream tflite_stream(std::filesystem::path(path), std::ios::in | std::ifstream::binary);
#else
            std::ifstream tflite_stream(path, std::ios::in | std::ifstream::binary);
#endif
            // the model usually starts with a 0x08 byte indicating the ir_version value
            // so this checker expects at least 3 valid ONNX keys to be found in the validated model
            const size_t EXPECTED_FIELDS_FOUND = 3u;
            std::unordered_set<::onnx::Field, std::hash<int>> onnx_fields_found = {};
            try {
                while (!model.eof() && onnx_fields_found.size() < EXPECTED_FIELDS_FOUND) {
                    const auto field = ::onnx::decode_next_field(model);

                    if (onnx_fields_found.count(field.first) > 0) {
                        // if the same field is found twice, this is not a valid ONNX model
                        return false;
                    } else {
                        onnx_fields_found.insert(field.first);
                        ::onnx::skip_payload(model, field.second);
                    }
                }

                return onnx_fields_found.size() == EXPECTED_FIELDS_FOUND;
            } catch (...) {
                return false;
            }
        } catch (...) {
            return false;
        }
    }

    /// Set iterator to the start position
    void reset() override {
        node_index = 0;
    }

    size_t size() const override {
        return m_decoders.size();
    }

    /// Moves to the next node in the graph
    void next() override {
        node_index++;
    }

    bool is_end() const override {
        return node_index >= m_decoders.size();
    }

    /// Return Decoder for the current node that iterator points to
    std::shared_ptr<ov::frontend::onnx::DecoderBase> get_decoder() const override;

    /// \brief Returns the number of sub-graphs that can be enumerated with get_subgraph
    size_t get_subgraph_size() const override;

    /// \brief Returns iterator for a subgraph created on demand
    /// If there is no query for specific sub-graph iterator shouldn't be created
    /// idx should be in range 0..get_subgraph_size()-1
    std::shared_ptr<ov::frontend::onnx::GraphIterator> get_subgraph(size_t idx) const override;

    std::int64_t get_opset_version(const std::string& domain) const override;
};

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

GraphIteratorProto::GraphIteratorProto(const std::wstring& path)
    : GraphIteratorProto(ov::util::wstring_to_string(path)) {}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

GraphIteratorProto::GraphIteratorProto(const std::string& path) {
    std::ifstream model_file(path, std::ios::binary | std::ios::in);
    FRONT_END_GENERAL_CHECK(model_file && model_file.is_open(), "Model file does not exist: ", path);

    m_model = std::make_shared<ModelProto>();
    m_model->ParseFromIstream(&model_file);
    model_file.close();
    if (m_model->has_graph()) {
        m_graph = &m_model->graph();
    } else {
        m_graph = nullptr;
        return;
    }
    std::map<std::string, std::shared_ptr<DecoderProtoTensor>> tensors{};
    for (const auto& value : m_graph->input()) {
        auto tensor = std::make_shared<DecoderProtoTensor>(&value, m_graph, 0, -1);
        m_decoders.push_back(tensor);
        if (tensors.count(tensor->get_tensor_info().m_tensor_name) > 0) {
            throw std::runtime_error("Tensor already exists");
        }
        tensors[tensor->get_tensor_info().m_tensor_name] = tensor;
    }
    for (const auto& value : m_graph->output()) {
        m_decoders.push_back(std::make_shared<DecoderProtoTensor>(&value, m_graph, -1, 0));
        auto tensor = std::make_shared<DecoderProtoTensor>(&value, m_graph, -1, 0);
        m_decoders.push_back(tensor);
        if (tensors.count(tensor->get_tensor_info().m_tensor_name) > 0) {
            throw std::runtime_error("Tensor already exists");
        }
        tensors[tensor->get_tensor_info().m_tensor_name] = tensor;
    }
    for (const auto& initializer : m_graph->initializer()) {
        auto tensor = std::make_shared<DecoderProtoTensor>(&initializer, m_graph, 0, 0);
        m_decoders.push_back(tensor);
        if (tensors.count(tensor->get_tensor_info().m_tensor_name) > 0) {
            throw std::runtime_error("Tensor already exists");
        }
        tensors[tensor->get_tensor_info().m_tensor_name] = tensor;
    }
    for (const auto& node : m_graph->node()) {
        std::vector<const ov::frontend::onnx::TensorMetaInfo*> input_tensors{};
        std::vector<const ov::frontend::onnx::TensorMetaInfo*> output_tensors{};
        input_tensors.reserve(node.input_size());
        output_tensors.reserve(node.output_size());
        for (const auto& name : node.input()) {
            if (tensors.count(name) == 0) {
                throw std::runtime_error("Input tensor isn't found for node \"" + name + "\"");
            }
            if (name != "") {
                input_tensors.push_back(&tensors[name]->get_tensor_info());
            }
        }
        for (const auto& name : node.output()) {
            if (name != "") {
                const auto& found_tensor = tensors.find(name);
                if (found_tensor == tensors.end()) {
                    const auto& initializer = std::find_if(m_graph->initializer().begin(),
                                                           m_graph->initializer().end(),
                                                           [&name](const TensorProto& value) {
                                                               return value.has_name() && value.name() == name;
                                                           });
                    std::shared_ptr<DecoderProtoTensor> tensor{nullptr};
                    if (initializer != m_graph->initializer().end()) {
                        tensor = std::make_shared<DecoderProtoTensor>(&*initializer, m_graph, 0, 0);
                    } else {
                        const auto& value_info = std::find_if(m_graph->value_info().begin(),
                                                              m_graph->value_info().end(),
                                                              [&name](const ValueInfoProto& value) {
                                                                  return value.has_name() && value.name() == name;
                                                              });
                        if (value_info != m_graph->value_info().end())
                            tensor = std::make_shared<DecoderProtoTensor>(&*value_info, m_graph, 0, 0);
                    }
                    if (tensor == nullptr) {
                        throw std::runtime_error("Tensor not found \"" + name + "\"");
                    }
                    tensors[name] = tensor;
                    output_tensors.push_back(&tensor->get_tensor_info());
                } else {
                    output_tensors.push_back(&found_tensor->second->get_tensor_info());
                }
            }
        }
        const std::string& domain = node.has_domain() && node.domain() != "ai.onnx" ? node.domain() : DEFAULT_DOMAIN;
        int64_t opset = get_opset_version(domain);
        if (opset == -1) {
            throw std::runtime_error("Operation version isn't found");
        }
        auto decoder_node =
            std::make_shared<DecoderProto>(&node, static_cast<uint64_t>(opset), m_graph, input_tensors, output_tensors);
        const auto& asd = decoder_node->get_domain();
        std::cout << asd << std::endl;
        m_decoders.push_back(decoder_node);
    }
}

size_t GraphIteratorProto::get_subgraph_size() const {
    return 0;
}

std::shared_ptr<ov::frontend::onnx::GraphIterator> GraphIteratorProto::get_subgraph(size_t idx) const {
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
    /*
    FRONT_END_GENERAL_CHECK(0 == idx, "There is no subgraph with idx ", idx);
    auto iterator = std::make_shared<GraphIteratorProto>();
    iterator->node_index = 0;
    iterator->m_model = m_model;
    iterator->m_subgraphs = {};  // TODO: check if we need to pass all sub-graphs here (while in a while situation)
    iterator->m_graph = m_subgraphs[idx];
    const auto operators = iterator->m_graph->operators();
    auto operators_vec = std::vector<const tflite::Operator*>{operators->begin(), operators->end()};
    iterator->m_nodes.assign(operators_vec.begin(), operators_vec.end());
    auto outputs = iterator->m_graph->outputs();
    auto inputs = iterator->m_graph->inputs();
    iterator->m_nodes.insert(iterator->m_nodes.begin(), outputs->begin(), outputs->end());
    iterator->m_nodes.insert(iterator->m_nodes.begin(), inputs->begin(), inputs->end());
    return iterator;
    */
}

std::shared_ptr<ov::frontend::onnx::DecoderBase> GraphIteratorProto::get_decoder() const {
    return m_decoders[node_index];
    /*
    auto tensors = m_graph->tensors();

    if (is_op) {
        auto node = m_nodes[node_index].as<const tflite::Operator*>();
        auto buffers = m_model->buffers();

        std::map<size_t, TensorInfo> input_info = {}, output_info = {};
        size_t i = 0;
        for (auto input : *node->inputs()) {
            if (input == -1)
                continue;
            auto buffer = (*buffers)[(*tensors)[input]->buffer()];
            auto tensor = (*tensors)[input];
            input_info[i++] = TensorInfo{tensor, buffer};
        }
        i = 0;
        for (auto output : *node->outputs()) {
            auto buffer = (*buffers)[(*tensors)[output]->buffer()];
            auto tensor = (*tensors)[output];
            output_info[i++] = TensorInfo{tensor, buffer};
        }
        auto op_codes = m_model->operator_codes();
        auto operator_code = (*op_codes)[node->opcode_index()];
        std::string type;
        if (operator_code->deprecated_builtin_code() <
            tflite::BuiltinOperator::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES) {
            type = tflite::EnumNamesBuiltinOperator()[operator_code->deprecated_builtin_code()];
        } else {
            type = tflite::EnumNamesBuiltinOperator()[operator_code->builtin_code()];
        }
        if (type == "CUSTOM") {
            type = operator_code->custom_code()->str();
        }
        auto name = std::to_string(node_index - m_graph->inputs()->size() - m_graph->outputs()->size());
        return std::make_shared<DecoderFlatBuffer>(node, type, name, input_info, output_info);
    } else {
        auto tensor_id = m_nodes[node_index].as<int32_t>();
        auto tensor = (*tensors)[tensor_id];
        auto info = TensorInfo{tensor, nullptr};
        auto inputs = m_graph->inputs();
        auto outputs = m_graph->outputs();

        auto input_it = std::find(inputs->begin(), inputs->end(), tensor_id);
        auto output_it = std::find(outputs->begin(), outputs->end(), tensor_id);
        int64_t input_idx =
            input_it == inputs->end() ? -1 : static_cast<int64_t>(std::distance(inputs->begin(), input_it));
        int64_t output_idx =
            output_it == outputs->end() ? -1 : static_cast<int64_t>(std::distance(outputs->begin(), output_it));
        return std::make_shared<DecoderFlatBufferTensors>(info, input_idx, output_idx);
    }
    */
}

std::int64_t GraphIteratorProto::get_opset_version(const std::string& domain) const {
    // copy the opsets and sort them (descending order)
    // then return the version from the first occurrence of a given domain
    auto opset_imports = m_model->opset_import();
    std::sort(std::begin(opset_imports),
              std::end(opset_imports),
              [](const OperatorSetIdProto& lhs, const OperatorSetIdProto& rhs) {
                  return lhs.version() > rhs.version();
              });

    for (const auto& opset_import : opset_imports) {
        if (domain == opset_import.domain()) {
            return opset_import.version();
        }
    }

    return -1;
}

}  // namespace test_iterator

#include <openvino/openvino.hpp>

TEST_P(FrontEndLoadFromTest, testLoadUsingTestGraphIterator) {
    const auto path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "addmul_abc.onnx"})
            .string();

    ov::frontend::FrontEnd::Ptr fe;

    auto iter = std::make_shared<test_iterator::GraphIteratorProto>(path);

    auto graph_iter = std::dynamic_pointer_cast<ov::frontend::onnx::GraphIterator>(iter);
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework("onnx"))
        << "Could not create the ONNX FE using a pointer GraphIterator";
    ASSERT_NE(m_frontEnd, nullptr);

    ASSERT_EQ(m_frontEnd->supported(graph_iter), true);

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(graph_iter)) << "Could not load the model";
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel)) << "Could not convert the model to OV representation";
    ASSERT_NE(model, nullptr);

    ov::serialize(model, "e:/test.xml");

    ASSERT_EQ(model->get_ordered_ops().size(), 3);
}
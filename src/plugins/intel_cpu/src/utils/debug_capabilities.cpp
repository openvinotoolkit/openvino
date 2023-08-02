
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <common/primitive_desc_iface.hpp>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include "memory_desc/blocked_memory_desc.h"
#include "onednn/iml_type_mapper.h"
#include "debug_capabilities.h"
#ifdef CPU_DEBUG_CAPS

#include "node.h"
#include "edge.h"
#include "graph.h"
#include <iomanip>
#include <cfloat>
#include <climits>
#include "nodes/input.h"
#include "nodes/eltwise.h"
#include "snippets/op/subgraph.hpp"
#include <ie_ngraph_utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>

namespace dnnl {
namespace impl {
std::ostream &operator<<(std::ostream &ss, const primitive_attr_t *attr);
std::ostream &operator<<(std::ostream &ss, alg_kind_t alg);
}
}

namespace ov {
namespace intel_cpu {

namespace {
    size_t replace_all(std::string & inout, std::string what, std::string with) {
        std::size_t count{};
        for (std::string::size_type pos{}; inout.npos != (pos = inout.find(what.data(), pos, what.length()));
             pos += with.length(), ++count) {
            inout.replace(pos, what.length(), with.data(), with.length());
        }
        return count;
    }
}

DebugLogEnabled::DebugLogEnabled(const char* file, const char* func, int line, const char* name) {
    // check ENV
    const char* p_filters = std::getenv("OV_CPU_DEBUG_LOG");
    if (!p_filters) {
        enabled = false;
        return;
    }

    // extract file name from __FILE__
    std::string file_path(file);
    std::string file_name(file);
    auto last_sep = file_path.find_last_of('/');
    if (last_sep == std::string::npos)
        last_sep = file_path.find_last_of('\\');
    if (last_sep != std::string::npos)
        file_name = file_path.substr(last_sep + 1);

    std::string file_name_with_line = file_name + ":" + std::to_string(line);
    tag = file_name_with_line + " " + func + "()";
    if (name != nullptr) {
        tag += " ";
        tag += name;
    }

    // check each filter patten:
    bool filter_match_action = true;
    if (p_filters[0] == '-') {
        p_filters++;
        filter_match_action = false;
    }

    bool match = false;
    const char* p0 = p_filters;
    const char* p1 = p0;
    while (*p0 != 0) {
        p1 = p0;
        while (*p1 != ';' && *p1 != 0)
            ++p1;
        std::string pattern(p0, p1 - p0);
        if (pattern == file_name || pattern == func || pattern == tag || pattern == file_name_with_line ||
            (name != nullptr && pattern == name)) {
            match = true;
            break;
        }
        p0 = p1;
        if (*p0 == ';')
            ++p0;
    }

    if (match)
        enabled = filter_match_action;
    else
        enabled = !filter_match_action;
}

void DebugLogEnabled::break_at(const std::string & log) {
    static const char* p_brk = std::getenv("OV_CPU_DEBUG_LOG_BRK");
    if (p_brk && log.find(p_brk) != std::string::npos) {
        std::cout << "[ DEBUG ] " << " Debug log breakpoint hit" << std::endl;
#if defined(_MSC_VER)
        __debugbreak();
#elif defined(__APPLE__) || defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
       __builtin_trap();
#else
        asm("int3");
#endif
    }
}

std::ostream & operator<<(std::ostream & os, const MemoryDesc& desc) {
    os << desc.getShape().toString()
       << " " << desc.getPrecision().name()
       << " " << desc.serializeFormat();
    return os;
}

std::ostream & operator<<(std::ostream & os, const dnnl::primitive_attr& attr) {
    return dnnl::impl::operator<<(os, attr.get());
}

std::ostream & operator<<(std::ostream & os, const dnnl::algorithm& alg) {
    return dnnl::impl::operator<<(os, convert_to_c(alg));
}

std::ostream & operator<<(std::ostream & os, const NodeDesc& desc) {
    std::stringstream ss;
    ss << "  " << impl_type_to_string(desc.getImplementationType()) << "(";
    const char * sep = "";
    for (auto & conf : desc.getConfig().inConfs) {
        ss << sep << *conf.getMemDesc();
        if (conf.inPlace() >= 0) ss << " inPlace:" << conf.inPlace();
        if (conf.constant()) ss << " constant";
        sep = ",";
    }
    ss << ") -> (";
    sep = "";
    for (auto & conf : desc.getConfig().outConfs) {
        ss << sep << *conf.getMemDesc();
        if (conf.inPlace() >= 0) ss << " inPlace:" << conf.inPlace();
        if (conf.constant()) ss << " constant";
        sep = ",";
    }
    ss << ")" << std::endl;
    auto str = ss.str();
    replace_all(str, "0 - ?", "?");
    os << str;
    return os;
}

std::ostream & operator<<(std::ostream & os, const Edge& edge) {
    os << edge.getParent()->getName() << "[" << edge.getInputNum() << "]->"
       << edge.getChild()->getName() << "[" << edge.getOutputNum() << "]";
    return os;
}

template<typename DT>
void dump_tensor(std::ostream& os, void * ptr, const VectorDims& dims) {
    DT * p = reinterpret_cast<DT*>(ptr);
    auto rank = dims.size();
    const char * sep = "";
    os << "{";

    if (rank > 1) os << "\n\t";

    auto last_dim_size = dims[rank - 1];

    auto sz = shape_size(dims);
    std::stringstream ss;
    int lines = 0;
    for (size_t i = 0; i < sz; i++) {
        if (ss.tellp() < 256)
            ss << p[i] << ",";

        if ((i % last_dim_size) == (last_dim_size - 1)) {
            os << lines << " : " <<  ss.str() << "...\n\t";
            ss.str("");
            lines++;
            if (lines > 16) {
                os << "... ... ... ... \n\t";
                break;
            }
        }
    }
    os << "}";
}

std::ostream& operator<<(std::ostream& os, const Memory& mem) {
    auto nbytes = mem.getSize();
    void* ptr = mem.getData();
    auto& dims = mem.getStaticDims();
    switch (mem.getDataType()) {
    case dnnl::memory::data_type::bf16:
        dump_tensor<ov::bfloat16>(os, ptr, dims);
        break;
    case dnnl::memory::data_type::f32:
        dump_tensor<float>(os, ptr, dims);
        break;
    case dnnl::memory::data_type::s32:
        dump_tensor<int32_t>(os, ptr, dims);
        break;
    case dnnl::memory::data_type::s8:
        dump_tensor<int8_t>(os, ptr, dims);
        break;
    case dnnl::memory::data_type::u8:
        dump_tensor<uint8_t>(os, ptr, dims);
        break;
    }
    return os;
}

bool print_mem_content(std::ostream & os, Node& node) {
    bool printed = false;
    if (node.getType() == intel_cpu::Type::Input && node.isConstant()) {
        if (auto input_node = reinterpret_cast<intel_cpu::node::Input *>(&node)) {
            auto pmem = input_node->getMemoryPtr();
            void * data = pmem->getData();
            auto shape = pmem->getDesc().getShape().getDims();
            if (shape_size(shape) <= 8) {
                os << "[";
                auto type = InferenceEngine::details::convertPrecision(pmem->getDesc().getPrecision());
                auto tensor = std::make_shared<ngraph::runtime::HostTensor>(type, shape, data);
                auto constop = std::make_shared<ngraph::op::Constant>(tensor);
                const char * comma = "";
                for (auto & v : constop->get_value_strings()) {
                    os << comma << v;
                    comma = ",";
                }
                os << "]";
                printed = true;
            }
        }
    }
    return printed;
}

std::ostream & operator<<(std::ostream & os, const Node &c_node) {
    Node & node = const_cast<Node &>(c_node);
    const int align_col = 50;
    const char * comma = "";
    auto node_id = [](Node & node) {
        auto id = node.getName();
        if (id.size() > 20)
            return node.getTypeStr() + "_" + std::to_string(node.getExecIndex());
        return id;
    };
    auto is_single_output_port = [](Node & node) {
        for (auto & e : node.getChildEdges()) {
            auto edge = e.lock();
            if (!edge) continue;
            if (edge->getInputNum() != 0)
                return false;
        }
        return true;
    };

    auto nodeDesc = node.getSelectedPrimitiveDescriptor();
    std::stringstream leftside;

    int num_output_port = 0;
    for (const auto& wptr : node.getChildEdges()) {
        auto edge = wptr.lock();
        if (num_output_port < edge->getInputNum() + 1)
            num_output_port = edge->getInputNum() + 1;
    }

    if (num_output_port) {
        if (num_output_port > 1) leftside << "(";
        comma = "";
        for (int i = 0; i < num_output_port; i++) {
            bool b_ouputed = false;
            auto edge = node.getChildEdgeAt(i);
            if (edge->getStatus() != Edge::Status::NotAllocated) {
                auto ptr = edge->getMemoryPtr();
                if (ptr) {
                    auto desc = &(ptr->getDesc());
                    auto shape_str = desc->getShape().toString();
                    replace_all(shape_str, " ", "");
                    leftside << comma << desc->getPrecision().name()
                                << "_" << desc->serializeFormat()
                                << "_" << shape_str
                                << "_" << ptr->getData();
                    b_ouputed = true;
                } else {
                    leftside << "(empty)";
                }
            }
            if (!b_ouputed && nodeDesc && i < static_cast<int>(nodeDesc->getConfig().outConfs.size())) {
                auto desc = nodeDesc->getConfig().outConfs[i].getMemDesc();
                auto shape_str = desc->getShape().toString();
                replace_all(shape_str, "0 - ?", "?");
                replace_all(shape_str, " ", "");
                leftside << comma << desc->getPrecision().name()
                            << "_" << desc->serializeFormat()
                            << "_" << shape_str;
                b_ouputed = true;
            }
            if (!b_ouputed) {
                leftside << comma << "???";
            }
            comma = ",";
        }
        if (num_output_port > 1) leftside << ")";
    } else if (nodeDesc) {
        // output Desc is enough since input is always in consistent
        // with output.
        /*
        auto& inConfs = nodeDesc->getConfig().inConfs;
        if (!inConfs.empty()) {
            os << " in:[";
            for (auto& c : inConfs) {
                os << c.getMemDesc()->getPrecision().name()
                        << c.getMemDesc()->
                        << "/" << c.getMemDesc()->serializeFormat()
                        << "; ";
            }
            os << "]";
        }*/

        auto& outConfs = nodeDesc->getConfig().outConfs;
        if (!outConfs.empty()) {
            if (outConfs.size() > 1) leftside << "(";
            comma = "";
            for (auto& c : outConfs) {
                auto shape_str = c.getMemDesc()->getShape().toString();
                replace_all(shape_str, "0 - ?", "?");
                leftside << comma << c.getMemDesc()->getPrecision().name()
                            << "_" << c.getMemDesc()->serializeFormat()
                            << "_" << shape_str;
                comma = ",";
            }
            if (outConfs.size() > 1) leftside << ")";
        }
    } else {
        // no SPD yet, use orginal shapes
        comma = "";
        for (size_t i = 0; i < node.getOriginalOutputPrecisions().size(); i++) {
            auto shape = node.getOutputShapeAtPort(i);
            std::string prec_name = "Undef";
            prec_name = node.getOriginalOutputPrecisionAtPort(i).name();
            auto shape_str = shape.toString();
            replace_all(shape_str, "0 - ?", "?");
            leftside << comma << prec_name
                        << "_" << shape_str;
            comma = ",";
        }
    }
    leftside << "  " << node_id(node) << " = ";
    os << "#" << node.getExecIndex() << " :" << std::right << std::setw(align_col) << leftside.str();
    os << std::left << node.getTypeStr();
    if (node.getAlgorithm() != Algorithm::Default)
        os << "." << algToString(node.getAlgorithm());
    os << " (";

    comma = "";
    for (size_t port = 0; port < node.getParentEdges().size(); ++port) {
        // find the Parent edge connecting to port
        for (const auto & e : node.getParentEdges()) {
            auto edge = e.lock();
            if (!edge) continue;
            if (edge->getOutputNum() != static_cast<int>(port)) continue;
            auto n = edge->getParent();
            os << comma;
            if (!print_mem_content(os, *edge->getParent())) {
                os << node_id(*edge->getParent());
                auto ptr = edge->getMemoryPtr();
                if (ptr) {
                    os << "(" << ptr->getData() << ")";
                }
                if (!is_single_output_port(*n))
                    os << "[" << edge->getInputNum() << "]";
            }
            comma = ",";
            break;
        }
    }

    print_mem_content(os, node);

    // additional properties
    if (node.getType() == intel_cpu::Type::Eltwise) {
        auto eltwise_node = reinterpret_cast<intel_cpu::node::Eltwise *>(&node);
        os << " | Alpha=" << eltwise_node->getAlpha()
        << ", Beta=" << eltwise_node->getBeta()
        << ", Gamma=" << eltwise_node->getGamma()
        << ", BroadcastingPolicy=";

        switch (eltwise_node->getBroadcastingPolicy()) {
            case intel_cpu::node::Eltwise::BroadcastingPolicy::PerChannel:
                os << "PerChannel";
                break;
            case intel_cpu::node::Eltwise::BroadcastingPolicy::PerTensor:
                os << "PerTensor";
                break;
            default:
                os << "?";
        }
    }

    os << ")  ";
    os << " " << node.getPrimitiveDescriptorType();

    // last line(s): fused layers
    os << " orglayers:" << node.getOriginalLayers();

    if (node.PerfCounter().count()) {
        os << " latency:" << node.PerfCounter().avg() << "(us) x" << node.PerfCounter().count();
    }

    for (auto & fn : node.getFusedWith()) {
        os << "\n\t  FusedWith: " << *fn;
    }

    // output result values
    /*
    for (int i = 0; i < num_output_port; i++) {
        auto edge = node.getChildEdgeAt(i);
        if (edge->getStatus() != Edge::Status::NotAllocated) {
            auto ptr = edge->getMemoryPtr();
            if (ptr->getDesc().getShape().isStatic()) {
                os << "\n\t ChildEdgeAt(" << i << "): " << *ptr << std::endl;
            }
        }
    }
    */
    // primArgs
    /*
    if (node.primArgs.size()) {
        comma = "";
        os << " primArgs={";
        for (auto & it : node.primArgs) {
            void * ptr = it.second.map_data();
            it.second.unmap_data(ptr);
            auto arg_id = it.first;
            os << comma << arg_id << ":" << ptr;
            comma = ",";
        }
        os << "}";
    }*/

    return os;
}

template <typename T>
void stream_output_scalar(std::ostream& os, const T& value) {
    if (std::is_floating_point<T>::value) {
        if (std::isnan(value)) {
            os << "NAN";
        } else if (std::isinf(value)) {
            os << (value > 0 ? "INFINITY" : "-INFINITY");
        } else if (value == FLT_MIN) {
                os << "FLT_MIN";
        } else if (value == -FLT_MIN) {
                os << "-FLT_MIN";
        } else if (value == FLT_MAX) {
                os << "FLT_MAX";
        } else if (value == -FLT_MAX) {
                os << "-FLT_MAX";
        } else {
            std::stringstream ss;
            ss << value;
            auto strv = ss.str();
            os << strv;
            if (strv.find(".") == std::string::npos && strv.find("e") == std::string::npos)
            os << ".0";
            if (std::is_same<T, float>::value)
            os << "f";
        }
    } else {
        os << value;
    }
}

// Print complex data structures in a textualized form to the console is an efficient way to investigate them
std::ostream & operator<<(std::ostream & os, const ov::intel_cpu::Graph& g) {
    os << "ov::intel_cpu::Graph " << g.GetName() << " {" << std::endl;
    for (auto &graphNode : g.GetNodes()) {
        os << *graphNode << std::endl;
    }
    os << "};" << std::endl;
    return os;
}

class OstreamAttributeVisitor : public ngraph::AttributeVisitor {
    std::ostream & os;
    const char * sep = "";

public:
    OstreamAttributeVisitor(std::ostream & os) : os(os) {}

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        if (auto a = ov::as_type<ov::AttributeAdapter<std::set<std::string>>>(&adapter)) {
            const auto& value = join(a->get());
            append_attribute(name.c_str(), value.c_str());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::vector<ov::element::Type>>>(&adapter)) {
            const auto& value = join(a->get());
            append_attribute(name.c_str(), value.c_str());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            const auto& value = a->get();
            append_attribute(name.c_str(), value.to_string().c_str());
        } else {
            append_attribute(name.c_str(), "?");
        }
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& adapter) override {
        append_attribute(name.c_str(), adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& adapter) override {
        append_attribute(name.c_str(), adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        append_attribute(name.c_str(), adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
        append_attribute(name.c_str(), adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<int32_t>& adapter) override {
        append_attribute(name.c_str(), adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<float>& adapter) override {
        append_attribute(name.c_str(), adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int>>& adapter) override {
        const auto& value = join(adapter.get());
        append_attribute(name.c_str(), value.c_str(), ' ');
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        const auto& value = join(adapter.get());
        append_attribute(name.c_str(), value.c_str(), ' ');
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        const auto& value = join(adapter.get());
        append_attribute(name.c_str(), value.c_str(), ' ');
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        const auto& value = join(adapter.get());
        append_attribute(name.c_str(), value.c_str(), ' ');
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        const auto& value = join(adapter.get());
        append_attribute(name.c_str(), value.c_str());
    }

    void append_attribute(const char * name, const std::string& value) {
        os << sep << "{\"" << name << "\", \"" << value << "\"}";
        sep = ", ";
    }

    void append_attribute(const char * name, const char * value, const char quote = '"') {
        os << sep << "{\"" << name << "\", " << quote << value << quote << "}";
        sep = ", ";
    }

    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
    void append_attribute(const char * name, const T& value) {
        os << sep << "{\"" << name << "\", ";
        stream_output_scalar(os, value);
        os << "}";
        sep = ", ";
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        append_attribute(name.c_str(), "Model");
    }

    template<class Container>
    inline std::string join(const Container& strs) {
        std::stringstream ss;
        ss << "{" << ov::intel_cpu::join(strs, ',') << "}";
        return ss.str();
    }
};

// so far we can only show correct delta on single stream configuration, which
// is enough for debug purpose
std::ostream& operator<<(std::ostream& os, const PrintableDelta& d) {
    double us_last = d.us_last;
    double us_all = d.us_all;
    os << "[+ " << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(3) << us_last / 1000 << "/"
       << us_all / 1000 << " ms]";
    return os;
}

std::ostream & operator<<(std::ostream & os, const Edge::ReorderStatus reorderStatus) {
    switch (reorderStatus) {
    case Edge::ReorderStatus::Regular: os << "Regular"; break;
    case Edge::ReorderStatus::Optimized: os << "Optimizer"; break;
    case Edge::ReorderStatus::No: os << "No"; break;
    }
    return os;
}

std::ostream & operator<<(std::ostream & os, const dnnl::primitive_desc& desc) {
    os << desc.get()->info();
    return os;
}

std::ostream & operator<<(std::ostream & os, const dnnl::memory::desc& desc) {
    char sep = '(';
    os << "dims:";
    const auto& ndims = desc.get()->ndims;
    const auto& dims = desc.get()->dims;
    for (int i = 0; i < ndims; i++) {
        os << sep << dims[i];
        sep = ',';
    }
    os << ")";

    const auto& strides = desc.get()->format_desc.blocking.strides;
    sep = '(';
    os << "strides:";
    for (int i = 0; i < ndims; i++) {
        os << sep << strides[i];
        sep = ',';
    }
    os << ")";

    const auto& inner_blks  = desc.get()->format_desc.blocking.inner_blks;
    const auto& inner_nblks = desc.get()->format_desc.blocking.inner_nblks;
    const auto& inner_idxs  = desc.get()->format_desc.blocking.inner_idxs;

    for (int i = 0; i < inner_nblks; i++) {
        os << inner_blks[i] << static_cast<char>('a' + inner_idxs[i]);
    }

    os << " " << dnnl_dt2str(desc.get()->data_type);
    return os;
}

std::ostream& operator<<(std::ostream& os, const impl_desc_type impl_type) {
    os <<  impl_type_to_string(impl_type);
    return os;
}

std::ostream & operator<<(std::ostream & os, const dnnl::memory::data_type dtype) {
    os << " " << dnnl_dt2str(static_cast<dnnl_data_type_t>(dtype));
    return os;
}

std::ostream & operator<<(std::ostream & os, const dnnl::memory::format_tag format_tag) {
    const auto c_format_tag = dnnl::memory::convert_to_c(format_tag);
    os << dnnl_fmt_tag2str(c_format_tag);
    return os;
}

DumpModel::DumpModel(const std::string& _file_name) {
    static int dump_index = 0;
    file_name = "dump_model" + std::to_string(dump_index) + "_" + _file_name;
    dump_index++;
}

void stream_output_constant(std::ostream & os, op::v0::Constant * constop) {
    auto shape = constop->get_shape();
    auto is_scalar = shape.size() == 0;
    auto total_size = shape_size(shape);
    auto eletype = constop->get_output_element_type(0);
    const char * sep = "";
    os << (is_scalar? "" : "{");
    if (eletype == ov::element::f32) {
        for (float value : constop->get_vector<float>()) {
            os << sep;
            stream_output_scalar(os, value);
            sep = ", ";
        }
    } else {
        for (auto v : constop->get_value_strings()) {
            os << sep << v;
            sep = ", ";
        }
    }
    os << (is_scalar? "" : "}");
}

void DumpModel::dump_cpp_style(std::ostream & os, const std::shared_ptr<ov::Model>& model) {
    const ov::Model& f = *model;
    std::string prefix = "";
    std::string tag = "";
    std::string sep = "";
    os << prefix;
    for (auto op : f.get_results()) {
        os << sep << op->get_name();
        sep = ",";
    }
    os << " " << f.get_friendly_name() << "(\n" << prefix;
    for (auto op : f.get_parameters()) {
        os << "    " << tag << op->get_friendly_name() << ",\n" << prefix;
    }
    os << ") {\n";

    // collect all scalar & short 1D vectors for literal-style display
    std::map<std::shared_ptr<ov::Node>, std::string> literal_consts;
    for (auto op : f.get_ordered_ops()) {
        const auto & type_info = op->get_type_info();
        if (auto constop = std::dynamic_pointer_cast<op::v0::Constant>(op)) {
            // only i32/f32 type const literal can be parsed by C++ compiler
            if (constop->get_output_element_type(0) != ov::element::i32 &&
                constop->get_output_element_type(0) != ov::element::f32)
                continue;
            auto shape = constop->get_shape();
            if (shape.size() > 1) continue;
            auto is_scalar = shape.size() == 0;
            if (shape_size(constop->get_shape()) > 8) continue;

            // literal construct
            std::stringstream ss;
            stream_output_constant(ss, constop.get());
            literal_consts[op] = ss.str();
        }
    }

    auto get_output_values_info = [](std::shared_ptr<ov::Node> & op) {
        std::stringstream ss;
        const char *sep = "";
        for (int i = 0; i < op->get_output_size(); i++) {
            ss << sep << op->get_output_element_type(i) << op->get_output_partial_shape(i);
            sep = " ";
        }
        return ss.str();
    };

    // change name convension
    std::map<ov::Node*, std::string> opname;
    std::map<std::string, int> opname_count;
    for (auto op : f.get_ordered_ops()) {
        auto name = op->get_friendly_name();
        std::replace(name.begin(), name.end(), '\\', '_');
        std::replace(name.begin(), name.end(), '/', '_');
        std::replace(name.begin(), name.end(), '.', '_');
        std::replace(name.begin(), name.end(), '[', '_');
        std::replace(name.begin(), name.end(), ']', '_');
        std::replace(name.begin(), name.end(), '-', 'n');

        int idx = 0;
        if (opname_count.count(name)) {
            idx = opname_count[name] + 1;
        }
        opname_count[name] = idx;

        if (idx)
            name += std::to_string(idx);

        opname[op.get()] = name;
    }

    for (auto op : f.get_ordered_ops()) {
        if (literal_consts.count(op)) continue;

        const auto & type_info = op->get_type_info();
        auto version_info = std::string(type_info.get_version());
        auto type = version_info + "::" + type_info.name;
        // fix name for cpu custom name
        if (version_info == "cpu_plugin_opset") {
            type = type_info.name;
            type += "Node";
        }
        auto name = opname[op.get()];
        os << prefix << "    ";

        if (auto constop = std::dynamic_pointer_cast<op::v0::Constant>(op)) {
            auto ele_size = shape_size(constop->get_shape());
            auto ele_type = constop->get_element_type();
            bool use_comment = false;
            if (ele_type.is_integral_number() && ele_size < 9) {
                use_comment = false;
                os << "auto " << name << " = GenConst({";
            } else {
                use_comment = true;
                os << "auto " << name << " = GenPattern<" << type << ">({/*";
            }
            if (ele_type == element::Type_t::f32) {
                os << PrintableVector<float>(constop->get_vector<float>());
            } else if (ele_type == element::Type_t::i8) {
                os << PrintableVector<int8_t>(constop->get_vector<int8_t>());
            } else if (ele_type == element::Type_t::u8) {
                os << PrintableVector<uint8_t>(constop->get_vector<uint8_t>());
            } else {
                if (ele_size < 9) {
                    sep = "";
                    for (auto v : constop->get_value_strings()) {
                        os << sep << v;
                        sep = ", ";
                    }
                } else {
                    os << "...";
                }
            }
            if (use_comment) os << "*/";
            os << "}, \"" << get_output_values_info(op) << "\");" << std::endl;
        } else {
            os << "auto " << name << " = GenPattern<" << type << ">({";
            sep = "";
            for (int i = 0; i < op->get_input_size(); i++) {
                auto vout = op->get_input_source_output(i);
                auto iop = vout.get_node_shared_ptr();
                if (iop->get_output_size() > 1) {
                    auto out_port = vout.get_index();
                    os << sep << tag << opname[iop.get()] << "->output(" << out_port << ")";
                } else {
                    if (literal_consts.count(iop))
                        os << sep << tag << literal_consts[iop];
                    else
                        os << sep << tag << opname[iop.get()];
                }
                sep = ", ";
            }
            // attributes are passed in using dict
            os << "}, \"" << get_output_values_info(op) << "\"";

            std::stringstream ss2;
            OstreamAttributeVisitor osvis(ss2);
            op->visit_attributes(osvis);
            auto str_attr = ss2.str();
            if (str_attr.size())
                os << ", {" << str_attr << "}";
            os << ");   //  " << op->get_friendly_name();

            os << "(";
            sep = "";
            for (int i = 0; i < op->get_input_size(); i++) {
                auto vout = op->get_input_source_output(i);
                auto iop = vout.get_node_shared_ptr();
                os << sep << tag << iop->get_friendly_name();
                if (iop->get_output_size() > 1) {
                    auto out_port = vout.get_index();
                    os << "[" << out_port << "]";
                }
                sep = ", ";
            }
            os << ")" << std::endl;;

            //auto op_output_size = op->get_output_size();
            //if (op_output_size > 1) {
            //    os << "    " << name << "->set_output_size(" << op_output_size << ");" << std::endl;
            //}
        }

        // recursively output subgraphs
        if (auto msubgraph = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(op)) {
            auto cnt = msubgraph->get_internal_subgraphs_size();
            for (int i = 0; i < cnt; i++) {
                os << "    MultiSubGraphOp " << tag << msubgraph->get_friendly_name() << "[" << i << "]" << std::endl;
                dump_cpp_style(os, msubgraph->get_function(i));
            }
        }
    }
    os << prefix << "}\n";
}

bool DumpModel::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::ofstream ofs(file_name);
    if (!ofs) {
        std::cout << "Error opening file " << file_name << " for output" << std::endl;
        return false;
    }
    dump_cpp_style(ofs, model);
    ofs.close();

    ov::serialize(model, file_name + ".xml", "/dev/null");
    std::cout << "=== Model dumpped into " << file_name << " and " << file_name << ".xml ===" << std::endl;

    return false;
}

bool DumpModel::run_on_graph(const ov::intel_cpu::Graph& graph) {
    // text format
    std::ofstream ofs(file_name, std::ofstream::out);
    ofs << graph;

    // xml format (will crash, so commented)
    // ov::serialize(graph.dump(), file_name + ".xml", "/dev/null");

    std::cout << "=== Graph dumpped into " << file_name << " and " << file_name << ".xml ===" << std::endl;
}

}   // namespace intel_cpu
}   // namespace ov
#endif

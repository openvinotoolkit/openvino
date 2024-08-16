
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef CPU_DEBUG_CAPS

#include "cpu_memory.h"
#include "debug_capabilities.h"
#include "node.h"
#include "edge.h"
#include "graph.h"
#include <iomanip>
#include "nodes/input.h"
#include "nodes/eltwise.h"

#include "openvino/op/util/multi_subgraph_base.hpp"
#include "common/primitive_desc_iface.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "onednn/iml_type_mapper.h"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include <memory>

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
#elif defined(__APPLE__) || defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_RISCV64)
       __builtin_trap();
#else
        asm("int3");
#endif
    }
}

std::ostream & operator<<(std::ostream & os, const MemoryDesc& desc) {
    os << desc.getShape().toString()
       << " " << desc.getPrecision().get_type_name()
       << " " << desc.serializeFormat();
    return os;
}

std::ostream & operator<<(std::ostream & os, const dnnl::primitive_attr& attr) {
    return dnnl::impl::operator<<(os, attr.get());
}

std::ostream & operator<<(std::ostream & os, const dnnl::algorithm& alg) {
    return dnnl::impl::operator<<(os, convert_to_c(alg));
}

std::ostream & operator<<(std::ostream & os, const PortConfig& config) {
    const char* sep = ",";
    os << sep << *config.getMemDesc();
    os << " inPlace:" << config.inPlace();
    config.constant() ? os << " constant" : os << " non-constant";
    return os;
}

std::ostream & operator<<(std::ostream & os, const NodeConfig& config) {
    os << "(";
    for (auto & conf : config.inConfs)
        os << conf;
    os << ") -> (";
    for (auto & conf : config.outConfs)
        os << conf;
    os << ")" << '\n';
    return os;
}

std::ostream & operator<<(std::ostream & os, const NodeDesc& desc) {
    os << "  " << impl_type_to_string(desc.getImplementationType());
    os << desc.getConfig();
    return os;
}

std::ostream & operator<<(std::ostream & os, const Edge& edge) {
    os << edge.getParent()->getName() << "[" << edge.getInputNum() << "]->"
       << edge.getChild()->getName() << "[" << edge.getOutputNum() << "]";
    return os;
}

std::ostream & operator<<(std::ostream & os, const Node &c_node) {
    Node & node = const_cast<Node &>(c_node);
    const int align_col = 50;
    const char * comma = "";
    auto node_id = [](Node & node) {
        auto id = node.getName();
        if (id.size() > 50)
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

    auto getData = [](const MemoryPtr& ptr) {
        std::string ret;
        try {
            std::stringstream ss;
            ss << ptr->getData();
            ret = ss.str();
        } catch (const std::exception& e) {
            ret = "?";
        }
        return ret;
    };

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
                    leftside << comma << desc->getPrecision().get_type_name()
                                << "_" << desc->serializeFormat()
                                << "_" << shape_str
                                << "&" << getData(ptr);
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
                leftside << comma << desc->getPrecision().get_type_name()
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
                os << c.getMemDesc()->getPrecision().get_type_name()
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
                leftside << comma << c.getMemDesc()->getPrecision().get_type_name()
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
            prec_name = node.getOriginalOutputPrecisionAtPort(i).get_type_name();
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
        os << comma;
        const char * sep2 = "";
        for (const auto & e : node.getParentEdges()) {
            auto edge = e.lock();
            if (!edge) continue;
            if (edge->getOutputNum() != static_cast<int>(port)) continue;
            auto n = edge->getParent();
            os << sep2;
            os << node_id(*edge->getParent());
            auto ptr = edge->getMemoryPtr();
            if (ptr) {
                os << "&" << getData(ptr);
            }
            if (!is_single_output_port(*n))
                os << "[" << edge->getInputNum() << "]";
            sep2 = "|"; // show all edges at single port(usually indicating bugs)
        }
        comma = ",";
    }

    if (node.getType() == intel_cpu::Type::Input && node.isConstant()) {
        if (auto input_node = reinterpret_cast<intel_cpu::node::Input *>(&node)) {
            auto pmem = input_node->getMemoryPtr();
            void * data = pmem->getData();
            auto shape = pmem->getDesc().getShape().getDims();

            if (shape_size(shape) <= 8) {
                auto type = pmem->getDesc().getPrecision();
                auto tensor = ov::Tensor(type, shape, data);
                auto constop = std::make_shared<ov::op::v0::Constant>(tensor);
                comma = "";
                for (auto & v : constop->get_value_strings()) {
                    os << comma << v;
                    comma = ",";
                }
            } else {
                os << "...";
            }
        } else {
            os << "?";
        }
    }

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
    os << " " << node.getOriginalLayers();
    os << " " << node.getParallelDomain();

    if (node.PerfCounter().count()) {
        os << " latency:" << node.PerfCounter().avg() << "(us) x" << node.PerfCounter().count();
    }

    for (auto & fn : node.getFusedWith()) {
        os << "\n\t  FusedWith: " << *fn;
    }

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
std::ostream & operator<<(std::ostream & os, const Shape& shape) {
    os << shape.toString();
    return os;
}

// Print complex data structures in a textualized form to the console is an efficient way to investigate them
std::ostream & operator<<(std::ostream & os, const Graph& g) {
    os << "ov::intel_cpu::Graph " << g.GetName() << " {" << std::endl;
    for (auto &graphNode : g.GetNodes()) {
        std::cout << *graphNode << std::endl;
    }
    os << "};" << std::endl;
    return os;
}

class OstreamAttributeVisitor : public ov::AttributeVisitor {
    std::ostream & os;

public:
    OstreamAttributeVisitor(std::ostream & os) : os(os) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (auto a = ov::as_type<ov::AttributeAdapter<std::set<std::string>>>(&adapter)) {
            const auto& value = join(a->get());
            append_attribute(name.c_str(), value.c_str());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::vector<ov::element::Type>>>(&adapter)) {
            const auto& value = join(a->get());
            append_attribute(name.c_str(), value.c_str());
        } else {
            append_attribute(name.c_str(), "?");
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        append_attribute(name.c_str(), std::to_string(adapter.get()).c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        append_attribute(name.c_str(), adapter.get().c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        append_attribute(name.c_str(), std::to_string(adapter.get()).c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        append_attribute(name.c_str(), std::to_string(adapter.get()).c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int>>& adapter) override {
        const auto& value = join(adapter.get());
        append_attribute(name.c_str(), value.c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        const auto& value = join(adapter.get());
        append_attribute(name.c_str(), value.c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        const auto& value = join(adapter.get());
        append_attribute(name.c_str(), value.c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        const auto& value = join(adapter.get());
        append_attribute(name.c_str(), value.c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        const auto& value = join(adapter.get());
        append_attribute(name.c_str(), value.c_str());
    }

    void append_attribute(const char * name, const char * value) {
        os << " " << name << "=" << value;
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        append_attribute(name.c_str(), "Model");
    }

    template<class Container>
    inline std::string join(const Container& strs) {
        std::stringstream ss;
        ss << "[" << ov::intel_cpu::join(strs, ',') << "]";
        return ss.str();
    }
};

std::ostream & operator<<(std::ostream & os, const PrintableModel& model) {
    const ov::Model& f = model.model;
    const std::string& tag = model.tag;
    const std::string& prefix = model.prefix;
    OstreamAttributeVisitor osvis(os);
    std::string sep = "";
    os << prefix;
    for (const auto& op : f.get_results()) {
        os << sep << op->get_name();
        sep = ",";
    }
    os << " " << f.get_friendly_name() << "(\n" << prefix;
    for (const auto& op : f.get_parameters()) {
        os << "\t" << tag << op->get_friendly_name() << ",\n" << prefix;
    }
    os << ") {\n";
    for (const auto& op : f.get_ordered_ops()) {
        auto type = op->get_type_name();
        auto name = op->get_friendly_name();
        os << prefix << "\t";
        if (op->get_output_size() > 1)
            os << "(";
        sep = "";
        for (size_t i = 0; i < op->get_output_size(); i++) {
            os << sep << op->get_output_element_type(i) << "_" << op->get_output_partial_shape(i);
            sep = ",";
        }
        if (op->get_output_size() > 1)
            os << ")";
        os << "  " << tag << name << " = " << type << "(";
        sep = "";
        for (size_t i = 0; i < op->get_input_size(); i++) {
            auto vout = op->get_input_source_output(i);
            auto iop = vout.get_node_shared_ptr();
            if (iop->get_output_size() > 1) {
                auto out_port = vout.get_index();
                os << sep << tag << iop->get_friendly_name() << "[" << out_port << "]";
            } else {
                os << sep << tag << iop->get_friendly_name();
            }
            sep = ",";
        }

        if (auto constop = std::dynamic_pointer_cast<op::v0::Constant>(op)) {
            if (constop->get_element_type() == element::Type_t::f32) {
                os << printable(constop->get_vector<float>());
            } else if (constop->get_element_type() == element::Type_t::i8) {
                os << printable(constop->get_vector<int8_t>());
            } else if (constop->get_element_type() == element::Type_t::u8) {
                os << printable(constop->get_vector<uint8_t>());
            } else {
                auto sz = shape_size(constop->get_shape());
                if (sz < 9) {
                    sep = "";
                    for (const auto& v : constop->get_value_strings()) {
                        os << sep << v;
                        sep = ",";
                    }
                } else {
                    os << "...";
                }
            }
        }

        os << ") \t attrs:";
        op->visit_attributes(osvis);
        os << std::endl;

        // recursively output subgraphs
        if (auto msubgraph = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(op)) {
            auto cnt = msubgraph->get_internal_subgraphs_size();
            for (size_t i = 0; i < cnt; i++) {
                os << "\t\t MultiSubGraphOp " << tag << msubgraph->get_friendly_name() << "[" << i << "]" << std::endl;
                os << PrintableModel(*msubgraph->get_function(i).get(), tag, prefix + "\t\t");
            }
        }
    }
    os << prefix << "}\n";
    os << prefix << "fp16_compress disabled Ngraph nodes:\n";
    for (const auto& op : f.get_ordered_ops()) {
        if (ov::fp16_compression_is_disabled(op) && !std::dynamic_pointer_cast<op::v0::Constant>(op))
            os << "\t" << tag << op->get_friendly_name() << "\n";
    }
    return os;
}

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
    os << " " << dnnl_fmt_kind2str(desc.get()->format_kind);
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

template <typename T>
std::string to_string(const T* values, size_t N, size_t maxsize) {
    std::stringstream ss;
    for (size_t i = 0; i < N; i++) {
        if (i > 0)
            ss << ",";
        if (ss.tellp() > static_cast<std::stringstream::pos_type>(maxsize)) {
            ss << "..." << N << "in total";
            break;
        }
        if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value)
            ss << static_cast<int>(values[i]);
        else
            ss << values[i];
    }
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const IMemory& mem) {
    const auto& desc = mem.getDesc();
    os << desc;
    if (mem.isDefined()) {
        os << " [";
        if (desc.getPrecision() == ov::element::i32) {
            os << to_string(mem.getDataAs<int32_t>(), mem.getSize() / sizeof(int32_t), 256);
        } else if (desc.getPrecision() == ov::element::f32) {
            os << to_string(mem.getDataAs<float>(), mem.getSize() / sizeof(float), 256);
        } else if (desc.getPrecision() == ov::element::i64) {
            os << to_string(mem.getDataAs<int64_t>(), mem.getSize() / sizeof(int64_t), 256);
        } else {
            os << " ? ";
        }
        os << "]";
    }
    return os;
}
// @todo remove
void print_dnnl_memory(const dnnl::memory& memory, const size_t size, const int id, const char* message) {
    const size_t s = memory.get_desc().get_size() / sizeof(float);
    std::cout << message << " " << id << " size: " << s << ", values: ";
    auto m = reinterpret_cast<float*>(memory.get_data_handle());
    for (size_t i = 0; i < std::min(s, size); i++) {
        std::cout << *m << " ";
        m++;
    }
    std::cout << "\n";
}

}   // namespace intel_cpu
}   // namespace ov

bool getEnvBool(const char* name) {
    static const bool env = ov::util::getenv_bool(name);
    return env;
}

#endif

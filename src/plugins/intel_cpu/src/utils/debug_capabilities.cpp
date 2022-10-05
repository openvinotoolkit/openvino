
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef CPU_DEBUG_CAPS

#include "debug_capabilities.h"
#include "node.h"
#include "edge.h"
#include <iomanip>
#include "nodes/input.h"
#include "nodes/eltwise.h"
#include <ie_ngraph_utils.hpp>

namespace ov {
namespace intel_cpu {

DebugLogEnabled::DebugLogEnabled(const char* file, const char* func, int line) {
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
    // check each filter patten:
    bool filter_match_action;
    if (p_filters[0] == '-') {
        p_filters++;
        filter_match_action = false;
    } else {
        filter_match_action = true;
    }

    bool match = false;
    const char* p0 = p_filters;
    const char* p1;
    while (*p0 != 0) {
        p1 = p0;
        while (*p1 != ';' && *p1 != 0)
            ++p1;
        std::string patten(p0, p1 - p0);
        if (patten == file_name || patten == func || patten == tag || patten == file_name_with_line) {
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
#else
        asm("int3");
#endif
    }
}

std::ostream & operator<<(std::ostream & os, const dnnl::memory::desc& desc) {
    char sep = '(';
    os << "dims:";
    for (int i = 0; i < desc.data.ndims; i++) {
        os << sep << desc.data.dims[i];
        sep = ',';
    }
    os << ")";

    sep = '(';
    os << "strides:";
    for (int i = 0; i < desc.data.ndims; i++) {
        os << sep << desc.data.format_desc.blocking.strides[i];
        sep = ',';
    }
    os << ")";

    for (int i = 0; i < desc.data.format_desc.blocking.inner_nblks; i++) {
        os << desc.data.format_desc.blocking.inner_blks[i] << static_cast<char>('a' + desc.data.format_desc.blocking.inner_idxs[i]);
    }

    os << " " << dnnl_dt2str(desc.data.data_type);
    return os;
}

std::ostream & operator<<(std::ostream & os, const MemoryDesc& desc) {
    os << desc.getShape().toString()
       << " " << desc.getPrecision().name()
       << " " << desc.serializeFormat();
    return os;
}

std::ostream & operator<<(std::ostream & os, const NodeDesc& desc) {
    os << "    ImplementationType: " << impl_type_to_string(desc.getImplementationType()) << std::endl;
    for (auto & conf : desc.getConfig().inConfs) {
        os << "    inConfs: " << *conf.getMemDesc();
        if (conf.inPlace() >= 0) os << " inPlace:" << conf.inPlace();
        if (conf.constant()) os << " constant";
        os << std::endl;
    }
    for (auto & conf : desc.getConfig().outConfs) {
        os << "    outConfs: " << *conf.getMemDesc();
        if (conf.inPlace() >= 0) os << " inPlace:" << conf.inPlace();
        if (conf.constant()) os << " constant";
        os << std::endl;
    }
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
    auto replace_all = [](std::string& inout, std::string what, std::string with) {
        std::size_t count{};
        for (std::string::size_type pos{};
            inout.npos != (pos = inout.find(what.data(), pos, what.length()));
            pos += with.length(), ++count) {
            inout.replace(pos, what.length(), with.data(), with.length());
        }
        return count;
    };
    auto nodeDesc = node.getSelectedPrimitiveDescriptor();
    std::stringstream leftside;

    int num_output_port = 0;
    for (auto wptr : node.getChildEdges()) {
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
                                << "_" << ptr->GetData();
                    b_ouputed = true;
                } else {
                    leftside << "(empty)";
                }
            }
            if (!b_ouputed && nodeDesc && i < nodeDesc->getConfig().outConfs.size()) {
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
        for (int i = 0; i < num_output_port; i++) {
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
    for (int port = 0; port < node.getParentEdges().size(); ++port) {
        // find the Parent edge connecting to port
        for (const auto & e : node.getParentEdges()) {
            auto edge = e.lock();
            if (!edge) continue;
            if (edge->getOutputNum() != port) continue;
            auto n = edge->getParent();
            os << comma;
            os << node_id(*edge->getParent());
            if (!is_single_output_port(*n))
                os << "[" << edge->getInputNum() << "]";
            comma = ",";
            break;
        }
    }

    if (node.getType() == intel_cpu::Type::Input && node.isConstant()) {
        if (auto input_node = reinterpret_cast<intel_cpu::node::Input *>(&node)) {
            auto pmem = input_node->getMemoryPtr();
            void * data = pmem->GetData();
            auto shape = pmem->getDesc().getShape().getDims();

            if (shape_size(shape) <= 8) {
                auto type = InferenceEngine::details::convertPrecision(pmem->getDesc().getPrecision());
                auto tensor = std::make_shared<ngraph::runtime::HostTensor>(type, shape, data);
                auto constop = std::make_shared<ngraph::op::Constant>(tensor);
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
        << ", Gamma=" << eltwise_node->getGamma();
    }

    os << ")  ";
    os << " " << node.getPrimitiveDescriptorType();

    // last line(s): fused layers
    os << " " << node.getOriginalLayers();

    if (node.PerfCounter().count()) {
        os << " latency:" << node.PerfCounter().avg() << "(us) x" << node.PerfCounter().count();
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

}   // namespace intel_cpu
}   // namespace ov

#endif
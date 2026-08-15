// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "paddle_reader.hpp"

#include "runtime/file_util.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>

namespace ov::core {
namespace vulkan {
namespace cross_platform {
namespace paddle_r {

namespace {

// ---- minimal protobuf wire-format reader (proto2) ---------------------------

struct pb_reader {
    const uint8_t* p;
    const uint8_t* end;

    uint64_t varint() {
        uint64_t v = 0;
        int shift = 0;
        while (p < end && shift < 64) {
            const uint8_t b = *p++;
            v |= static_cast<uint64_t>(b & 0x7F) << shift;
            if (!(b & 0x80))
                return v;
            shift += 7;
        }
        if (shift >= 64)
            p = end;
        return v;
    }

    bool field(uint32_t& no, uint32_t& wire) {
        if (p >= end)
            return false;
        const uint64_t tag = varint();
        no = static_cast<uint32_t>(tag >> 3);
        wire = static_cast<uint32_t>(tag & 7);
        return true;
    }

    void skip(uint32_t wire) {
        switch (wire) {
            case 0:
                varint();
                break;
            case 1:
                p += 8;
                break;
            case 2: {
                const uint32_t n = static_cast<uint32_t>(varint());
                p += n;
                break;
            }
            case 5:
                p += 4;
                break;
            default:  // groups not used by framework.proto
                p = end;
                break;
        }
        if (p > end)
            p = end;
    }

    bool submsg(std::vector<uint8_t>& out) {
        if (p >= end)
            return false;
        const uint32_t n = static_cast<uint32_t>(varint());
        if (p + n > end) {
            p = end;
            return false;
        }
        out.assign(p, p + n);
        p += n;
        return true;
    }

    bool str(std::string& s) {
        std::vector<uint8_t> v;
        if (!submsg(v))
            return false;
        s.assign(reinterpret_cast<const char*>(v.data()), v.size());
        return true;
    }

    float fixed32() {
        float f = 0;
        if (p + 4 <= end) {
            std::memcpy(&f, p, 4);
            p += 4;
        } else {
            p = end;
        }
        return f;
    }
};

// ---- parsed model structures ------------------------------------------------

struct var_info {
    std::string name;
    bool persistable = false;
    int32_t type = 0;       // VarType::Type enum (7 = LOD_TENSOR)
    int32_t data_type = 0;  // tensor element type (5 = FP32)
    std::vector<int64_t> dims;
};

struct attr_val {
    int32_t kind = -1;  // AttrType enum
    int32_t i = 0;
    int64_t l = 0;
    bool b = false;
    float f = 0;
    std::string name;  // attr name (field 1)
    std::string s;     // STRING value (field 5)
    std::vector<int32_t> ints;
    std::vector<float> floats;
    std::vector<std::string> strings;
    std::vector<int64_t> longs;
};

struct op_info {
    std::string type;
    // parameter -> list of argument var names
    std::vector<std::pair<std::string, std::vector<std::string>>> inputs;
    std::vector<std::pair<std::string, std::vector<std::string>>> outputs;
    std::map<std::string, attr_val> attrs;
};

// ---- message parsers ---------------------------------------------------------

var_info parse_var(pb_reader& r) {
    var_info v;
    uint32_t no, wire;
    while (r.field(no, wire)) {
        if (no == 1 && wire == 2) {
            r.str(v.name);
        } else if (no == 2 && wire == 2) {
            std::vector<uint8_t> vt;
            r.submsg(vt);
            pb_reader vr{vt.data(), vt.data() + vt.size()};
            uint32_t n2, w2;
            while (vr.field(n2, w2)) {
                if (n2 == 1 && w2 == 0) {
                    v.type = static_cast<int32_t>(vr.varint());
                } else if (n2 == 3 && w2 == 2) {  // lod_tensor
                    std::vector<uint8_t> lt;
                    vr.submsg(lt);
                    pb_reader lr{lt.data(), lt.data() + lt.size()};
                    uint32_t n3, w3;
                    while (lr.field(n3, w3)) {
                        if (n3 == 1 && w3 == 2) {  // tensor (TensorDesc)
                            std::vector<uint8_t> td;
                            lr.submsg(td);
                            pb_reader tr{td.data(), td.data() + td.size()};
                            uint32_t n4, w4;
                            while (tr.field(n4, w4)) {
                                if (n4 == 1 && w4 == 0)
                                    v.data_type = static_cast<int32_t>(tr.varint());
                                else if (n4 == 2 && w4 == 0)
                                    v.dims.push_back(static_cast<int64_t>(tr.varint()));
                                else
                                    tr.skip(w4);
                            }
                        } else {
                            lr.skip(w3);
                        }
                    }
                } else {
                    vr.skip(w2);
                }
            }
        } else if (no == 3 && wire == 0) {
            v.persistable = r.varint() != 0;
        } else {
            r.skip(wire);
        }
    }
    return v;
}

// Read a single OpDesc.Attr / Var.Desc.Attr style message body already sliced.
attr_val parse_attr(pb_reader& r) {
    attr_val a;
    uint32_t no, wire;
    while (r.field(no, wire)) {
        switch (no) {
            case 1:
                if (wire == 2)
                    r.str(a.name);
                else
                    r.skip(wire);
                break;
            case 2:
                if (wire == 0)
                    a.kind = static_cast<int32_t>(r.varint());
                else
                    r.skip(wire);
                break;
            case 3:
                if (wire == 0)
                    a.i = static_cast<int32_t>(r.varint());
                else
                    r.skip(wire);
                break;
            case 4:
                if (wire == 5)
                    a.f = r.fixed32();
                else
                    r.skip(wire);
                break;
            case 5:
                if (wire == 2)
                    r.str(a.s);
                else
                    r.skip(wire);
                break;
            case 6:
                if (wire == 0)
                    a.ints.push_back(static_cast<int32_t>(r.varint()));
                else
                    r.skip(wire);
                break;
            case 7:
                if (wire == 5)
                    a.floats.push_back(r.fixed32());
                else
                    r.skip(wire);
                break;
            case 8:
                if (wire == 2) {
                    std::string s;
                    r.str(s);
                    a.strings.push_back(std::move(s));
                } else
                    r.skip(wire);
                break;
            case 10:
                if (wire == 0)
                    a.b = r.varint() != 0;
                else
                    r.skip(wire);
                break;
            case 13:
                if (wire == 0)
                    a.l = static_cast<int64_t>(r.varint());
                else
                    r.skip(wire);
                break;
            case 15:
                if (wire == 0)
                    a.longs.push_back(static_cast<int64_t>(r.varint()));
                else
                    r.skip(wire);
                break;
            default:
                r.skip(wire);
                break;
        }
    }
    return a;
}

op_info parse_op(pb_reader& r) {
    op_info o;
    uint32_t no, wire;
    while (r.field(no, wire)) {
        if (no == 3 && wire == 2) {
            r.str(o.type);
        } else if ((no == 1 || no == 2) && wire == 2) {
            std::vector<uint8_t> vm;
            r.submsg(vm);
            pb_reader vr{vm.data(), vm.data() + vm.size()};
            std::pair<std::string, std::vector<std::string>> entry;
            uint32_t n2, w2;
            while (vr.field(n2, w2)) {
                if (n2 == 1 && w2 == 2)
                    vr.str(entry.first);
                else if (n2 == 2 && w2 == 2) {
                    std::string s;
                    vr.str(s);
                    entry.second.push_back(std::move(s));
                } else
                    vr.skip(w2);
            }
            if (no == 1)
                o.inputs.push_back(std::move(entry));
            else
                o.outputs.push_back(std::move(entry));
        } else if (no == 4 && wire == 2) {
            std::vector<uint8_t> am;
            r.submsg(am);
            pb_reader ar{am.data(), am.data() + am.size()};
            attr_val a = parse_attr(ar);
            o.attrs[a.name] = std::move(a);
        } else {
            r.skip(wire);
        }
    }
    return o;
}

// ---- helpers on parsed data --------------------------------------------------

const var_info* find_var(const std::vector<var_info>& vars, const std::string& name) {
    for (const auto& v : vars)
        if (v.name == name)
            return &v;
    return nullptr;
}

// first argument of the given input/output port, or "" if absent
const std::vector<std::string>* port_args(const std::vector<std::pair<std::string, std::vector<std::string>>>& ports,
                                          const std::string& name) {
    for (const auto& e : ports)
        if (e.first == name)
            return &e.second;
    return nullptr;
}

[[noreturn]] void unsupported(const std::string& what) {
    throw std::runtime_error("paddle_r: unsupported: " + what);
}

std::vector<size_t> to_size(const std::vector<int64_t>& d) {
    std::vector<size_t> out(d.begin(), d.end());
    return out;
}

size_t elem_product(const std::vector<int64_t>& d) {
    size_t n = 1;
    for (int64_t v : d)
        n *= static_cast<size_t>(v);
    return n;
}

// Returns the ints/longs attr payload; empty when the attr is absent.
std::vector<size_t> attr_ints(const attr_val* a) {
    if (!a || a->ints.empty() && a->longs.empty())
        return {};
    if (!a->longs.empty())
        return std::vector<size_t>(a->longs.begin(), a->longs.end());
    return std::vector<size_t>(a->ints.begin(), a->ints.end());
}

}  // namespace

ir_graph paddle_parse_program(
    const std::vector<uint8_t>& program,
    const std::function<std::vector<uint8_t>(const std::string&)>& load_param) {
    // ---- parse ProgramDesc -> main block (idx 0) ----------------------------
    std::vector<var_info> vars;
    std::vector<op_info> ops;
    pb_reader pr{program.data(), program.data() + program.size()};
    uint32_t no, wire;
    while (pr.field(no, wire)) {
        if (no == 1 && wire == 2) {  // blocks
            std::vector<uint8_t> blk;
            pr.submsg(blk);
            pb_reader br{blk.data(), blk.data() + blk.size()};
            int32_t idx = -1;
            std::vector<var_info> bvars;
            std::vector<op_info> bops;
            uint32_t n2, w2;
            while (br.field(n2, w2)) {
                if (n2 == 1 && w2 == 0)
                    idx = static_cast<int32_t>(br.varint());
                else if (n2 == 3 && w2 == 2) {
                    std::vector<uint8_t> vv;
                    br.submsg(vv);
                    pb_reader vr{vv.data(), vv.data() + vv.size()};
                    bvars.push_back(parse_var(vr));
                } else if (n2 == 4 && w2 == 2) {
                    std::vector<uint8_t> ov;
                    br.submsg(ov);
                    pb_reader or_{ov.data(), ov.data() + ov.size()};
                    bops.push_back(parse_op(or_));
                } else {
                    br.skip(w2);
                }
            }
            if (idx == 0) {
                vars = std::move(bvars);
                ops = std::move(bops);
            }
        } else {
            pr.skip(wire);
        }
    }
    if (vars.empty() && ops.empty())
        unsupported("empty ProgramDesc (no main block)");

    ir_graph g;

    // Declared shapes for every LOD_TENSOR var (params, weights, intermediates).
    for (const auto& v : vars)
        if (v.type == 7 && !v.dims.empty())
            g.tensor_shapes[v.name] = to_size(v.dims);

    // ---- classify every buffer: parameter / constant / op-produced ----------
    // params come from feed ops (program order).
    std::map<std::string, ir_op> kinds;
    for (const auto& op : ops) {
        if (op.type == "feed") {
            for (const auto& o : op.outputs)
                for (const auto& a : o.second)
                    kinds[a] = ir_op::parameter;
        }
    }
    // persistable weights consumed by any op -> constants
    for (const auto& op : ops) {
        if (op.type == "feed" || op.type == "fetch")
            continue;
        for (const auto& in : op.inputs)
            for (const auto& a : in.second) {
                if (kinds.count(a))
                    continue;
                const var_info* v = find_var(vars, a);
                if (v && v->persistable)
                    kinds[a] = ir_op::constant;
            }
    }

    // ---- nodes: parameters, then constants, then ops (topological order) ----
    ir_node n;
    n.id.clear();

    auto add_param = [&](const var_info& v) {
        if (v.type != 7)
            unsupported("parameter '" + v.name + "' is not LOD_TENSOR");
        if (v.data_type != 5)
            unsupported("parameter '" + v.name + "' is not FP32");
        ir_node p;
        p.id = v.name;
        p.op = ir_op::parameter;
        g.nodes.push_back(std::move(p));
    };

    auto add_constant = [&](const var_info& v) {
        if (v.type != 7)
            unsupported("weight '" + v.name + "' is not LOD_TENSOR");
        if (v.data_type != 5)
            unsupported("weight '" + v.name + "' is not FP32 (only f32 weights supported)");
        const auto bytes = load_param(v.name);
        const size_t count = bytes.size() / sizeof(float);
        if (count * sizeof(float) != bytes.size() || count != elem_product(v.dims))
            unsupported("weight '" + v.name + "' size does not match its dims");
        ir_node c;
        c.id = v.name;
        c.op = ir_op::constant;
        g.nodes.push_back(std::move(c));
        std::vector<float> data(count);
        if (!bytes.empty())
            std::memcpy(data.data(), bytes.data(), bytes.size());
        g.constant_data[v.name] = std::move(data);
    };

    for (const auto& op : ops)
        if (op.type == "feed")
            for (const auto& o : op.outputs)
                for (const auto& a : o.second)
                    if (const var_info* v = find_var(vars, a); v)
                        add_param(*v);

    for (const auto& [name, k] : kinds)
        if (k == ir_op::constant)
            if (const var_info* v = find_var(vars, name); v)
                add_constant(*v);

    // ---- shape helpers -------------------------------------------------------
    auto shape = [&](const std::string& id) -> const std::vector<size_t>& {
        static const std::vector<size_t> empty;
        auto it = g.tensor_shapes.find(id);
        return it == g.tensor_shapes.end() ? empty : it->second;
    };

    // ---- translate supported ops ---------------------------------------------
    for (const auto& op : ops) {
        const std::string& t = op.type;
        auto arg = [&](const std::string& port) -> std::string {
            const auto* v = port_args(op.inputs, port);
            if (v && !v->empty())
                return v->front();
            const auto* o = port_args(op.outputs, port);
            if (o && !o->empty())
                return o->front();
            return "";
        };
        auto out_id = [&]() -> std::string {
            for (const auto& o : op.outputs)
                if (!o.second.empty())
                    return o.second.front();
            return "";
        };

        if (t == "feed" || t == "fetch")
            continue;

        const std::string oid = out_id();
        if (oid.empty())
            unsupported("op '" + t + "' without output");

        ir_node node;
        node.id = oid;

        if (t == "relu") {
            const std::string x = arg("X");
            node.op = ir_op::relu;
            node.inputs = {x};
            g.tensor_shapes[oid] = shape(x);
        } else if (t == "elementwise_add") {
            const std::string x = arg("X"), y = arg("Y");
            if (shape(x) != shape(y))
                unsupported("elementwise_add: shapes differ (no broadcast)");
            node.op = ir_op::add;
            node.inputs = {x, y};
            g.tensor_shapes[oid] = shape(x);
        } else if (t == "matmul") {
            const std::string x = arg("X"), y = arg("Y");
            const bool t_x = op.attrs.count("transpose_X") ? op.attrs.at("transpose_X").b : false;
            const bool t_y = op.attrs.count("transpose_Y") ? op.attrs.at("transpose_Y").b : false;
            if (t_x)
                unsupported("matmul: transpose_X");
            const auto& sx = shape(x);
            const auto& sy = shape(y);
            const size_t M = sx.at(0);
            const size_t K = sx.at(1);
            const size_t N = t_y ? sy.at(0) : sy.at(1);
            node.op = ir_op::matmul;
            node.matmul_transpose_b = t_y;
            node.inputs = {x, y};
            g.tensor_shapes[oid] = {M, N};
        } else if (t == "conv2d") {
            const std::string x = arg("X"), w = arg("Filter"), b = arg("Bias");
            const auto& sx = shape(x);
            const auto& sw = shape(w);
            if (sx.size() != 4 || sw.size() != 4)
                unsupported("conv2d: expected 4D NCHW input/filter");
            auto st = attr_ints(op.attrs.count("strides") ? &op.attrs.at("strides") : nullptr);
            auto pd = attr_ints(op.attrs.count("paddings") ? &op.attrs.at("paddings") : nullptr);
            if (st.empty())
                st = {1, 1};
            if (pd.empty())
                pd = {0, 0};
            const int32_t groups = op.attrs.count("groups") ? op.attrs.at("groups").i : 1;
            if (groups != 1)
                unsupported("conv2d: groups != 1");
            const std::string df = op.attrs.count("data_format") ? op.attrs.at("data_format").s : "NCHW";
            if (df != "NCHW")
                unsupported("conv2d: data_format " + df);
            const size_t SH = st.size() > 1 ? st[0] : st[0];
            const size_t SW = st.size() > 1 ? st[1] : st[0];
            const size_t PH = pd.size() > 1 ? pd[0] : 0;
            const size_t PW = pd.size() > 2 ? pd[2] : (pd.size() > 1 ? pd[1] : 0);
            const size_t KH = sw[2], KW = sw[3];
            const size_t OH = (sx[2] + 2 * PH - KH) / SH + 1;
            const size_t OW = (sx[3] + 2 * PW - KW) / SW + 1;
            node.op = ir_op::convolution;
            node.pool.strides = {SH, SW};
            node.pool.pads_begin = {PH, PW};
            node.inputs = {x, w};
            if (!b.empty())
                node.inputs.push_back(b);
            g.tensor_shapes[oid] = {sx[0], sw[0], OH, OW};
        } else if (t == "max_pool2d" || t == "avg_pool2d") {
            const std::string x = arg("X");
            const auto& sx = shape(x);
            if (sx.size() != 4)
                unsupported(t + ": expected 4D NCHW input");
            const bool ceil_mode = op.attrs.count("ceil_mode") ? op.attrs.at("ceil_mode").b : false;
            if (ceil_mode)
                unsupported(t + ": ceil_mode (floor only)");
            if (t == "avg_pool2d" && op.attrs.count("exclusive") && !op.attrs.at("exclusive").b)
                unsupported("avg_pool2d: exclusive=false");
            auto ks = attr_ints(op.attrs.count("ksize") ? &op.attrs.at("ksize") : nullptr);
            auto st = attr_ints(op.attrs.count("strides") ? &op.attrs.at("strides") : nullptr);
            auto pd = attr_ints(op.attrs.count("paddings") ? &op.attrs.at("paddings") : nullptr);
            if (ks.empty())
                unsupported(t + ": ksize missing");
            if (st.empty())
                st = {1, 1};
            if (pd.empty())
                pd = {0, 0};
            const size_t KH = ks[0], KW = ks.size() > 1 ? ks[1] : ks[0];
            const size_t SH = st.size() > 1 ? st[0] : st[0];
            const size_t SW = st.size() > 1 ? st[1] : st[0];
            const size_t PH = pd.size() > 1 ? pd[0] : 0;
            const size_t PW = pd.size() > 2 ? pd[2] : (pd.size() > 1 ? pd[1] : 0);
            const size_t OH = (sx[2] + 2 * PH - KH) / SH + 1;
            const size_t OW = (sx[3] + 2 * PW - KW) / SW + 1;
            node.op = t == "max_pool2d" ? ir_op::max_pool : ir_op::avg_pool;
            node.pool.kernel = {KH, KW};
            node.pool.strides = {SH, SW};
            node.pool.pads_begin = {PH, PW};
            node.inputs = {x};
            g.tensor_shapes[oid] = {sx[0], sx[1], OH, OW};
        } else {
            unsupported("op '" + t + "'");
        }

        g.nodes.push_back(std::move(node));
    }

    // ---- I/O ports -----------------------------------------------------------
    for (const auto& op : ops)
        if (op.type == "feed")
            for (const auto& o : op.outputs)
                for (const auto& a : o.second)
                    g.inputs.push_back(a);
    for (const auto& op : ops)
        if (op.type == "fetch")
            for (const auto& in : op.inputs)
                for (const auto& a : in.second)
                    g.outputs.push_back(a);

    if (g.inputs.empty() || g.outputs.empty())
        unsupported("model has no feed/fetch I/O");

    return g;
}

// Extracts the raw payload (field 3 = data bytes) of a persistable LoDTensor
// file. The on-disk weight file is itself a protobuf:
//   LoDTensor { version=1, lod_tensor=2(LoDTensorDesc{tensor=1(TensorDesc{
//     data_type=1, dims=2...})}), data=3 }
std::vector<uint8_t> lodtensor_data(const std::vector<uint8_t>& file) {
    std::vector<uint8_t> data;
    pb_reader r{file.data(), file.data() + file.size()};
    uint32_t no, wire;
    while (r.field(no, wire)) {
        if (no == 3 && wire == 2) {
            std::vector<uint8_t> d;
            if (!r.submsg(d))
                unsupported("weight file: truncated data field");
            data = std::move(d);
        } else {
            r.skip(wire);
        }
    }
    if (data.empty())
        unsupported("weight file: no data payload");
    return data;
}

ir_graph paddle_load_model(const std::string& dir) {
    const auto program = ov::util::load_binary(ov::util::make_path(dir + "/__model__"));
    std::vector<uint8_t> prog(program.size());
    for (size_t i = 0; i < program.size(); ++i)
        prog[i] = static_cast<uint8_t>(program[i]);
    return paddle_parse_program(prog, [&](const std::string& var_name) {
        const auto raw = ov::util::load_binary(ov::util::make_path(dir + "/" + var_name));
        std::vector<uint8_t> file(raw.size());
        for (size_t i = 0; i < raw.size(); ++i)
            file[i] = static_cast<uint8_t>(raw[i]);
        return lodtensor_data(file);
    });
}

}  // namespace paddle_r
}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov::core

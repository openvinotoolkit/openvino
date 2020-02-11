// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iosfwd>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include <ie_data.h>
#include <ie_blob.h>
#include <ie_layers.h>

#include <details/ie_exception.hpp>

#include <vpu/utils/extra.hpp>
#include <vpu/utils/io.hpp>
#include <vpu/utils/small_vector.hpp>

namespace vpu {

namespace ie = InferenceEngine;

//
// DotSerializer
//

class DotSerializer final {
public:
    class Ident final {
    public:
        explicit Ident(DotSerializer& out) : _out(out) {
            ++_out._ident;
        }

        ~Ident() {
            --_out._ident;
        }

    private:
        DotSerializer& _out;
    };

public:
    explicit DotSerializer(std::ostream& os) : _os(os) {}

    DotSerializer(const DotSerializer& other) = delete;
    DotSerializer& operator=(const DotSerializer&) = delete;

    template <typename... Args>
    void append(const char* format, const Args&... args) {
        for (size_t i = 0; i < _ident; ++i)
            _os << "    ";

        formatPrint(_os, format, args...);

        _os << std::endl;
    }

private:
    std::ostream& _os;
    size_t _ident = 0;

    friend class Ident;
};

#define VPU_DOT_IDENT(dotOut) vpu::DotSerializer::Ident VPU_COMBINE(dotIdent, __LINE__) (dotOut)

//
// DotLabel
//

class DotLabel final {
public:
    DotLabel(const std::string& caption, DotSerializer& out);
    explicit DotLabel(DotLabel& other);
    ~DotLabel();

    DotLabel(const DotLabel& other) = delete;
    DotLabel& operator=(const DotLabel&) = delete;

    template <typename K, typename V>
    void appendPair(const K& key, const V& val);

    template <typename... Args>
    void appendValue(const char* format, const Args&... args);

    void addIdent();

private:
    DotSerializer& _out;
    DotLabel* _parent = nullptr;
    size_t _ident = 0;
    std::ostringstream _ostr;
};

//
// printTo
//

template <typename T>
void printTo(DotLabel& lbl, const T& val);

template <typename T1, typename T2>
void printTo(DotLabel& lbl, const std::pair<T1, T2>& p);

template <typename T>
void printTo(DotLabel& lbl, const std::vector<T>& cont);

template <typename T>
void printTo(DotLabel& lbl, const std::set<T>& cont);

template <typename T, class H>
void printTo(DotLabel& lbl, const std::unordered_set<T, H>& cont);

template <typename K, typename V>
void printTo(DotLabel& lbl, const std::map<K, V>& map);

template <typename K, typename V, class H>
void printTo(DotLabel& lbl, const std::unordered_map<K, V, H>& map);

void printTo(DotLabel& lbl, const ie::DataPtr& ieData);

void printTo(DotLabel& lbl, const ie::Blob::Ptr& ieBlob);

void printTo(DotLabel& lbl, const ie::CNNLayerPtr& ieLayer);

class Any;
void printTo(DotLabel& lbl, const Any& any);

class AttributesMap;
void printTo(DotLabel& lbl, const AttributesMap& attrs);

template <typename T, int Capacity>
void printTo(DotLabel& lbl, const SmallVector<T, Capacity>& cont);

//
// Implementation
//

template <typename T>
void printTo(DotLabel& lbl, const T& val) {
    lbl.appendValue("%s", val);
}

template <typename T1, typename T2>
void printTo(DotLabel& lbl, const std::pair<T1, T2>& p) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("first", p.first);
    subLbl.appendPair("second", p.second);
}

template <class Cont>
void printContainer(DotLabel& lbl, const Cont& cont) {
    if (cont.size() > 4) {
        DotLabel subLbl(lbl);

        size_t ind = 0;
        for (const auto& val : cont) {
            subLbl.addIdent();
            subLbl.appendValue("%s", val);
            if (ind + 1 < cont.size()) {
                subLbl.appendValue(",\\l");
            }
            if (ind > 8) {
                subLbl.appendValue("...");
                break;
            }
            ++ind;
        }
    } else {
        lbl.appendValue("%s", cont);
    }
}

template <typename T>
void printTo(DotLabel& lbl, const std::vector<T>& cont) {
    printContainer(lbl, cont);
}

template <typename T>
void printTo(DotLabel& lbl, const std::set<T>& cont) {
    printContainer(lbl, cont);
}

template <typename T, class H>
void printTo(DotLabel& lbl, const std::unordered_set<T, H>& cont) {
    printContainer(lbl, cont);
}

template <class Map>
void printMap(DotLabel& lbl, const Map& map) {
    DotLabel subLbl(lbl);
    for (const auto& p : map) {
        subLbl.appendPair(p.first, p.second);
    }
}

template <typename K, typename V>
void printTo(DotLabel& lbl, const std::map<K, V>& map) {
    printMap(lbl, map);
}

template <typename K, typename V, class H>
void printTo(DotLabel& lbl, const std::unordered_map<K, V, H>& map) {
    printMap(lbl, map);
}

template <typename T, int Capacity>
void printTo(DotLabel& lbl, const SmallVector<T, Capacity>& cont) {
    printContainer(lbl, cont);
}

template <typename K, typename V>
void DotLabel::appendPair(const K& key, const V& val) {
    addIdent();
    printTo(*this, key);
    appendValue(" = ");
    printTo(*this, val);
    appendValue("\\l");
}

template <typename... Args>
void DotLabel::appendValue(const char* format, const Args&... args) {
    formatPrint(_ostr, format, args...);
}

}  // namespace vpu

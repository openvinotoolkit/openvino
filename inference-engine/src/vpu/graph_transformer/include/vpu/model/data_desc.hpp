// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <map>
#include <type_traits>
#include <functional>
#include <utility>

#include <ie_layouts.h>

#include <vpu/model/base.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/io.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/checked_cast.hpp>

//
// Description (type, layout, dimensions, strides) for Data objects inside the VPU Model.
//
// The VPU Model uses own represenatation of Data layout and dimensions.
// The dimensions are stored in a special container in memory-independent order.
// Each dimension has unique name, which can be represented as an index (eg. `width` : 0, `height` : 1, etc.).
// The DimsOrder parameter provides information about actual layout in the memory.
// During the Fathom Blob serialization VPU Graph Transformer will convert the dimensions from
// memory-independant order to memory order from minor to major dimension.
//

namespace vpu {

namespace ie = InferenceEngine;

//
// DataType
//

// Must be synchronized with MvTensor
VPU_DECLARE_ENUM(DataType,
    FP16 = 0,
    U8 = 1,
//     S32 = 2,  // TODO: remove from MvTensor
    FP32 = 3,
    I8 = 4
)

//
// Dim
//

//
// Named dimensions for better readability.
//

VPU_DECLARE_ENUM(Dim,
    Invalid = -1,
    W = 0,
    H = 1,
    C = 2,
    N = 3,
    _5 = 4,
    _6 = 5,
    _7 = 6,
    _8 = 7
)

//
// StorageOrder
//

//
// Types that are used to store order permutation in packed format.
//

using StorageOrder64 = uint64_t;
using StorageOrder32 = uint32_t;

// High-order digit excluded.
const int MAX_DIMS_64 = std::numeric_limits<StorageOrder64>::digits / 4 - 1;

const int MAX_DIMS_32 = std::numeric_limits<StorageOrder32>::digits / 4;

//
// DimValues
//

//
// Container to store dimensions values (sizes, offsets, strides).
// Internally it is a map from Dim to `int`.
// Should be used together with DimsOrder to get the permutation array.
//

template <typename T>
class DimValues_ final {
    static_assert(std::is_trivial<T>::value, "std::is_trivial<T>::value");

    using ValuesCont = std::array<std::pair<Dim, T>, MAX_DIMS_64>;
    using FlagsCont = std::array<bool, MAX_DIMS_64>;

public:
    template <bool IsConst>
    class Iterator final {
    public:
        using ValuesContInner = typename std::conditional<IsConst, const ValuesCont, ValuesCont>::type;
        using FlagsContInner = const FlagsCont;

        using value_type = typename std::conditional<IsConst, const std::pair<Dim, T>, std::pair<Dim, T>>::type;
        using pointer = value_type*;
        using reference = value_type&;
        using iterator_category = std::bidirectional_iterator_tag;
        using difference_type = std::ptrdiff_t;

        Iterator() = default;

        Iterator(const Iterator&) = default;
        Iterator& operator=(const Iterator&) = default;

        Iterator(Dim cur, ValuesContInner& values, FlagsContInner& flags) : _cur(cur), _values(&values), _flags(&flags) {
            advance();
        }

        reference operator*() const {
            auto curInd = static_cast<int32_t>(_cur);
            IE_ASSERT(curInd >= 0 && curInd < MAX_DIMS_64);
            IE_ASSERT((*_flags)[curInd]);

            return (*_values)[curInd];
        }

        Iterator& operator++() {
            auto curInd = static_cast<int32_t>(_cur);
            IE_ASSERT(curInd >= 0 && curInd < MAX_DIMS_64);
            IE_ASSERT((*_flags)[curInd]);

            _cur = static_cast<Dim>(static_cast<int32_t>(_cur) + 1);
            advance();
            return *this;
        }
        Iterator operator++(int) {
            auto curInd = static_cast<int32_t>(_cur);
            IE_ASSERT(curInd >= 0 && curInd < MAX_DIMS_64);
            IE_ASSERT((*_flags)[curInd]);

            auto tmp(*this);
            _cur = static_cast<Dim>(static_cast<int32_t>(_cur) + 1);
            advance();
            return tmp;
        }

        Iterator& operator--() {
            auto curInd = static_cast<int32_t>(_cur);
            IE_ASSERT(curInd >= 0 && curInd < MAX_DIMS_64);
            IE_ASSERT((*_flags)[curInd]);

            _cur = static_cast<Dim>(static_cast<int32_t>(_cur) - 1);
            moveBack();
            return *this;
        }
        Iterator operator--(int) {
            auto curInd = static_cast<int32_t>(_cur);
            IE_ASSERT(curInd >= 0 && curInd < MAX_DIMS_64);
            IE_ASSERT((*_flags)[curInd]);

            auto tmp(*this);
            _cur = static_cast<Dim>(static_cast<int32_t>(_cur) - 1);
            moveBack();
            return tmp;
        }

        bool operator==(const Iterator& other) const { return _cur == other._cur; }
        bool operator!=(const Iterator& other) const { return _cur != other._cur; }

    private:
        void advance() {
            auto curInd = static_cast<int32_t>(_cur);
            while (curInd >= 0 && curInd < MAX_DIMS_64 && !(*_flags)[curInd]) {
                ++curInd;
            }

            if (curInd == MAX_DIMS_64) {
                curInd = -1;
            }

            _cur = static_cast<Dim>(curInd);
        }

        void moveBack() {
            auto curInd = static_cast<int32_t>(_cur);
            while (curInd >= 0 && curInd < MAX_DIMS_64 && !(*_flags)[curInd]) {
                --curInd;
            }

            _cur = static_cast<Dim>(curInd);
        }

    private:
        Dim _cur = Dim::Invalid;
        ValuesContInner* _values;
        FlagsContInner* _flags;
    };

    using value_type = std::pair<Dim, T>;
    using iterator = Iterator<false>;
    using const_iterator = Iterator<true>;

    DimValues_() {
        _flags.fill(false);
    }
    explicit DimValues_(std::initializer_list<value_type> data) {
        _flags.fill(false);

        for (const auto& p : data) {
            auto ind = static_cast<int32_t>(p.first);
            IE_ASSERT(ind >= 0 && ind < MAX_DIMS_64);
            IE_ASSERT(!_flags[ind]);

            _values[ind] = p;
            _flags[ind] = true;
        }

        _size = data.size();
    }

    DimValues_(const DimValues_&) = default;
    DimValues_& operator=(const DimValues_&) = default;

    size_t size() const { return _size; }
    bool empty() const { return _size == 0; }

    void clear() {
        _flags.fill(false);
        _size = 0;
    }
    void erase(Dim d) {
        auto ind = static_cast<int32_t>(d);
        IE_ASSERT(ind >= 0 && ind < MAX_DIMS_64);

        if (_flags[ind]) {
            IE_ASSERT(_size > 0);

            _flags[ind] = false;
            --_size;
        }
    }

    bool has(Dim d) const {
        auto ind = static_cast<int32_t>(d);
        IE_ASSERT(ind >= 0 && ind < MAX_DIMS_64);

        return _flags[ind];
    }

    const T& operator[](Dim d) const {
        auto ind = static_cast<int32_t>(d);
        IE_ASSERT(ind >= 0 && ind < MAX_DIMS_64);
        IE_ASSERT(_flags[ind]);

        return _values[ind].second;
    }
    const T& get(Dim d, const T& def) const {
        auto ind = static_cast<int32_t>(d);
        IE_ASSERT(ind >= 0 && ind < MAX_DIMS_64);

        return _flags[ind] ? _values[ind].second : def;
    }

    void set(Dim d, const T& val) {
        auto ind = static_cast<int32_t>(d);
        IE_ASSERT(ind >= 0 && ind < MAX_DIMS_64);

        if (!_flags[ind]) {
            _flags[ind] = true;
            ++_size;
        }

        _values[ind] = std::make_pair(d, val);
    }

    iterator begin() { return iterator(Dim::W, _values, _flags); }
    iterator end() { return iterator(Dim::Invalid, _values, _flags); }

    const_iterator begin() const { return const_iterator(Dim::W, _values, _flags); }
    const_iterator end() const { return const_iterator(Dim::Invalid, _values, _flags); }

    const_iterator cbegin() const { return const_iterator(Dim::W, _values, _flags); }
    const_iterator cend() const { return const_iterator(Dim::Invalid, _values, _flags); }

    std::array<T, MAX_DIMS_64> toVector(const T& emptyValue) const {
        std::array<T, MAX_DIMS_64> out;
        out.fill(emptyValue);

        for (int ind = 0; ind < MAX_DIMS_64; ++ind) {
            if (_flags[ind]) {
                out[ind] = _values[ind].second;
            }
        }

        return out;
    }

    bool operator==(const DimValues_& other) const {
        for (int ind = 0; ind < MAX_DIMS_64; ++ind) {
            if (_flags[ind] != other._flags[ind]) {
                return false;
            }
            if (_flags[ind] && _values[ind].second != other._values[ind].second) {
                return false;
            }
        }
        return true;
    }
    bool operator!=(const DimValues_& other) const {
        for (int ind = 0; ind < MAX_DIMS_64; ++ind) {
            if (_flags[ind] != other._flags[ind]) {
                return true;
            }
            if (_flags[ind] && _values[ind].second != other._values[ind].second) {
                return true;
            }
        }
        return false;
    }

    void printTo(std::ostream& os) const {
        os << "[";

        int realInd = 0;
        for (int ind = 0; ind < MAX_DIMS_64; ++ind) {
            if (_flags[ind]) {
                vpu::printTo(os, _values[ind].first);
                os << " : ";
                vpu::printTo(os, _values[ind].second);
                if (realInd + 1 < _size) {
                    os << ", ";
                }
                ++realInd;
            }
        }

        os << "]";
    }

private:
    ValuesCont _values = {};
    FlagsCont _flags;
    size_t _size = 0;
};

template <typename T>
void printTo(std::ostream& os, const DimValues_<T>& dims) {
    dims.printTo(os);
}

using DimValues = DimValues_<int>;

//
// DimsOrder
//

StorageOrder64 maskOrder(StorageOrder64 fullOrder, int size);

class DimsOrder final {
public:
    //
    // Predefined orders
    //

    static DimsOrder C;
    static DimsOrder NC;
    static DimsOrder CHW;
    static DimsOrder HWC;
    static DimsOrder HCW;
    static DimsOrder NCHW;
    static DimsOrder NHWC;
    static DimsOrder NHCW;

    //
    // Constructor
    //

    DimsOrder() = default;
    static DimsOrder fromCode(StorageOrder64 code);
    static DimsOrder fromNumDims(int numDims);
    static DimsOrder fromPermutation(const SmallVector<Dim, MAX_DIMS_64>& perm);

    //
    // Accessors
    //

    bool empty() const { return _code == 0; }

    int numDims() const;

    bool hasDim(Dim d) const;
    int dimInd(Dim d) const;

    StorageOrder64 code() const { return _code; }

    //
    // Information about dimension order
    //

    // Convert from packed format to array of dimensions from minor to major.
    SmallVector<Dim, MAX_DIMS_64> toPermutation() const;

    // Get memory indeces for each dimension.
    DimValues toIndices() const;

    //
    // Relayout helpers
    //

    // In-place modification.
    void moveDim(Dim dim, int newPos);

    // Makes new object.
    DimsOrder createMovedDim(Dim dim, int newPos) const;

private:
    StorageOrder64 _code = 0;
};

bool isOrdersCompatible(DimsOrder order1, DimsOrder order2);

inline bool operator==(DimsOrder order1, DimsOrder order2) {
    return order1.code() == order2.code();
}
inline bool operator!=(DimsOrder order1, DimsOrder order2) {
    return order1.code() != order2.code();
}

void printTo(std::ostream& os, DimsOrder order);

struct DimsOrderHash final {
    size_t operator()(DimsOrder order) const {
        return std::hash<StorageOrder64>()(order.code());
    }
};

using DimsOrderSet = std::unordered_set<DimsOrder, DimsOrderHash>;
template <typename Val>
using DimsOrderMap = std::unordered_map<DimsOrder, Val, DimsOrderHash>;

//
// DataDesc
//

class DataDesc final {
public:
    //
    // Constructors
    //

    DataDesc() = default;

    template <typename IntValue, typename = typename std::enable_if<std::is_integral<IntValue>::value>::type>
    DataDesc(DataType type, DimsOrder dimsOrder, std::initializer_list<IntValue> dims) :
            _type(type), _dimsOrder(dimsOrder) {
        auto perm = _dimsOrder.toPermutation();
        IE_ASSERT(dims.size() == perm.size());

        int ind = 0;
        for (auto val : dims) {
            _dims.set(perm[ind], val);
            ++ind;
        }
    }

    template <typename IntValue, typename = typename std::enable_if<std::is_integral<IntValue>::value>::type>
    DataDesc(DimsOrder dimsOrder, std::initializer_list<IntValue> dims) : DataDesc(DataType::FP16, dimsOrder, dims) {}

    template <typename IntValue, typename = typename std::enable_if<std::is_integral<IntValue>::value>::type>
    explicit DataDesc(std::initializer_list<IntValue> dims) : DataDesc(DataType::FP16, DimsOrder::fromNumDims(dims.size()), dims) {}

    explicit DataDesc(const ie::TensorDesc& ieDesc);

    DataDesc(DataType type, DimsOrder dimsOrder, const DimValues& dims);

    //
    // DataType
    //

    DataType type() const { return _type; }

    void setType(DataType type) { _type = type; }

    int elemSize() const;

    //
    // Dims
    //

    int numDims() const { return _dimsOrder.numDims(); }

    const DimValues& dims() const { return _dims; }

    int dim(Dim d) const { return _dims[d]; }
    int dim(Dim d, int defVal) const { return _dims.has(d) ? _dims[d] : defVal; }

    void setDim(Dim d, int val);

    int totalDimSize() const;

    //
    // DimsOrder
    //

    DimsOrder dimsOrder() const { return _dimsOrder; }

    void moveDim(Dim dim, int newPos) {
        _dimsOrder.moveDim(dim, newPos);
    }

    void reorder(DimsOrder dimsOrder);

private:
    DataType _type = DataType::FP16;
    DimsOrder _dimsOrder;
    DimValues _dims;
};

void printTo(std::ostream& os, const DataDesc& desc);
void printTo(DotLabel& lbl, const DataDesc& desc);

//
// DimStride
//

VPU_DECLARE_ENUM(DimStride,
    Any,
    Compact,
    Aligned
)

const int STRIDE_ALIGNMENT = 16;

//
// StridesRequirement
//

//
// Container for stride requirement per each dimensions (in memory order).
//

class StridesRequirement final {
public:
    StridesRequirement() { _map[0] = DimStride::Compact; }

    static StridesRequirement empty() { return StridesRequirement().add(0, DimStride::Any); }
    static StridesRequirement compact();

    StridesRequirement& add(int index, DimStride stride) {
        IE_ASSERT(index >= 0 && index < MAX_DIMS_64);
        _map[index] = stride;
        return *this;
    }

    StridesRequirement& remove(int index) {
        IE_ASSERT(index >= 0 && index < MAX_DIMS_64);
        _map[index] = DimStride::Any;
        return *this;
    }

    DimStride get(int index) const {
        IE_ASSERT(index >= 0 && index < MAX_DIMS_64);
        return _map[index];
    }

    bool operator==(const StridesRequirement& other) const {
        return (_map == other._map);
    }
    bool operator!=(const StridesRequirement& other) const {
        return (_map != other._map);
    }

private:
    std::array<DimStride, MAX_DIMS_64> _map{{DimStride::Any}};
};

void printTo(std::ostream& os, const StridesRequirement& reqs);
void printTo(DotLabel& lbl, const StridesRequirement& reqs);

DimValues calcStrides(const DataDesc& desc, const StridesRequirement& reqs);

bool checkStride(
        const DimValues& strides,
        const DataDesc& desc,
        int ind,
        DimStride req);
bool checkStrides(
        const DataDesc& desc,
        const DimValues& strides,
        const StridesRequirement& reqs);

int calcTotalByteSize(const DataDesc& desc, const DimValues& strides);

//
// BatchSupport
//

VPU_DECLARE_ENUM(BatchSupport,
    Split,
    ReplicateConstContent
)

}  // namespace vpu

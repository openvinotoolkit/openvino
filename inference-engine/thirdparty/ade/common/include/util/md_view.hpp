// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_MD_VIEW_HPP
#define UTIL_MD_VIEW_HPP

#include <array>
#include <iterator>
#include <algorithm>
#include <numeric>

#include "util/math.hpp"
#include "util/assert.hpp"
#include "util/memory_range.hpp"
#include "util/checked_cast.hpp"
#include "util/md_size.hpp"
#include "util/md_span.hpp"
#include "util/iota_range.hpp"
#include "util/algorithm.hpp"

namespace util
{

/// Actual dimension spec (length + step)
struct SliceDimension final
{
   int length = 0;
   int step   = 0;
};

inline SliceDimension make_dimension(int l, int s) //TODO: move to C++14
{
    SliceDimension ret;
    ret.length = l;
    ret.step = s;
    return ret;
}

template<typename ParentT, typename DiffT = int>
class MdViewIteratorImpl final : public std::iterator<std::random_access_iterator_tag, DiffT>
{
   ParentT* m_parent = nullptr;
   int m_currentPos = -1;
   inline int checkPos(int pos) const
   {
      assert(nullptr != m_parent);
      assert(pos >= 0);
      assert(pos <= m_parent->dimensions.back().length); // equality for end iterators
      return pos;
   }
   inline void check() const
   {
      checkPos(m_currentPos);
   }

   using this_t = MdViewIteratorImpl<ParentT, DiffT>;
   inline static void check(const this_t& a1, const this_t& a2)
   {
      a1.check();
      a2.check();
      assert(a1.m_parent == a2.m_parent);
   }

   using diff_t = DiffT;
   using val_t = decltype(ParentT()[0]);
public:

   MdViewIteratorImpl() = default;
   MdViewIteratorImpl(ParentT* parent, int pos): m_parent(parent), m_currentPos(checkPos(pos)) {}
   MdViewIteratorImpl(const MdViewIteratorImpl&) = default;
   MdViewIteratorImpl& operator=(const MdViewIteratorImpl&) = default;

   inline this_t& operator+=(diff_t rhs) { check(); m_currentPos = checkPos(m_currentPos + rhs); return *this;}
   inline this_t& operator-=(diff_t rhs) { check(); m_currentPos = checkPos(m_currentPos - rhs); return *this;}
   inline val_t  operator*()  const { check(); return (*m_parent)[m_currentPos];}
   // TODO: Implement this using internally stored object
   // inline val_t* operator->() const { check(); return &((*m_parent)[m_currentPos]);}
   inline val_t& operator[](diff_t rhs) const { check(); return (*m_parent)[checkPos(m_currentPos + rhs)];}

   inline this_t& operator++() { check(); m_currentPos = checkPos(m_currentPos + 1); return *this; }
   inline this_t& operator--() { check(); m_currentPos = checkPos(m_currentPos - 1); return *this; }
   inline this_t operator++(int) const { check(); this_t tmp(*this); m_currentPos = checkPos(m_currentPos + 1); return tmp; }
   inline this_t operator--(int) const { check(); this_t tmp(*this); m_currentPos = checkPos(m_currentPos - 1); return tmp; }
   inline diff_t operator+(const this_t& rhs) const { check(*this, rhs); return m_currentPos + rhs.m_currentPos; }
   inline diff_t operator-(const this_t& rhs) const { check(*this, rhs); return m_currentPos - rhs.m_currentPos; }
   inline this_t operator+(diff_t rhs) const { check(); return this_t(m_parent,m_currentPos + rhs); }
   inline this_t operator-(diff_t rhs) const { check(); return this_t(m_parent,m_currentPos - rhs); }
   friend inline this_t operator+(diff_t lhs, const this_t& rhs) { rhs.check(); return this_t(rhs.m_parent,rhs.m_currentPos + lhs); }
   friend inline this_t operator-(diff_t lhs, const this_t& rhs) { rhs.check(); return this_t(rhs.m_parent,rhs.m_currentPos - lhs); }

   inline bool operator==(const this_t& rhs) const { check(*this, rhs); return m_currentPos == rhs.m_currentPos; }
   inline bool operator!=(const this_t& rhs) const { check(*this, rhs); return m_currentPos != rhs.m_currentPos; }
   inline bool operator>(const this_t& rhs)  const { check(*this, rhs); return m_currentPos >  rhs.m_currentPos; }
   inline bool operator<(const this_t& rhs)  const { check(*this, rhs); return m_currentPos <  rhs.m_currentPos; }
   inline bool operator>=(const this_t& rhs) const { check(*this, rhs); return m_currentPos >= rhs.m_currentPos; }
   inline bool operator<=(const this_t& rhs) const { check(*this, rhs); return m_currentPos <= rhs.m_currentPos; }
};

namespace details
{
template<typename T>
inline bool view_check_elem_size(std::size_t size)
{
    return size == sizeof(T);
}

template<>
inline bool view_check_elem_size<void>(std::size_t /*size*/)
{
    return true;
}

inline void mdview_copy_helper(std::size_t depth,
                               util::MemoryRange<void> src, const util::MemoryRange<const SliceDimension>& srcDims,
                               util::MemoryRange<void> dst, const util::MemoryRange<const SliceDimension>& dstDims)
{
    ASSERT(depth > 0);
    ASSERT(!srcDims.empty());
    ASSERT(srcDims.size == dstDims.size);
    ASSERT(srcDims.back().length == dstDims.back().length);
    if (srcDims.size > depth)
    {
        const auto srcStep = static_cast<std::size_t>(srcDims.back().step);
        const auto dstStep = static_cast<std::size_t>(dstDims.back().step);
        const auto len = static_cast<std::size_t>(srcDims.back().length);
        const auto& prevSrcDim = srcDims[srcDims.size - 2];
        const auto& prevDstDim = dstDims[dstDims.size - 2];
        const auto srcSliceSize = static_cast<std::size_t>(prevSrcDim.length * prevSrcDim.step);
        const auto dstSliceSize = static_cast<std::size_t>(prevDstDim.length * prevDstDim.step);
        for (auto i: util::iota(len))
        {
            mdview_copy_helper(depth,
                               util::slice(src, i * srcStep, srcSliceSize), util::slice(srcDims,0, srcDims.size - 1),
                               util::slice(dst, i * dstStep, dstSliceSize), util::slice(dstDims,0, dstDims.size - 1));
        }

    }
    else
    {
        util::raw_copy(src, dst);
    }
}
}

template<std::size_t MaxDimensions, typename DataT>
struct DynMdView;

/// Wrapper used to interpret raw memory range as arbitrary dimensional data (array, image, etc)
template<int Dimensions, typename DataT>
struct MdView final
{
   static_assert(Dimensions > 0, "Invalid dimensions");
   using this_t       = MdView<Dimensions, DataT>;
   using const_this_t = MdView<Dimensions, const DataT>;
   using value_type   = DataT;
   using dimensions_t = std::array<SliceDimension, Dimensions>;
   dimensions_t dimensions = {};
   util::MemoryRange<void> data;

   MdView()                         = default;
   MdView(const MdView&)            = default;
   MdView& operator=(const MdView&) = default;

   MdView(std::nullptr_t) {}

   MdView(const util::MemoryRange<void>& data_, const dimensions_t& dims):
      dimensions(dims), data(data_) {}

   template<std::size_t MaxDimensions, typename T>
   MdView(const DynMdView<MaxDimensions, T>& view)
   {
       ASSERT(nullptr != view);
       ASSERT(view.count() == Dimensions);
       ASSERT(details::view_check_elem_size<DataT>(view.elementSize()));
       std::copy(view.dimensions.begin(),
                 view.dimensions.begin() + Dimensions,
                 dimensions.begin());
       data = view.mem;
   }

   /// Single element size
   size_t ElementSize() const
   {
      return checked_cast<size_t>(dimensions[0].step);
   }

   /// Access raw data
   DataT* Data()
   {
      return static_cast<DataT*>(data.data);
   }

   /// Access raw data
   const DataT* Data() const
   {
      return static_cast<const DataT*>(data.data);
   }

   int Size() const
   {
      return std::accumulate(dimensions.begin(), dimensions.end(), int{1},
                             [](const int acc, const SliceDimension& dim){return acc*dim.length;});
   }

   template<typename Dummy = void> // hack for sfinae
   typename std::enable_if<(2 == Dimensions), typename std::conditional<true, int, Dummy>::type >::type
   Step() const
   {
      return dimensions[1].step;
   }

   int Width() const
   {
      return dimensions[0].length;
   }

   template<typename Dummy = void> // hack for sfinae
   typename std::enable_if<(Dimensions >= 2), typename std::conditional<true, int, Dummy>::type >::type
   Height() const
   {
      return dimensions[1].length;
   }

   template<typename Dummy = void> // hack for sfinae
   typename std::enable_if<(Dimensions >= 3), typename std::conditional<true, int, Dummy>::type >::type
   Depth() const
   {
      return dimensions[2].length;
   }

   template<typename Dummy = void> // hack for sfinae
   typename std::enable_if<(Dimensions > 1), typename std::conditional<true, MdView<Dimensions - 1, DataT>, Dummy>::type >::type
   operator[](int pos)
   {
      assert(pos >= 0);
      const auto& dim = dimensions.back();
      const auto& prevDim = dimensions[dimensions.size() - 2];
      assert(pos < dim.length);
      MdView<Dimensions - 1, DataT> ret;
      std::copy_n(dimensions.begin(), Dimensions - 1, ret.dimensions.begin());
      const auto dataSize = ElementSize() + (prevDim.length - 1) * prevDim.step;
      ret.data = data.Slice(dim.step * pos, dataSize);
      return ret;
   }

   template<typename Dummy = void> // hack for sfinae
   typename std::enable_if<(Dimensions > 1), typename std::conditional<true, MdView<Dimensions - 1, const DataT>, Dummy>::type >::type
   operator[](int pos) const
   {
      assert(pos >= 0);
      const auto& dim = dimensions.back();
      const auto& prevDim = dimensions[dimensions.size() - 2];
      assert(pos < dim.length);
      MdView<Dimensions - 1, const DataT> ret;
      std::copy_n(dimensions.begin(), Dimensions - 1, ret.dimensions.begin());
      const auto dataSize = ElementSize() + (prevDim.length - 1) * prevDim.step;
      ret.data = data.Slice(dim.step * pos, dataSize);
      return ret;
   }

   template<typename Dummy = DataT> // hack for sfinae
   typename std::enable_if<(Dimensions == 1 && sizeof(Dummy) > 0), Dummy& >::type
   operator[](int pos)
   {
      assert(pos >= 0);
      assert(pos < dimensions.back().length);
      return data.reinterpret<DataT>()[pos];
   }

   template<typename Dummy = DataT> // hack for sfinae
   typename std::enable_if<(Dimensions == 1 && sizeof(Dummy) > 0), const Dummy& >::type
   operator[](int pos) const
   {
      assert(pos >= 0);
      assert(pos < dimensions.back().length);
      return data.reinterpret<DataT>()[pos];
   }


   template<typename Dummy = void> // hack for sfinae
   using iterator       = MdViewIteratorImpl<MdView<Dimensions, DataT>>;

   template<typename Dummy = void> // hack for sfinae
   using const_iterator = MdViewIteratorImpl<const MdView<Dimensions, const DataT>>;

   template<typename Dummy = void> // hack for sfinae
   typename std::conditional<true, MdViewIteratorImpl<MdView<Dimensions, DataT>>, Dummy>::type
   begin()
   {
      return iterator<void>(this, 0);
   }

   template<typename Dummy = void> // hack for sfinae
   typename std::conditional<true, MdViewIteratorImpl<MdView<Dimensions, DataT>>, Dummy>::type
   end()
   {
      return iterator<void>(this, dimensions.back().length);
   }


   template<typename Dummy = void> // hack for sfinae
   typename std::conditional<true, MdViewIteratorImpl<const MdView<Dimensions, const DataT>>, Dummy>::type
   begin() const
   {
      return const_iterator<void>(this, 0);
   }

   template<typename Dummy = void> // hack for sfinae
   typename std::conditional<true, MdViewIteratorImpl<const MdView<Dimensions, const DataT>>, Dummy>::type
   end() const
   {
      return const_iterator<void>(this, dimensions.back().length);
   }
};

template<int Dimensions, typename DataT>
inline bool operator==(const MdView<Dimensions,DataT>& view, std::nullptr_t)
{
    return view.data == nullptr;
}

template<int Dimensions, typename DataT>
inline bool operator==(std::nullptr_t, const MdView<Dimensions,DataT>& view)
{
    return view.data == nullptr;
}

template<int Dimensions, typename DataT>
inline bool operator!=(const MdView<Dimensions,DataT>& view, std::nullptr_t)
{
    return view.data != nullptr;
}

template<int Dimensions, typename DataT>
inline bool operator!=(std::nullptr_t, const MdView<Dimensions,DataT>& view)
{
    return view.data != nullptr;
}

/// View typedef for scalar data (as one-element array)
template<typename DataT>
using ScalarView = MdView<1, DataT>;

/// View typedef for array data
template<typename DataT>
using ArrayView = MdView<1, DataT>;

/// View typedef for image data
template<typename DataT>
using ImageView = MdView<2, DataT>;

/// Dynamically sized memory view
template<std::size_t MaxDimensions, typename DataT>
struct DynMdView final
{
    using dimensions_t = std::array<SliceDimension, MaxDimensions>;
    dimensions_t dimensions;
    std::size_t dims_count = 0;
    util::MemoryRange<void> mem;

    DynMdView() = default;
    DynMdView(const DynMdView&) = default;
    DynMdView& operator=(const DynMdView&) = default;

    DynMdView(std::nullptr_t) {}

    DynMdView(const util::MemoryRange<void>& mem_,
              std::initializer_list<SliceDimension> dims):
        dims_count(util::checked_cast<decltype(this->dims_count)>(dims.size())),
        mem(mem_)
    {
        ASSERT(dims.size() <= MaxDimensions);
        std::copy(dims.begin(), dims.end(), dimensions.begin());
    }

    template<int Dimensions>
    DynMdView(const MdView<Dimensions, DataT>& src):
        dims_count(Dimensions)
    {
        static_assert(Dimensions <= MaxDimensions, "Invalid dimensions count");
        std::copy(src.dimensions.begin(),
                  src.dimensions.end(),
                  dimensions.begin());
        mem = src.data;
    }

    DataT* data()
    {
        ASSERT(nullptr != mem);
        return mem.data;
    }

    const DataT* data() const
    {
        ASSERT(nullptr != mem);
        return mem.data;
    }

    std::size_t count() const
    {
        return dims_count;
    }

    size_t elementSize() const
    {
        ASSERT(count() > 0);
        return checked_cast<size_t>(dimensions[0].step);
    }

    util::DynMdSize<MaxDimensions> size() const
    {
        util::DynMdSize<MaxDimensions> ret;
        ret.resize(count());
        for (auto i: util::iota(count()))
        {
            ret[i] = dimensions[i].length;
        }
        return ret;
    }

    DynMdView<MaxDimensions, DataT> slice(const DynMdSpan<MaxDimensions>& span) const
    {
        ASSERT(count() > 0);
        ASSERT(span.dims_count() == count());
        for (auto i: util::iota(count()))
        {
            ASSERT(span[i].begin <= span[i].end);
            ASSERT(span[i].begin >= 0 && span[i].end <= dimensions[i].length);
        }
        DynMdView<MaxDimensions, DataT> ret;
        ret.dims_count = count();

        std::size_t dataSize = elementSize();
        std::size_t dataOffset = 0;
        for (auto i: util::iota(count()))
        {
            dataOffset += dimensions[i].step * span[i].begin;
            ret.dimensions[i].length = span[i].length();
            ret.dimensions[i].step = dimensions[i].step;
            dataSize += (span[i].length() - 1) * dimensions[i].step;
        }
        ret.mem = mem.Slice(dataOffset, dataSize);
        return ret;
    }

    std::size_t sizeInBytes() const
    {
        ASSERT(nullptr != *this);
        ASSERT(count() > 0);
        std::size_t dataSize = elementSize();
        for (auto i: util::iota(count()))
        {
            dataSize += (dimensions[i].length - 1) * dimensions[i].step;
        }
        ASSERT(mem.size >= dataSize);
        return dataSize;
    }

    template<typename T>
    DynMdView<MaxDimensions, T> reinterpret() const
    {
        if (nullptr == *this)
        {
            return nullptr;
        }
        if (!std::is_void<T>::value)
        {
            const std::size_t new_elem_size = sizeof(util::conditional_t< std::is_void<T>::value, char, T >);
            ASSERT(elementSize() == new_elem_size);
        }
        DynMdView<MaxDimensions, T> ret;
        ret.dims_count = count();
        std::copy(dimensions.begin(), dimensions.begin() + count(), ret.dimensions.begin());
        ret.mem = mem;
        return ret;
    }
};

template<std::size_t MaxDimensions, typename DataT>
inline bool operator==(const DynMdView<MaxDimensions, DataT>& view, std::nullptr_t)
{
    return view.mem == nullptr;
}

template<std::size_t MaxDimensions, typename DataT>
inline bool operator==(std::nullptr_t, const DynMdView<MaxDimensions, DataT>& view)
{
    return view.mem == nullptr;
}

template<std::size_t MaxDimensions, typename DataT>
inline bool operator!=(const DynMdView<MaxDimensions, DataT>& view, std::nullptr_t)
{
    return view.mem != nullptr;
}

template<std::size_t MaxDimensions, typename DataT>
inline bool operator!=(std::nullptr_t, const DynMdView<MaxDimensions, DataT>& view)
{
    return view.mem != nullptr;
}

template<std::size_t MaxDimensions, typename Allocator, typename DimT, typename AlignT>
inline DynMdView<MaxDimensions, void> alloc_view(std::size_t elementSize,
                                                 const MemoryRange<DimT>& dimensions,
                                                 const MemoryRange<AlignT>& alignment,
                                                 Allocator&& alloc)
{
    ASSERT(elementSize > 0);
    ASSERT(size(dimensions) == size(alignment));
    ASSERT(util::all_of(dimensions, [](DimT dim)
    {
       return dim > 0;
    }));
    ASSERT(util::all_of(alignment, [](AlignT align)
    {
       return align > 0;
    }));
    const auto dims_count = size(dimensions);
    ASSERT(dims_count > 0);
    ASSERT(dims_count <= MaxDimensions);

    auto blockAlign = checked_cast<std::size_t>(alignment[0]);
    auto blockSize  = checked_cast<std::size_t>(dimensions[0] * elementSize);
    DynMdView<MaxDimensions, void> ret;
    ret.dims_count = util::checked_cast<decltype(ret.dims_count)>(dims_count);
    using LengthT = decltype(ret.dimensions[0].length);
    using StepT   = decltype(ret.dimensions[0].step);
    ret.dimensions[0].length = util::checked_cast<LengthT>(dimensions[0]);
    ret.dimensions[0].step   = util::checked_cast<StepT>(elementSize);
    for (auto i: util::iota(static_cast<decltype(dims_count)>(1), dims_count))
    {
       blockAlign = std::max(blockAlign, static_cast<std::size_t>(alignment[i]));
       const auto step = align_size(blockSize, static_cast<std::size_t>(alignment[i]));
       blockSize = step * dimensions[i];
       ret.dimensions[i].length = util::checked_cast<LengthT>(dimensions[i]);
       ret.dimensions[i].step   = util::checked_cast<StepT>(step);
    }

    ret.mem = memory_range(static_cast<void*>(alloc(blockSize, blockAlign)), blockSize);
    return ret;
}

/// Copy data from one view to other
/// Both viewvs must have same dimensions
template<std::size_t MaxDimensions, typename DataT>
inline void view_copy(const DynMdView<MaxDimensions, DataT>& src, const DynMdView<MaxDimensions, DataT>& dst)
{
    ASSERT(src.count() > 0);
    ASSERT(src.count() == dst.count());
    ASSERT(nullptr != src);
    ASSERT(nullptr != dst);
    ASSERT(src.elementSize() == dst.elementSize());
    ASSERT(src.dimensions[0].length == dst.dimensions[0].length);
    auto elemSize = src.elementSize() * src.dimensions[0].length;

    for (auto i: util::iota(static_cast<decltype(src.count())>(1), src.count()))
    {
        if (elemSize != src.dimensions[i].step ||
            elemSize != dst.dimensions[i].step)
        {
            //First bad stride, copy all previous dimesions as single buffer
            details::mdview_copy_helper(i,
                                        src.mem, util::memory_range(src.dimensions.data(), src.count()),
                                        dst.mem, util::memory_range(dst.dimensions.data(), dst.count()));
            return;
        }
        ASSERT(src.dimensions[i].length == dst.dimensions[i].length);
        elemSize *= src.dimensions[i].length;
    }
    // All strides were good, copy entire buffer
    details::mdview_copy_helper(src.count(),
                                src.mem, util::memory_range(src.dimensions.data(), src.count()),
                                dst.mem, util::memory_range(dst.dimensions.data(), dst.count()));
}

}
#endif // UTIL_MD_VIEW_HPP

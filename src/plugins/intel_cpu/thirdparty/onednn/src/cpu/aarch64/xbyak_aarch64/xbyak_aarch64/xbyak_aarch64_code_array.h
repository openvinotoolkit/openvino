#pragma once
/*******************************************************************************
 * Copyright 2019-2021 FUJITSU LIMITED
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "xbyak_aarch64_err.h"
#include "xbyak_aarch64_inner.h"

static const size_t CSIZE = sizeof(uint32_t);

inline void *AlignedMalloc(size_t size, size_t alignment) {
#ifdef _MSC_VER
  return _aligned_malloc(size, alignment);
#else
  void *p;
  int ret = posix_memalign(&p, alignment, size);
  return (ret == 0) ? p : 0;
#endif
}

inline void AlignedFree(void *p) {
#ifdef _MSC_VER
  _aligned_free(p);
#else
  free(p);
#endif
}

template <class To, class From> inline const To CastTo(From p) throw() { return (const To)(size_t)(p); }

struct Allocator {
  virtual uint32_t *alloc(size_t size) { return reinterpret_cast<uint32_t *>(AlignedMalloc(size, inner::getPageSize())); }
  virtual void free(uint32_t *p) { AlignedFree(p); }
  virtual ~Allocator() {}
  /* override to return false if you call protect() manually */
  virtual bool useProtect() const { return true; }
};

#ifdef XBYAK_USE_MMAP_ALLOCATOR
class MmapAllocator : Allocator {
  typedef std::unordered_map<uintptr_t, size_t> SizeList;
  SizeList sizeList_;

public:
  uint32_t *alloc(size_t size) {
    const size_t alignedSizeM1 = inner::getPageSize() - 1;
    size = (size + alignedSizeM1) & ~alignedSizeM1;
#ifdef MAP_ANONYMOUS
    int mode = MAP_PRIVATE | MAP_ANONYMOUS;
#elif defined(MAP_ANON)
    int mode = MAP_PRIVATE | MAP_ANON;
#else
#error "not supported"
#endif
#ifdef XBYAK_USE_MAP_JIT
    mode |= MAP_JIT;
#endif
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, mode, -1, 0);
    if (p == MAP_FAILED)
      throw Error(ERR_CANT_ALLOC);
    assert(p);
    sizeList_[(uintptr_t)p] = size;
    return (uint32_t *)p;
  }
  void free(uint32_t *p) {
    if (p == 0)
      return;
    SizeList::iterator i = sizeList_.find((uintptr_t)p);
    if (i == sizeList_.end())
      throw Error(ERR_BAD_PARAMETER);
    if (munmap((void *)i->first, i->second) < 0)
      throw Error(ERR_MUNMAP);
    sizeList_.erase(i);
  }
};
#endif

// 2nd parameter for constructor of CodeArray(maxSize, userPtr, alloc)
void *const AutoGrow = (void *)1;          //-V566
void *const DontSetProtectRWE = (void *)2; //-V566

class CodeArray {
  enum Type {
    USER_BUF = 1, // use userPtr(non alignment, non protect)
    ALLOC_BUF,    // use new(alignment, protect)
    AUTO_GROW     // automatically move and grow memory if necessary
  };
  CodeArray(const CodeArray &rhs);
  void operator=(const CodeArray &);
  bool isAllocType() const { return type_ == ALLOC_BUF || type_ == AUTO_GROW; }

  // type of partially applied function for encoding
  typedef std::function<uint32_t(int64_t)> EncFunc;

  struct AddrInfo {
    size_t codeOffset; // position to write
    size_t jmpAddr;    // value to write
    EncFunc encFunc;   // encoding function
    AddrInfo(size_t _codeOffset, size_t _jmpAddr, EncFunc encFunc) : codeOffset(_codeOffset), jmpAddr(_jmpAddr), encFunc(encFunc) {}
    uint32_t getVal() const { return encFunc((int64_t)(jmpAddr - codeOffset) * CSIZE); }
  };

  typedef std::list<AddrInfo> AddrInfoList;
  AddrInfoList addrInfoList_;
  const Type type_;
#ifdef XBYAK_USE_MMAP_ALLOCATOR
  MmapAllocator defaultAllocator_;
#else
  Allocator defaultAllocator_;
#endif
  Allocator *alloc_;

protected:
  friend class LabelManager;
  size_t maxSize_; // max size of code size (per uint32_t)
  uint32_t *top_;
  size_t size_; // code size
  bool isCalledCalcJmpAddress_;

  bool useProtect() const { return alloc_->useProtect(); }
  /*
    allocate new memory and copy old data to the new area
  */
  void growMemory() {
    const size_t newSize = (std::max<size_t>)(DEFAULT_MAX_CODE_SIZE, getMaxSize() * 2);
    uint32_t *newTop = alloc_->alloc(newSize);
    if (newTop == 0)
      throw Error(ERR_CANT_ALLOC);
    for (size_t i = 0; i < size_; i++)
      newTop[i] = top_[i];
    alloc_->free(top_);
    top_ = newTop;
    maxSize_ = newSize / CSIZE;
  }
  /*
    calc jmp address for AutoGrow mode
  */
  void calcJmpAddress() {
    if (isCalledCalcJmpAddress_)
      return;
    for (AddrInfoList::const_iterator i = addrInfoList_.begin(), ie = addrInfoList_.end(); i != ie; ++i) {
      uint32_t disp = i->getVal();
      rewrite(i->codeOffset, disp);
    }
    isCalledCalcJmpAddress_ = true;
  }

public:
  enum ProtectMode {
    PROTECT_RW = 0,  // read/write
    PROTECT_RWE = 1, // read/write/exec
    PROTECT_RE = 2   // read/exec
  };
  explicit CodeArray(size_t maxSize, void *userPtr = 0, Allocator *allocator = 0)
      : type_(userPtr == AutoGrow ? AUTO_GROW : (userPtr == 0 || userPtr == DontSetProtectRWE) ? ALLOC_BUF : USER_BUF), alloc_(allocator ? allocator : (Allocator *)&defaultAllocator_), maxSize_(maxSize / CSIZE),
        top_(type_ == USER_BUF ? reinterpret_cast<uint32_t *>(userPtr) : alloc_->alloc((std::max<size_t>)(maxSize, CSIZE))), size_(0), isCalledCalcJmpAddress_(false) {
    if (maxSize_ > 0 && top_ == 0)
      throw Error(ERR_CANT_ALLOC);
    if ((type_ == ALLOC_BUF && userPtr != DontSetProtectRWE && useProtect()) && !setProtectMode(PROTECT_RWE, false)) {
      alloc_->free(top_);
      throw Error(ERR_CANT_PROTECT);
    }
  }
  virtual ~CodeArray() {
    if (isAllocType()) {
      if (useProtect())
        setProtectModeRW(false);
      alloc_->free(top_);
    }
  }
  bool setProtectMode(ProtectMode mode, bool throwException = true) {
    bool isOK = protect(top_, getMaxSize(), mode);
    if (isOK)
      return true;
    if (throwException)
      throw Error(ERR_CANT_PROTECT);
    return false;
  }
  bool setProtectModeRE(bool throwException = true) { return setProtectMode(PROTECT_RE, throwException); }
  bool setProtectModeRW(bool throwException = true) { return setProtectMode(PROTECT_RW, throwException); }
  void resetSize() {
    size_ = 0;
    addrInfoList_.clear();
    isCalledCalcJmpAddress_ = false;
  }
  void clearCodeArray() {
    for (size_t i = 0; i < size_; i++) {
      top_[i] = 0;
    }
    size_ = 0;
  }

  // write 4 byte data
  void dd(uint32_t code) {
    if (size_ >= maxSize_) {
      if (type_ == AUTO_GROW) {
        growMemory();
      } else {
        throw Error(ERR_CODE_IS_TOO_BIG);
      }
    }
    top_[size_++] = code;
  }
  const uint8_t *getCode() const { return reinterpret_cast<uint8_t *>(top_); }
  template <class F> const F getCode() const { return reinterpret_cast<F>(top_); }
  const uint8_t *getCurr() const { return reinterpret_cast<uint8_t *>(&top_[size_]); }
  template <class F> const F getCurr() const { return reinterpret_cast<F>(&top_[size_]); }
  // return byte size
  size_t getSize() const { return size_ * CSIZE; }
  size_t getMaxSize() const { return maxSize_ * CSIZE; }
  // set byte size
  void setSize(size_t size) {
    if (size > getMaxSize())
      throw Error(ERR_OFFSET_IS_TOO_BIG);
    size_ = size / CSIZE;
  }
  void dump() const {
    for (size_t i = 0; i < size_; ++i) {
      printf("%08X\n", top_[i]);
    }
  }
  /*
    @param offset [in] offset from top
    @param disp [in] offset from the next of jmp
  */
  void rewrite(size_t offset, uint32_t disp) {
    assert(offset < maxSize_);
    uint32_t *const data = top_ + offset;
    *data = disp;
  }
  void save(size_t offset, size_t jmpAddr, const EncFunc &encFunc) { addrInfoList_.push_back(AddrInfo(offset, jmpAddr, encFunc)); }
  bool isAutoGrow() const { return type_ == AUTO_GROW; }
  bool isCalledCalcJmpAddress() const { return isCalledCalcJmpAddress_; }
  /**
     change exec permission of memory
     @param addr [in] buffer address
     @param size [in] buffer size
     @param protectMode [in] mode(RW/RWE/RE)
     @return true(success), false(failure)
  */
  static inline bool protect(const void *addr, size_t size, int protectMode) {
#if defined(_WIN32)
    const DWORD c_rw = PAGE_READWRITE;
    const DWORD c_rwe = PAGE_EXECUTE_READWRITE;
    const DWORD c_re = PAGE_EXECUTE_READ;
    DWORD mode;
#else
    const int c_rw = PROT_READ | PROT_WRITE;
    const int c_rwe = PROT_READ | PROT_WRITE | PROT_EXEC;
    const int c_re = PROT_READ | PROT_EXEC;
    int mode;
#endif

    switch (protectMode) {
    case PROTECT_RW:
      mode = c_rw;
      break;
    case PROTECT_RWE:
      mode = c_rwe;
      break;
    case PROTECT_RE:
      mode = c_re;
      break;
    default:
      return false;
    }
#if defined(__GNUC__) || defined(__APPLE__)
    size_t pageSize = inner::getPageSize();
    size_t iaddr = reinterpret_cast<size_t>(addr);
    size_t roundAddr = iaddr & ~(pageSize - static_cast<size_t>(1));

    return mprotect(reinterpret_cast<void *>(roundAddr), size + (iaddr - roundAddr), mode) == 0;
#elif defined(_WIN32)
    DWORD oldProtect;
    return VirtualProtect(const_cast<void *>(addr), size, mode, &oldProtect) != 0;
#else
    return true;
#endif
  }
  /**
     get aligned memory pointer
     @param addr [in] address
     @param alignedSize [in] power of two
     @return aligned addr by alingedSize
  */
  static inline uint32_t *getAlignedAddress(uint32_t *addr, size_t alignedSize = 16) { return reinterpret_cast<uint32_t *>((reinterpret_cast<size_t>(addr) + alignedSize - 1) & ~(alignedSize - static_cast<size_t>(1))); }
};

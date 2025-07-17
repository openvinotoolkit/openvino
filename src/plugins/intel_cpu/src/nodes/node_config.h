// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

class PortDescBase {
public:
    virtual ~PortDescBase() = default;

    /**
     * @brief Check if the port desc can be accepted.
     *
     * @warning This operation is not commutative desc1.isCompatible(desc2) != desc2.isCompatible(desc1) in general case
     *
     * @return True if the port desc may be accepted false otherwise
     */
    [[nodiscard]] bool isCompatible(const PortDescBase& rhs) const {
        return typeid(*this) == typeid(rhs) && this->compareImpl(rhs);
    }
    [[nodiscard]] virtual MemoryDescPtr getMemDesc() const = 0;

protected:
    [[nodiscard]] virtual bool compareImpl(const PortDescBase& rhs) const = 0;
};

using PortDescBasePtr = std::shared_ptr<PortDescBase>;
using PortDescBaseCPtr = std::shared_ptr<const PortDescBase>;

template <class T>
class PortDescBase_ : public PortDescBase {
protected:
    PortDescBase_() = default;
    [[nodiscard]] bool compareImpl(const PortDescBase& rhs) const override /*final*/ {
        return static_cast<const T&>(*this).isCompatible(static_cast<const T&>(rhs));
    }
};

class PortDescGeneric : public PortDescBase_<PortDescGeneric> {
public:
    explicit PortDescGeneric(MemoryDescPtr memDesc) : _memDesc(std::move(memDesc)) {
        OPENVINO_ASSERT(_memDesc, "ParameterMismatch: PortDescGeneric constructor got nullptr");
    }
    [[nodiscard]] bool isCompatible(const PortDescGeneric& rhs) const {
        return _memDesc->isCompatible(*rhs._memDesc);
    }
    [[nodiscard]] MemoryDescPtr getMemDesc() const override {
        return _memDesc;
    }

private:
    MemoryDescPtr _memDesc;
};

class PortDescBlocked : public PortDescBase_<PortDescBlocked> {
public:
    using CmpMask = BlockedMemoryDesc::CmpMask;

    PortDescBlocked(BlockedMemoryDescPtr memDesc, CmpMask cmpMask) : _memDesc(std::move(memDesc)), _cmpMask(cmpMask) {
        OPENVINO_ASSERT(_memDesc, "ParameterMismatch: PortDescBlocked constructor got nullptr");
    }
    [[nodiscard]] bool isCompatible(const PortDescBlocked& rhs) const {
        return _memDesc->isCompatible(*rhs._memDesc, _cmpMask) && (((~_cmpMask) | rhs._cmpMask).all());
    }
    [[nodiscard]] MemoryDescPtr getMemDesc() const override {
        return _memDesc;
    }

private:
    BlockedMemoryDescPtr _memDesc;
    CmpMask _cmpMask = BlockedMemoryDesc::FULL_MASK;
};

class PortConfig {
public:
    PortConfig() = default;

    PortConfig(const MemoryDescPtr& desc,
               BlockedMemoryDesc::CmpMask cmpMask = BlockedMemoryDesc::FULL_MASK,
               int inPlacePort = -1,
               bool isConstant = false)
        : _desc(createPortDesc(desc, cmpMask)),
          _inPlacePort(inPlacePort),
          _constant(isConstant) {}

    // prevent implicit convertion of cmpMask
    PortConfig(MemoryDescPtr desc, int cmpMask, int inPlacePort = -1, bool isConstant = false) = delete;

    PortConfig(const PortConfig& rhs) = default;

    PortConfig& operator=(const PortConfig& rhs) = default;

    PortConfig(PortConfig&& rhs) = default;
    PortConfig& operator=(PortConfig&& rhs) = default;

    [[nodiscard]] int inPlace() const {
        return _inPlacePort;
    }

    void inPlace(int port) {
        _inPlacePort = port;
    }

    [[nodiscard]] bool constant() const {
        return _constant;
    }

    void constant(bool constant) {
        _constant = constant;
    }

    [[nodiscard]] MemoryDescPtr getMemDesc() const {
        return _desc->getMemDesc();
    }

    [[nodiscard]] PortDescBasePtr getPortDesc() const {
        return _desc;
    }

    void setMemDesc(const MemoryDescPtr& desc) {
        _desc = createPortDesc(desc, BlockedMemoryDesc::FULL_MASK);
    }

    void setMemDesc(const BlockedMemoryDescPtr& desc, BlockedMemoryDesc::CmpMask cmpMask) {
        _desc = createPortDesc(desc, cmpMask);
    }

    [[nodiscard]] bool hasZeroDims() const {
        const auto desc = getMemDesc();
        return desc->getShape().hasZeroDims() && !desc->empty();
    }

private:
    static PortDescBasePtr createPortDesc(const MemoryDescPtr& desc, BlockedMemoryDesc::CmpMask cmpMask) {
        if (desc->getType() & Blocked) {
            return createPortDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(desc), cmpMask);
        }

        return std::make_shared<PortDescGeneric>(desc);
    }

    static PortDescBasePtr createPortDesc(const BlockedMemoryDescPtr& desc, BlockedMemoryDesc::CmpMask cmpMask) {
        return std::make_shared<PortDescBlocked>(desc, cmpMask);
    }

    PortDescBasePtr _desc;
    int _inPlacePort = -1;
    bool _constant = false;
};

struct NodeConfig {
    NodeConfig() = default;

    NodeConfig(std::vector<PortConfig> inConfs, std::vector<PortConfig> outConfs)
        : inConfs(std::move(inConfs)),
          outConfs(std::move(outConfs)) {}

    std::vector<PortConfig> inConfs;
    std::vector<PortConfig> outConfs;
};

}  // namespace ov::intel_cpu

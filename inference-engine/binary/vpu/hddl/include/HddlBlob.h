//
// Copyright © 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version October 2018). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#ifndef HDDL_CLIENT_HDDLBLOB_H
#define HDDL_CLIENT_HDDLBLOB_H

#include "HddlCommon.h"
#include <atomic>
#include <memory>
#include <tuple>

namespace hddl {
class HddlBlobImpl;
class HddlClientImpl;
class HDDL_EXPORT_API HddlBlob {
public:
    HddlBlobImpl* m_impl;

    HddlBlob();
    virtual ~HddlBlob();

    HddlBlob(const HddlBlob&) = delete;
    HddlBlob& operator=(const HddlBlob&) = delete;

    virtual int allocate(size_t size);
    int         reallocate(size_t size);

    void*       getData();
    const void* getData() const;
    size_t      getSize() const;

    /* setRange() sets only the range from offset
     * to (offset+size) of the buffer will be used in inference. */
    void                        setRange(size_t offset, size_t size);
    std::tuple<size_t, size_t>  getRange() const;

protected:
    void setAuxImpl(HddlBlobImpl* impl);
};

class HddlAuxBlobImpl;
class HDDL_EXPORT_API  HddlAuxBlob : public HddlBlob
{
public:
    typedef std::shared_ptr<HddlAuxBlob> Ptr;

    HddlAuxBlob(HddlAuxInfoType auxInfoType);
    ~HddlAuxBlob();

    HddlAuxBlob(const HddlAuxBlob&) = delete;
    HddlAuxBlob& operator=(const HddlAuxBlob&) = delete;

    const void* getAuxData(HddlAuxInfoType auxInfoType, size_t* size) const;
    int allocate(size_t size);

private:
    friend class HddlClientImpl;
    HddlAuxBlobImpl* m_auxImpl;
};


} // namespace hddl

#endif //HDDL_SERVICE_HDDLBUFFER_H

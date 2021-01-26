/*********************************************************************************************************************************************************************************************************************************************************************************************
#   Intel® Single Event API
#
#   This file is provided under the BSD 3-Clause license.
#   Copyright (c) 2021, Intel Corporation
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#       Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#       Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#       Neither the name of the Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#   IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
**********************************************************************************************************************************************************************************************************************************************************************************************/

#import <Cocoa/Cocoa.h>

// custom instrument path: ~/Library/Application Support/Instruments/PlugIns/Instruments
// examples: https://github.com/maddox/regexkit/tree/master/Source/DTrace
//           https://github.com/andreberg/BMScript/tree/master/DTrace/Instruments

@interface IntelSEAPI : NSObject {}
    + (void)task_begin :(const char*)name domain:(const char*)domain taskid:(uint64_t)taskid parentid:(uint64_t)parentid;
    + (void)task_end: (const char*)name domain:(const char*)domain taskid:(uint64_t)taskid meta:(const char*)meta;
    + (void)task_begin_overlapped :(const char*)name domain:(const char*)domain taskid:(uint64_t)taskid parentid:(uint64_t)parentid;
    + (void)task_end_overlapped :(const char*)name domain:(const char*)domain taskid:(uint64_t)taskid meta:(const char*)meta;
    + (void)counter :(const char*)name domain:(const char*)domain value:(int64_t)value;
    + (void)marker :(const char*)name domain:(const char*)domain id:(int64_t)id scope:(const char*)scope;
@end

@implementation IntelSEAPI {}
    + (void)task_begin :(const char*)name domain:(const char*)domain taskid:(uint64_t)taskid parentid:(uint64_t)parentid;{}
    + (void)task_end :(const char*)name domain:(const char*)domain taskid:(uint64_t)taskid meta:(const char*)meta;{}
    + (void)task_begin_overlapped :(const char*)name domain:(const char*)domain taskid:(uint64_t)taskid parentid:(uint64_t)parentid;{}
    + (void)task_end_overlapped :(const char*)name domain:(const char*)domain taskid:(uint64_t)taskid meta:(const char*)meta;{}
    + (void)counter :(const char*)name domain:(const char*)domain value:(int64_t)value;{}
    + (void)marker :(const char*)name domain:(const char*)domain id:(int64_t)id scope:(const char*)scope;{}
@end

void DTraceTaskBegin(const char* domain, uint64_t taskid, uint64_t parentid, const char* name)
{
    [IntelSEAPI task_begin :name domain:domain taskid:taskid parentid:parentid];
}

void DTraceTaskEnd(const char* domain, uint64_t taskid, const char* meta, const char* name)
{
    [IntelSEAPI task_end :name domain:domain taskid:taskid meta:meta];
}

void DTraceTaskBeginOverlapped(const char* domain, uint64_t taskid, uint64_t parentid, const char* name)
{
    [IntelSEAPI task_begin_overlapped :name domain:domain taskid:taskid parentid:parentid];
}

void DTraceTaskEndOverlapped(const char* domain, uint64_t taskid, const char* meta, const char* name)
{
    [IntelSEAPI task_end_overlapped :name domain:domain taskid:taskid meta:meta];
}

void DTraceTaskCounter(const char* domain, int64_t value, const char* name)
{
    [IntelSEAPI counter :name domain:domain value:value];
}

void DTraceMarker(const char* domain, uint64_t id, const char* name, const char* scope)
{
    [IntelSEAPI marker :name domain:domain id:id scope:scope];
}


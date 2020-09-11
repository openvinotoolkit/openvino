// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "win_pthread.h"
#include <string.h>
#include <stdlib.h>

#include "XLinkStringUtils.h"

typedef HRESULT(WINAPI* SetThreadDescriptionDecl)(HANDLE hThread,
                                                  PCWSTR lpThreadDescription);
typedef HRESULT(WINAPI* GetThreadDescriptionDecl)(HANDLE hThread,
                                                  PWSTR* ppszThreadDescription);

//Mutex implementation
int pthread_mutex_lock(pthread_mutex_t *mutex)
{
    EnterCriticalSection(mutex);
    return 0;
}

int pthread_mutex_unlock(pthread_mutex_t *mutex)
{
    LeaveCriticalSection(mutex);
    return 0;
}

int pthread_mutex_init(pthread_mutex_t *mutex,
                       pthread_mutexattr_t *attr)
{
    (void)attr;
    InitializeCriticalSection(mutex);

    return 0;
}

int pthread_mutex_destroy(pthread_mutex_t *mutex)
{
    DeleteCriticalSection(mutex);
    return 0;
}

//State implementation
unsigned _pthread_get_state(pthread_attr_t *attr, unsigned flag)
{
    return attr->pthread_state & flag;
}

int _pthread_set_state(pthread_attr_t *attr, unsigned flag, unsigned val)
{
    if (~flag & val) return EINVAL;
    attr->pthread_state &= ~flag;
    attr->pthread_state |= val;

    return 0;
}

//Attribute implementation
int pthread_attr_init(pthread_attr_t *attr)
{
    attr->pthread_state = PTHREAD_CANCEL_ENABLE;
    attr->stack = NULL;
    attr->stack_size = 0;
    return 0;
}

int pthread_attr_destroy(pthread_attr_t *attr)
{
    (void)attr;
    return 0;
}

int pthread_attr_setinheritsched(pthread_attr_t *attr, int flag)
{
    return _pthread_set_state(attr, PTHREAD_INHERIT_SCHED, flag);
}

int pthread_attr_getinheritsched(pthread_attr_t *attr, int *flag)
{
    *flag = _pthread_get_state(attr, PTHREAD_INHERIT_SCHED);
    return 0;
}

#define pthread_attr_getschedpolicy(ATTR, POLICY) ENOTSUP
#define pthread_attr_setschedpolicy(ATTR, POLICY) ENOTSUP


//Pthread creation
unsigned int __stdcall _pthread_start_routine(void *args)
{
    pthread_t *thread = args;
    thread->tid = GetCurrentThreadId();
    thread->arg = thread->start_routine(thread->arg);
    return 0;
}

int pthread_create(pthread_t *thread, pthread_attr_t *attr,
                   void *(*start_routine)(void *), void *arg)
{
    unsigned stack_size = 0;

    /* Save data in pthread_t */
    thread->arg = arg;
    thread->start_routine = start_routine;
    thread->pthread_state = PTHREAD_CANCEL_ENABLE;
    thread->handle = (HANDLE)-1;
    _ReadWriteBarrier();

    if (attr)
    {
        thread->pthread_state = attr->pthread_state;
        stack_size = attr->stack_size;
    }

    thread->handle = (HANDLE)_beginthreadex((void *)NULL, stack_size, _pthread_start_routine, thread, 0, NULL);

    /* Failed */
    if (!thread->handle)
        return 1;

    return 0;
}

int pthread_detach(pthread_t thread)
{
    CloseHandle(thread.handle);
    _ReadWriteBarrier();
    thread.handle = 0;

    return 0;
}

int _pthread_join(pthread_t *thread, void **res)
{

    DWORD result = WaitForSingleObject(thread->handle, INFINITE);

    switch (result) {
        case WAIT_OBJECT_0:
            if (res)
                *res = thread->arg;
            return 0;
        case WAIT_ABANDONED:
            return EINVAL;
        default:
            return 1;
    }
}

pthread_t pthread_self(void)
{
    pthread_t t = { 0 };

    t.tid = GetCurrentThreadId();
    t.arg = NULL;
    t.start_routine = NULL;
    t.pthread_state = PTHREAD_CANCEL_ENABLE;
    t.handle = GetCurrentThread();

    return t;
}

void pthread_exit(void *res)
{
    if(res)
    {
        _endthreadex(*(int *)res);
    }
    else
        _endthreadex(0);
}

#define MAX_THREAD_NAME_LENGHT 256

int pthread_setname_np(pthread_t target_thread, const char * name)
{
    wchar_t wstr[MAX_THREAD_NAME_LENGHT];
    mbstowcs(wstr, name, MAX_THREAD_NAME_LENGHT);

    SetThreadDescriptionDecl set_thread_description_func =
        (SetThreadDescriptionDecl)(GetProcAddress(
            GetModuleHandle("Kernel32.dll"), "SetThreadDescription"));

    if (!set_thread_description_func) {
        return 0;
    }

    HRESULT hr = set_thread_description_func(target_thread.handle, wstr);

    if (FAILED(hr))
    {
        return 1;
    }
    return 0;
}

int pthread_getname_np(pthread_t target_thread, char *buf, size_t len)
{
    wchar_t* wstr;

    GetThreadDescriptionDecl get_thread_description_func =
        (GetThreadDescriptionDecl)(GetProcAddress(
            GetModuleHandle("Kernel32.dll"), "GetThreadDescription"));

    if (!get_thread_description_func) {
        return 0;
    }

    HRESULT hr = get_thread_description_func(target_thread.handle, &wstr);
    if (FAILED(hr))
    {
        return 1;
    }

    char data[MAX_THREAD_NAME_LENGHT];
    wcstombs(data, wstr, MAX_THREAD_NAME_LENGHT);
    LocalFree(wstr);

    mv_strncpy(buf, len, data, len - 1);
    return 0;
}

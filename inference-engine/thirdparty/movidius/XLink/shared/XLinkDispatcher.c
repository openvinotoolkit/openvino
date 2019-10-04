// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// @file
///
/// @brief     Application configuration Leon header
///
#ifndef _GNU_SOURCE
#define _GNU_SOURCE // fix for warning: implicit declaration of function ‘pthread_setname_np’
#endif

#include "stdio.h"
#include "stdint.h"
#include "stdlib.h"
#include "string.h"
#include <assert.h>
#include <stdlib.h>

#if (defined(_WIN32) || defined(_WIN64))
# include "win_pthread.h"
# include "win_semaphore.h"
#else
# include <pthread.h>
# ifndef __APPLE__
#  include <semaphore.h>
# endif
#endif

#include "XLinkDispatcher.h"
#include "XLinkPrivateDefines.h"
#include "XLink.h"
#include "XLink_tool.h"

#define MVLOG_UNIT_NAME xLink
#include "mvLog.h"

typedef enum {
    EVENT_ALLOCATED,
    EVENT_PENDING,
    EVENT_BLOCKED,
    EVENT_READY,
    EVENT_SERVED,
} xLinkEventState_t;

typedef struct xLinkEventPriv_t {
    xLinkEvent_t packet;
    xLinkEvent_t *retEv;
    xLinkEventState_t isServed;
    xLinkEventOrigin_t origin;
    sem_t* sem;
    void* data;
    uint32_t pad;
} xLinkEventPriv_t;

typedef struct {
    sem_t sem;
    pthread_t threadId;
    int refs;
} localSem_t;

typedef struct{
    xLinkEventPriv_t* end;
    xLinkEventPriv_t* base;

    xLinkEventPriv_t* curProc;
    xLinkEventPriv_t* cur;
    __attribute__((aligned(64))) xLinkEventPriv_t q[MAX_EVENTS];

}eventQueueHandler_t;
/**
 * @brief Scheduler for each device
 */
typedef struct {
    xLinkDeviceHandle_t deviceHandle; //will be device handler
    int schedulerId;

    int queueProcPriority;

    sem_t addEventSem;
    sem_t notifyDispatcherSem;
    volatile uint32_t resetXLink;
    uint32_t semaphores;
    pthread_t xLinkThreadId;

    eventQueueHandler_t lQueue; //local queue
    eventQueueHandler_t rQueue; //remote queue
    localSem_t eventSemaphores[MAXIMUM_SEMAPHORES];
} xLinkSchedulerState_t;

extern char* TypeToStr(int type);

#if (defined(_WIN32) || defined(_WIN64))
static void* __cdecl eventSchedulerRun(void* ctx);
#else
static void* eventSchedulerRun(void*);
#endif
//These will be common for all, Initialized only once
struct dispatcherControlFunctions* glControlFunc;
int numSchedulers;
xLinkSchedulerState_t schedulerState[MAX_SCHEDULERS];
sem_t addSchedulerSem;

//below workaround for "C2088 '==': illegal for struct" error
int pthread_t_compare(pthread_t a, pthread_t b)
{
#if (defined(_WIN32) || defined(_WIN64) )
    return ((a.tid == b.tid));
#else
    return  (a == b);
#endif
}

static int unrefSem(sem_t* sem,  xLinkSchedulerState_t* curr) {
    ASSERT_X_LINK(curr != NULL);
    localSem_t* temp = curr->eventSemaphores;
    while (temp < curr->eventSemaphores + MAXIMUM_SEMAPHORES) {
        if (&temp->sem == sem) {
            temp->refs--;
            if (temp->refs == 0) {
                curr->semaphores--;
                ASSERT_X_LINK(sem_destroy(&temp->sem) != -1);
                temp->refs = -1;
            }
            return 1;
        }
        temp++;
    }
    mvLog(MVLOG_WARN,"unrefSem : sem wasn't found\n");
    return 0;
}
static sem_t* getCurrentSem(pthread_t threadId, xLinkSchedulerState_t* curr, int inc_ref)
{
    ASSERT_X_LINK_R(curr != NULL, NULL);

    localSem_t* sem = curr->eventSemaphores;
    while (sem < curr->eventSemaphores + MAXIMUM_SEMAPHORES) {
        if (pthread_t_compare(sem->threadId, threadId) && sem->refs > 0) {
            sem->refs += inc_ref;
            return &sem->sem;
        }
        sem++;
    }
    return NULL;
}

static sem_t* createSem(xLinkSchedulerState_t* curr)
{
    ASSERT_X_LINK_R(curr != NULL, NULL);


    sem_t* sem = getCurrentSem(pthread_self(), curr, 0);
    if (sem) // it already exists, error
        return NULL;
    else
    {
        if (curr->semaphores < MAXIMUM_SEMAPHORES) {
            localSem_t* temp = curr->eventSemaphores;
            while (temp < curr->eventSemaphores + MAXIMUM_SEMAPHORES) {
                if (temp->refs < 0) {
                    sem = &temp->sem;
                    if (temp->refs == -1) {
                        if (sem_init(sem, 0, 0))
                            perror("Can't create semaphore\n");
                    }
                    curr->semaphores++;
                    temp->refs = 1;
                    temp->threadId = pthread_self();

                    break;
                }
                temp++;
            }
            if (!sem)
                return NULL;
        }
        else
            return NULL;
        return sem;
    }
}

#if (defined(_WIN32) || defined(_WIN64))
static void* __cdecl eventReader(void* ctx)
#else
static void* eventReader(void* ctx)
#endif
{
    xLinkSchedulerState_t *curr = (xLinkSchedulerState_t*)ctx;
    ASSERT_X_LINK_R(curr, NULL);

    xLinkEvent_t event = { 0 };// to fix error C4700 in win
    event.header.id = -1;
    event.deviceHandle = curr->deviceHandle;

    mvLog(MVLOG_INFO,"eventReader thread started");

    while (!curr->resetXLink) {
        int sc = glControlFunc->eventReceive(&event);
        mvLog(MVLOG_DEBUG,"Reading %s (scheduler %d, fd %p, event id %d, event stream_id %u, event size %u)\n",
              TypeToStr(event.header.type), curr->schedulerId, event.deviceHandle.xLinkFD, event.header.id, event.header.streamId, event.header.size);

#ifdef __PC__
        if (event.header.type == XLINK_RESET_RESP) {
            curr->resetXLink = 1;
            mvLog(MVLOG_INFO,"eventReader thread stopped: reset");
            break;
        }
#endif

        if (sc) {
            // Only run this logic on the host side, the FW does not need this logic
#ifdef __PC__
            if (sem_post(&curr->notifyDispatcherSem)) {
                mvLog(MVLOG_ERROR,"can't post semaphore\n"); // stop eventSchedulerRun thread
            }
            mvLog(MVLOG_ERROR,"eventReader thread stopped (err %d)", sc);
#endif
            break;
        }
    }
    return 0;
}

static int isEventTypeRequest(xLinkEventPriv_t* event)
{
    if (event->packet.header.type < XLINK_REQUEST_LAST)
        return 1;
    else
        return 0;
}

static void markEventBlocked(xLinkEventPriv_t* event)
{
    event->isServed = EVENT_BLOCKED;
}

static void markEventReady(xLinkEventPriv_t* event)
{
    event->isServed = EVENT_READY;
}

static void eventPost(xLinkEventPriv_t* event)
{
    if (event->retEv){
        // the xLinkEventPriv_t slot pointed by "event" will be
        // re-cycled as soon as we mark it as EVENT_SERVED,
        // so before that, we copy the result event into XLink API layer
        *(event->retEv) = event->packet;
    }
    if(event->sem){
        if (sem_post(event->sem)) {
            mvLog(MVLOG_ERROR,"can't post semaphore\n");
        }
    }
}

static void markEventServed(xLinkEventPriv_t* event)
{
    eventPost(event);
    event->isServed = EVENT_SERVED;
}

static int dispatcherRequestServe(xLinkEventPriv_t * event, xLinkSchedulerState_t* curr){
    ASSERT_X_LINK(curr != NULL);
    ASSERT_X_LINK(isEventTypeRequest(event));
    xLinkEventHeader_t *header = &event->packet.header;
    if (header->flags.bitField.block){ //block is requested
        markEventBlocked(event);
    } else if(header->flags.bitField.localServe == 1 ||
              (header->flags.bitField.ack == 0
               && header->flags.bitField.nack == 1)){ //this event is served locally, or it is failed
#ifdef __PC__
        markEventServed(event);
#else
        eventPost(event);
        return 1;
#endif
    }else if (header->flags.bitField.ack == 1
              && header->flags.bitField.nack == 0){
        event->isServed = EVENT_PENDING;
        mvLog(MVLOG_DEBUG,"------------------------UNserved %s\n",
              TypeToStr(event->packet.header.type));
    }else{
        ASSERT_X_LINK(0);
    }
    return 0;
}


static int dispatcherResponseServe(xLinkEventPriv_t * event, xLinkSchedulerState_t* curr)
{
    int i = 0;
    ASSERT_X_LINK(curr != NULL);
    ASSERT_X_LINK(!isEventTypeRequest(event));
    for (i = 0; i < MAX_EVENTS; i++)
    {
        xLinkEventHeader_t *header = &curr->lQueue.q[i].packet.header;
        xLinkEventHeader_t *evHeader = &event->packet.header;

        if (curr->lQueue.q[i].isServed == EVENT_PENDING &&
            header->id == evHeader->id &&
            header->type == evHeader->type - XLINK_REQUEST_LAST -1)
        {
            mvLog(MVLOG_DEBUG,"----------------------ISserved %s\n",
                  TypeToStr(header->type));
            //propagate back flags
            header->flags = evHeader->flags;
            markEventServed(&curr->lQueue.q[i]);
            break;
        }
    }
    if (i == MAX_EVENTS) {
        mvLog(MVLOG_FATAL,"no request for this response: %s %d\n", TypeToStr(event->packet.header.type), event->origin);
        printf("#### (i == MAX_EVENTS) %s %d %d\n", TypeToStr(event->packet.header.type), event->origin, (int)event->packet.header.id);
        for (i = 0; i < MAX_EVENTS; i++)
        {
            xLinkEventHeader_t *header = &curr->lQueue.q[i].packet.header;

            printf("%d) header->id %i, header->type %s(%i), curr->lQueue.q[i].isServed %i, EVENT_PENDING %i\n", i, (int)header->id
                , TypeToStr(header->type), header->type, curr->lQueue.q[i].isServed, EVENT_PENDING);

        }
        ASSERT_X_LINK(0);
    }
    return 0;
}

static inline xLinkEventPriv_t* getNextElementWithState(xLinkEventPriv_t* base, xLinkEventPriv_t* end,
                                                        xLinkEventPriv_t* start, xLinkEventState_t state){
    xLinkEventPriv_t* tmp = start;
    while (start->isServed != state){
        CIRCULAR_INCREMENT_BASE(start, end, base);
        if(tmp == start){
            break;
        }
    }
    if(start->isServed == state){
        return start;
    }else{
        return NULL;
    }
}

static xLinkEventPriv_t* searchForReadyEvent(xLinkSchedulerState_t* curr)
{
    ASSERT_X_LINK_R(curr != NULL, NULL);
    xLinkEventPriv_t* ev = NULL;

    ev = getNextElementWithState(curr->lQueue.base, curr->lQueue.end, curr->lQueue.base, EVENT_READY);
    if(ev){
        mvLog(MVLOG_DEBUG,"ready %s %d \n",
              TypeToStr((int)ev->packet.header.type),
              (int)ev->packet.header.id);
    }
    return ev;
}

static xLinkEventPriv_t* getNextQueueElemToProc(eventQueueHandler_t *q ){
    xLinkEventPriv_t* event = NULL;
    if (q->cur != q->curProc) {
        event = getNextElementWithState(q->base, q->end, q->curProc, EVENT_ALLOCATED);
        q->curProc = event;
        CIRCULAR_INCREMENT_BASE(q->curProc, q->end, q->base);
    }
    return event;
}

/**
 * @brief Add event to Queue
 * @note It called from dispatcherAddEvent
 */
static xLinkEvent_t* addNextQueueElemToProc(xLinkSchedulerState_t* curr,
                                            eventQueueHandler_t *q, xLinkEvent_t* event,
                                            sem_t* sem, xLinkEventOrigin_t o){
    xLinkEvent_t* ev;
    xLinkEventPriv_t* eventP = getNextElementWithState(q->base, q->end, q->cur, EVENT_SERVED);
    if (eventP == NULL) {
        mvLog(MVLOG_ERROR, "getNextElementWithState returned NULL");
        return NULL;
    }
    mvLog(MVLOG_DEBUG, "Received event %s %d", TypeToStr(event->header.type), o);
    ev = &eventP->packet;
    if (eventP->sem) {
        if ((XLinkError_t)unrefSem(eventP->sem,  curr) == X_LINK_ERROR) {
            mvLog(MVLOG_WARN, "Failed to unref sem");
        }
    }
    eventP->sem = sem;
    eventP->packet = *event;
    eventP->origin = o;
    if (o == EVENT_LOCAL) {
        // XLink API caller provided buffer for return the final result to
        eventP->retEv = event;
    }else{
        eventP->retEv = NULL;
    }
    q->cur = eventP;
    eventP->isServed = EVENT_ALLOCATED;
    CIRCULAR_INCREMENT_BASE(q->cur, q->end, q->base);
    return ev;
}

static xLinkEventPriv_t* dispatcherGetNextEvent(xLinkSchedulerState_t* curr)
{
    ASSERT_X_LINK_R(curr != NULL, NULL);

    if (XLinkWaitSem(&curr->notifyDispatcherSem)) {
        mvLog(MVLOG_ERROR,"can't post semaphore\n");
    }

    xLinkEventPriv_t* event = NULL;
    event = searchForReadyEvent(curr);
    if (event) {
        return event;
    }

    eventQueueHandler_t* hPriorityQueue = curr->queueProcPriority ? &curr->lQueue : &curr->rQueue;
    eventQueueHandler_t* lPriorityQueue = curr->queueProcPriority ? &curr->rQueue : &curr->lQueue;
    curr->queueProcPriority = curr->queueProcPriority ? 0 : 1;

    event = getNextQueueElemToProc(hPriorityQueue);
    if (event) {
        return event;
    }
    event = getNextQueueElemToProc(lPriorityQueue);

    return event;
}

static pthread_mutex_t reset_mutex = PTHREAD_MUTEX_INITIALIZER;

static int isAvailableScheduler(xLinkSchedulerState_t* curr)
{
    if (curr->schedulerId == -1) {
        mvLog(MVLOG_WARN,"Scheduler has already been reset or cleaned");
        return 0; // resetted already
    }
    return 1;
}

static void closeDeviceFdAndResetScheduler(xLinkSchedulerState_t* curr)
{
    mvLog(MVLOG_INFO, "Dispatcher Cleaning...");
    glControlFunc->closeDeviceFd(&curr->deviceHandle);
    curr->schedulerId = -1;
    curr->resetXLink = 1;
    sem_destroy(&curr->addEventSem);
    sem_destroy(&curr->notifyDispatcherSem);
    localSem_t* temp = curr->eventSemaphores;
    while (temp < curr->eventSemaphores + MAXIMUM_SEMAPHORES) {
        // unblock potentially blocked event semaphores
        sem_post(&temp->sem);
        sem_destroy(&temp->sem);
        temp->refs = -1;
        temp++;
    }
    numSchedulers--;
    mvLog(MVLOG_INFO,"Cleaning Successfully\n");
}

static int dispatcherReset(xLinkSchedulerState_t* curr)
{
    ASSERT_X_LINK(curr != NULL);
#ifdef __PC__
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&reset_mutex), 1);

    if(!isAvailableScheduler(curr)) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&reset_mutex));
        return 1;
    }
#endif

    mvLog(MVLOG_INFO, "Resetting...");

    glControlFunc->closeLink(curr->deviceHandle.xLinkFD, 1);
    if (sem_post(&curr->notifyDispatcherSem)) {
        mvLog(MVLOG_ERROR,"can't post semaphore\n"); //to allow us to get a NULL event
    }
    xLinkEventPriv_t* event = dispatcherGetNextEvent(curr);
    while (event != NULL) {
        mvLog(MVLOG_INFO, "dropped event is %s, status %d\n",
              TypeToStr(event->packet.header.type), event->isServed);

#ifdef __PC__
        markEventServed(event);
#endif
        event = dispatcherGetNextEvent(curr);
    }

    event = getNextElementWithState(curr->lQueue.base, curr->lQueue.end, curr->lQueue.base, EVENT_PENDING);
    while (event != NULL) {
        mvLog(MVLOG_INFO,"Pending event is %s, size is %d, Mark it served\n", TypeToStr(event->packet.header.type), event->packet.header.size);
        markEventServed(event);
        event = getNextElementWithState(curr->lQueue.base, curr->lQueue.end, curr->lQueue.base, EVENT_PENDING);
    }

#ifdef __PC__
    closeDeviceFdAndResetScheduler(curr);
    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&reset_mutex));
#else
    glControlFunc->closeDeviceFd(&curr->deviceHandle);
    curr->schedulerId = -1;
    numSchedulers--;
#endif

    mvLog(MVLOG_DEBUG,"Reset Successfully\n");
    return 0;
}

#if (defined(_WIN32) || defined(_WIN64))
static void* __cdecl eventSchedulerRun(void* ctx)
#else
static void* eventSchedulerRun(void* ctx)
#endif
{
    int schedulerId = *((int*) ctx);
    mvLog(MVLOG_DEBUG,"%s() schedulerId %d\n", __func__, schedulerId);
    ASSERT_X_LINK_R(schedulerId < MAX_SCHEDULERS, NULL);

    xLinkSchedulerState_t* curr = &schedulerState[schedulerId];
    pthread_t readerThreadId;        /* Create thread for reader.
                        This thread will notify the dispatcher of any incoming packets*/
    pthread_attr_t attr;
    int sc;
    int res;
    if (pthread_attr_init(&attr) != 0) {
        mvLog(MVLOG_ERROR,"pthread_attr_init error");
        return NULL;
    }
#ifndef __PC__
    if (pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED) != 0) {
        pthread_attr_destroy(&attr);
        mvLog(MVLOG_ERROR,"pthread_attr_setinheritsched error");
        return NULL;
    }
    if (pthread_attr_setschedpolicy(&attr, SCHED_RR) != 0) {
        pthread_attr_destroy(&attr);
        mvLog(MVLOG_ERROR,"pthread_attr_setschedpolicy error");
        return NULL;
    }
#endif
    sc = pthread_create(&readerThreadId, &attr, eventReader, curr);
    if (sc) {
        mvLog(MVLOG_ERROR, "Thread creation failed");
        if (pthread_attr_destroy(&attr) != 0) {
            perror("Thread attr destroy failed\n");
        }
        return NULL;
    }
#ifndef __APPLE__
    char eventReaderThreadName[MVLOG_MAXIMUM_THREAD_NAME_SIZE];
    snprintf(eventReaderThreadName, sizeof(eventReaderThreadName), "EventRead%.2dThr", schedulerId);
    sc = pthread_setname_np(readerThreadId, eventReaderThreadName);
    if (sc != 0) {
        perror("Setting name for event reader thread failed");
    }
#endif
#ifdef __PC__
    sc = pthread_attr_destroy(&attr);
    if (sc) {
        mvLog(MVLOG_WARN, "Thread attr destroy failed");
    }
#endif

    xLinkEventPriv_t* event;
    xLinkEventPriv_t response;

    mvLog(MVLOG_INFO,"Scheduler thread started");

    while (!curr->resetXLink) {
        event = dispatcherGetNextEvent(curr);
        if(event == NULL)
        {
            mvLog(MVLOG_ERROR,"Dispatcher received NULL event!");
            /// Skip the event instead of asserting, so only
            /// the particular xlink chan will crash
#ifdef __PC__
            break;
#else
            continue;
#endif
        }
        ASSERT_X_LINK_R(event->packet.deviceHandle.xLinkFD == curr->deviceHandle.xLinkFD, NULL);
        getRespFunction getResp;
        xLinkEvent_t* toSend;

        if (event->origin == EVENT_LOCAL){
            getResp = glControlFunc->localGetResponse;
            toSend = &event->packet;
        }else{
            getResp = glControlFunc->remoteGetResponse;
            toSend = &response.packet;
        }

        res = getResp(&event->packet, &response.packet);
        if (isEventTypeRequest(event)){
            int served = 0;
            if (event->origin == EVENT_LOCAL){ //we need to do this for locals only
                served = dispatcherRequestServe(event, curr);
            }
            if (res == 0 && event->packet.header.flags.bitField.localServe == 0) {
#ifndef __PC__
                /*
                 * Device part: reset device if sending failed
                 */
                ASSERT_X_LINK_R(glControlFunc->eventSend(toSend) == 0, NULL);
#else
                (void)served;
                if (toSend->header.type == XLINK_RESET_REQ) {
                    if(toSend->deviceHandle.protocol == X_LINK_PCIE) {
                        toSend->header.type = XLINK_PING_REQ;
                        curr->resetXLink = 1;
                        mvLog(MVLOG_DEBUG, "Request for reboot not sent, only ping event");
                    } else {
#if defined(NO_BOOT)
                        toSend->header.type = XLINK_PING_REQ;
                        curr->resetXLink = 1;
                        mvLog(MVLOG_INFO, "Request for reboot not sent, only ping event");
#endif
                    }
                }

                if (glControlFunc->eventSend(toSend) != 0) {
                    mvLog(MVLOG_ERROR, "Event sending failed");
                }
#endif
            }
#ifndef __PC__
            if (event->origin == EVENT_REMOTE || served) {
                event->isServed = EVENT_SERVED;
            }
#endif
        } else {
            if (event->origin == EVENT_REMOTE){ // match remote response with the local request
                dispatcherResponseServe(event, curr);
            }
#ifndef __PC__
            event->isServed = EVENT_SERVED;
#endif
        }

        //TODO: dispatcher shouldn't know about this packet. Seems to be easily move-able to protocol
#ifndef __PC__
        if (event->origin == EVENT_REMOTE) {
            if (event->packet.header.type == XLINK_RESET_REQ) {
                curr->resetXLink = 1;
            }
        }
#else
        if (event->packet.header.type == XLINK_RESET_REQ) {
            curr->resetXLink = 1;
        }

        // remote event is served in one round
        if (event->origin == EVENT_REMOTE){
            event->isServed = EVENT_SERVED;
        }
#endif
    }
    sc = pthread_join(readerThreadId, NULL);
    if (sc) {
        mvLog(MVLOG_ERROR, "Waiting for thread failed");
    }

#ifndef __PC__
    if (pthread_attr_destroy(&attr) != 0) {
        mvLog(MVLOG_ERROR,"pthread_attr_destroy error");
        return NULL;
    }
#endif

    if (dispatcherReset(curr) != 0) {
        mvLog(MVLOG_WARN, "Failed to reset");
    }

    if (curr->resetXLink != 1) {
        mvLog(MVLOG_ERROR,"Scheduler thread stopped");
    } else {
        mvLog(MVLOG_INFO,"Scheduler thread stopped");
    }

    return NULL;
}

static int createUniqueID()
{
    static int id = 0xa;
    return id++;
}

static xLinkSchedulerState_t* findCorrespondingScheduler(void* xLinkFD)
{
    int i;
    if (xLinkFD == NULL) { //in case of myriad there should be one scheduler
        if (numSchedulers == 1)
            return &schedulerState[0];
        else
            NULL;
    }
    for (i=0; i < MAX_SCHEDULERS; i++)
        if (schedulerState[i].schedulerId != -1 &&
            schedulerState[i].deviceHandle.xLinkFD == xLinkFD)
            return &schedulerState[i];

    return NULL;
}
///////////////// External Interface //////////////////////////
/*Adds a new event with parameters and returns event id*/
xLinkEvent_t* dispatcherAddEvent(xLinkEventOrigin_t origin, xLinkEvent_t *event)
{
    xLinkSchedulerState_t* curr = findCorrespondingScheduler(event->deviceHandle.xLinkFD);
    ASSERT_X_LINK_R(curr != NULL, NULL);

    if(curr->resetXLink) {
        return NULL;
    }
    mvLog(MVLOG_DEBUG, "Receiving event %s %d\n", TypeToStr(event->header.type), origin);
    if (XLinkWaitSem(&curr->addEventSem)) {
        mvLog(MVLOG_ERROR,"can't wait semaphore\n");
        return NULL;
    }

    sem_t *sem = NULL;
    xLinkEvent_t* ev;
    if (origin == EVENT_LOCAL) {
        event->header.id = createUniqueID();
        sem = getCurrentSem(pthread_self(), curr, 1);
        if (!sem) {
            sem = createSem(curr);
        }
        if (!sem) {
            mvLog(MVLOG_WARN,"No more semaphores. Increase XLink or OS resources\n");
            if (sem_post(&curr->addEventSem)) {
                mvLog(MVLOG_ERROR,"can't post semaphore\n");
            }

            return NULL;
        }
        event->header.flags.raw = 0;
        event->header.flags.bitField.ack = 1;
        ev = addNextQueueElemToProc(curr, &curr->lQueue, event, sem, origin);
    } else {
        ev = addNextQueueElemToProc(curr, &curr->rQueue, event, NULL, origin);
    }
    if (sem_post(&curr->addEventSem)) {
        mvLog(MVLOG_ERROR,"can't post semaphore\n");
    }
    if (sem_post(&curr->notifyDispatcherSem)) {
        mvLog(MVLOG_ERROR, "can't post semaphore\n");
    }
    return ev;
}

int dispatcherWaitEventComplete(xLinkDeviceHandle_t* deviceHandle, unsigned int timeout)
{
    xLinkSchedulerState_t* curr = findCorrespondingScheduler(deviceHandle->xLinkFD);
    ASSERT_X_LINK(curr != NULL);

    sem_t* id = getCurrentSem(pthread_self(), curr, 0);
    if (id == NULL) {
        return -1;
    }
#ifndef __PC__
    (void)timeout;
    return XLinkWaitSem(id);
#else
    int rc = XLinkWaitSemUserMode(id, timeout);
    if (rc) {
        xLinkEvent_t event = {0};
        event.header.type = XLINK_RESET_REQ;
        event.deviceHandle = *deviceHandle;
        mvLog(MVLOG_ERROR,"waiting is timeout, sending reset remote event");
        dispatcherAddEvent(EVENT_LOCAL, &event);
        id = getCurrentSem(pthread_self(), curr, 0);
        if (id == NULL || XLinkWaitSemUserMode(id, timeout)) {
            dispatcherReset(curr);
        }
    }

    return rc;
#endif
}

int dispatcherUnblockEvent(eventId_t id, xLinkEventType_t type, streamId_t stream, void* xLinkFD)
{
    xLinkSchedulerState_t* curr = findCorrespondingScheduler(xLinkFD);
    ASSERT_X_LINK(curr != NULL);

    mvLog(MVLOG_DEBUG,"unblock\n");
    xLinkEventPriv_t* blockedEvent;
    for (blockedEvent = curr->lQueue.q;
         blockedEvent < curr->lQueue.q + MAX_EVENTS;
         blockedEvent++)
    {
        if (blockedEvent->isServed == EVENT_BLOCKED &&
            ((blockedEvent->packet.header.id == id || id == -1)
             && blockedEvent->packet.header.type == type
             && blockedEvent->packet.header.streamId == stream))
        {
            mvLog(MVLOG_DEBUG,"unblocked**************** %d %s\n",
                  (int)blockedEvent->packet.header.id,
                  TypeToStr((int)blockedEvent->packet.header.type));
            markEventReady(blockedEvent);
            if (sem_post(&curr->notifyDispatcherSem)){
                mvLog(MVLOG_ERROR, "can't post semaphore\n");
            }
            return 1;
        } else {
            mvLog(MVLOG_DEBUG,"%d %s\n",
                  (int)blockedEvent->packet.header.id,
                  TypeToStr((int)blockedEvent->packet.header.type));
        }
    }
    return 0;
}

int findAvailableScheduler()
{
    int i;
    for (i = 0; i < MAX_SCHEDULERS; i++)
        if (schedulerState[i].schedulerId == -1)
            return i;
    return -1;
}

/**
 * Initialize scheduler for device
 */
int dispatcherStart(xLinkDeviceHandle_t* deviceHandle)
{
#ifdef __PC__
    if (deviceHandle->xLinkFD == NULL) {
        mvLog(MVLOG_ERROR, "Invalid device filedescriptor");
        return -1;
    }
#endif

    pthread_attr_t attr;
    int eventIdx;
    if (numSchedulers >= MAX_SCHEDULERS)
    {
        mvLog(MVLOG_ERROR,"Max number Schedulers reached!\n");
        return -1;
    }
    int idx = findAvailableScheduler();
    if (idx == -1) {
        mvLog(MVLOG_ERROR,"Max number Schedulers reached!\n");
        return -1;
    }

    memset(&schedulerState[idx], 0, sizeof(xLinkSchedulerState_t));

    schedulerState[idx].semaphores = 0;
    schedulerState[idx].queueProcPriority = 0;

    schedulerState[idx].resetXLink = 0;
    schedulerState[idx].deviceHandle = *deviceHandle;
    schedulerState[idx].schedulerId = idx;

    schedulerState[idx].lQueue.cur = schedulerState[idx].lQueue.q;
    schedulerState[idx].lQueue.curProc = schedulerState[idx].lQueue.q;
    schedulerState[idx].lQueue.base = schedulerState[idx].lQueue.q;
    schedulerState[idx].lQueue.end = &schedulerState[idx].lQueue.q[MAX_EVENTS];

    schedulerState[idx].rQueue.cur = schedulerState[idx].rQueue.q;
    schedulerState[idx].rQueue.curProc = schedulerState[idx].rQueue.q;
    schedulerState[idx].rQueue.base = schedulerState[idx].rQueue.q;
    schedulerState[idx].rQueue.end = &schedulerState[idx].rQueue.q[MAX_EVENTS];

    for (eventIdx = 0 ; eventIdx < MAX_EVENTS; eventIdx++)
    {
        schedulerState[idx].rQueue.q[eventIdx].isServed = EVENT_SERVED;
        schedulerState[idx].lQueue.q[eventIdx].isServed = EVENT_SERVED;
    }

    if (sem_init(&schedulerState[idx].addEventSem, 0, 1)) {
        perror("Can't create semaphore\n");
        return -1;
    }
    if (sem_init(&schedulerState[idx].notifyDispatcherSem, 0, 0)) {
        perror("Can't create semaphore\n");
    }
    localSem_t* temp = schedulerState[idx].eventSemaphores;
    while (temp < schedulerState[idx].eventSemaphores + MAXIMUM_SEMAPHORES) {
        temp->refs = -1;
        temp++;
    }
    if (pthread_attr_init(&attr) != 0) {
        mvLog(MVLOG_ERROR,"pthread_attr_init error");
#ifdef __PC__
        return -1;
#endif
    }

#ifndef __PC__
    if (pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED) != 0) {
        mvLog(MVLOG_ERROR,"pthread_attr_setinheritsched error");
        pthread_attr_destroy(&attr);
    }
    if (pthread_attr_setschedpolicy(&attr, SCHED_RR) != 0) {
        mvLog(MVLOG_ERROR,"pthread_attr_setschedpolicy error");
        pthread_attr_destroy(&attr);
    }
#endif

    XLinkWaitSem(&addSchedulerSem);
    mvLog(MVLOG_DEBUG,"%s() starting a new thread - schedulerId %d \n", __func__, idx);
    int sc = pthread_create(&schedulerState[idx].xLinkThreadId,
                            &attr,
                            eventSchedulerRun,
                            (void*)&schedulerState[idx].schedulerId);
    if (sc) {
        mvLog(MVLOG_ERROR,"Thread creation failed with error: %d", sc);
        if (pthread_attr_destroy(&attr) != 0) {
            perror("Thread attr destroy failed\n");
        }
#ifdef __PC__
        return -1;
#endif
    }
#ifndef __APPLE__
    char schedulerThreadName[MVLOG_MAXIMUM_THREAD_NAME_SIZE];
    snprintf(schedulerThreadName, sizeof(schedulerThreadName), "Scheduler%.2dThr", schedulerState[idx].schedulerId);
    sc = pthread_setname_np(schedulerState[idx].xLinkThreadId, schedulerThreadName);
    if (sc != 0) {
        perror("Setting name for indexed scheduler thread failed");
    }
#endif
#ifdef __PC__
    pthread_detach(schedulerState[idx].xLinkThreadId);
#endif

    numSchedulers++;
    if (pthread_attr_destroy(&attr) != 0) {
        mvLog(MVLOG_ERROR,"pthread_attr_destroy error");
    }

    sem_post(&addSchedulerSem);

    return 0;
}

int dispatcherInitialize(struct dispatcherControlFunctions* controlFunc) {
    // create thread which will communicate with the pc

    int i;
    if (!controlFunc ||
        !controlFunc->eventReceive ||
        !controlFunc->eventSend ||
        !controlFunc->localGetResponse ||
        !controlFunc->remoteGetResponse)
    {
        return -1;
    }

    glControlFunc = controlFunc;
    if (sem_init(&addSchedulerSem, 0, 1)) {
        perror("Can't create semaphore\n");
    }
    numSchedulers = 0;
    for (i = 0; i < MAX_SCHEDULERS; i++){
        schedulerState[i].schedulerId = -1;
    }

#ifndef __PC__
    xLinkDeviceHandle_t temp = {0};
    temp.protocol = X_LINK_ANY_PROTOCOL;
    return dispatcherStart(&temp); //myriad has one
#else
    return 0;
#endif
}

int dispatcherClean(void* xLinkFD)
{
    xLinkSchedulerState_t* curr = findCorrespondingScheduler(xLinkFD);
    ASSERT_X_LINK(curr != NULL);

    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&reset_mutex), 1);
    if(!isAvailableScheduler(curr)) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&reset_mutex));
        return 1;
    }
    mvLog(MVLOG_INFO, "Start Clean Dispatcher...");
    closeDeviceFdAndResetScheduler(curr);
    mvLog(MVLOG_INFO, "Clean Dispatcher Successfully...");
    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&reset_mutex));
    return 0;
}

/* end of file */

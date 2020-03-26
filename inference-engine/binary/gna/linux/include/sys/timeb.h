#ifndef __SYS_TIMEB_H
#define __SYS_TIMEB_H

#include <time.h>

struct timeb {
    time_t time;
    unsigned short millitm;
    short timezone;
    short dstflag;
};

static inline int ftime(struct timeb *tp) {
    const unsigned int ONE_MS_IN_NS = 100000;
    struct timespec ts;

    int err = clock_gettime(CLOCK_REALTIME, &ts);
    if (err)
        return -1;

    tp->time = ts.tv_sec;
    tp->millitm = ts.tv_nsec / ONE_MS_IN_NS;
    return 0;
}

#endif /* __SYS_TIMEB_H */

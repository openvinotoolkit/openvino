# -*- coding: utf-8 -*-

# Copyright (C) 2014 Yahoo! Inc. All Rights Reserved.
# Copyright 2011 OpenStack Foundation.
#
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import collections
import contextlib
import functools
import threading
from typing import Optional

from fasteners import _utils


class ReaderWriterLock(object):
    """An inter-thread readers writer lock."""

    WRITER = 'w'  #: Writer owner type/string constant.
    READER = 'r'  #: Reader owner type/string constant.

    def __init__(self,
                 condition_cls=threading.Condition,
                 current_thread_functor=threading.current_thread):
        """
        Args:
            condition_cls:
                Optional custom `Condition` primitive used for synchronization.
            current_thread_functor:
                Optional function that returns the identity of the thread in case
                threads are not properly identified by threading.current_thread
        """
        self._writer = None
        self._writer_entries = 0
        self._pending_writers = collections.deque()
        self._readers = {}
        self._cond = condition_cls()
        self._current_thread = current_thread_functor

    @property
    def has_pending_writers(self) -> bool:
        """Check if there pending writers

        Returns:
            Whether there are pending writers.
        """
        return bool(self._pending_writers)

    def is_writer(self, check_pending: bool = True) -> bool:
        """Check if caller is a writer (optionally pending writer).

        Args:
            check_pending:
                Whether to check for pending writer status.

        Returns:
            Whether the caller is the active (or optionally pending) writer.
        """
        me = self._current_thread()
        if self._writer == me:
            return True
        if check_pending:
            return me in self._pending_writers
        else:
            return False

    def is_reader(self) -> bool:
        """Check if caller is a reader.

        Returns:
            Whether the caller is an active reader.
        """
        me = self._current_thread()
        return me in self._readers

    @property
    def owner(self) -> Optional[str]:
        """Caller ownership (if any) of the lock

        Returns:
            `'w'` if caller is a writer, `'r'` if caller is a reader, None otherwise.
        """
        """Returns whether the lock is locked by a writer or reader."""
        if self._writer is not None:
            return self.WRITER
        if self._readers:
            return self.READER
        return None

    def acquire_read_lock(self):
        """Acquire a read lock.

        Will wait until no active or pending writers.

        Raises:
            RuntimeError: if a pending writer tries to acquire a read lock.
        """
        me = self._current_thread()
        self._acquire_read_lock(me)

    def release_read_lock(self):
        """Release a read lock.

        Raises:
            RuntimeError: if the current thread does not own a read lock.
        """
        me = self._current_thread()
        self._release_read_lock(me)

    def _acquire_read_lock(self, me):
        if me in self._pending_writers:
            raise RuntimeError("Writer %s can not acquire a read lock"
                               " while waiting for the write lock"
                               % me)
        with self._cond:
            while True:
                # No active writer, or we are the writer;
                # Also no pending writers;
                # we are good to become a reader.
                if self._writer is None or self._writer == me:
                    if me in self._readers:
                        # ok to get a lock if current thread already has one
                        self._readers[me] = self._readers[me] + 1
                        break
                    elif (self._writer == me) or not self.has_pending_writers:
                        self._readers[me] = 1
                        break
                # An active or pending writer; guess we have to wait.
                self._cond.wait()

    def _release_read_lock(self, me, raise_on_not_owned=True):
        # I am no longer a reader, remove *one* occurrence of myself.
        # If the current thread acquired two read locks, then it will
        # still have to remove that other read lock; this allows for
        # basic reentrancy to be possible.
        with self._cond:
            try:
                me_instances = self._readers[me]
                if me_instances > 1:
                    self._readers[me] = me_instances - 1
                else:
                    self._readers.pop(me)
            except KeyError:
                if raise_on_not_owned:
                    raise RuntimeError(f"Thread {me} does not own a read lock")
            self._cond.notify_all()

    @contextlib.contextmanager
    def read_lock(self):
        """Context manager that grants a read lock.

        Will wait until no active or pending writers.

        Raises:
            RuntimeError: if a pending writer tries to acquire a read lock.
        """
        me = self._current_thread()
        self._acquire_read_lock(me)
        try:
            yield self
        finally:
            self._release_read_lock(me, raise_on_not_owned=False)

    def _acquire_write_lock(self, me):
        if self.is_reader():
            raise RuntimeError("Reader %s to writer privilege"
                               " escalation not allowed" % me)

        with self._cond:
            self._pending_writers.append(me)
            while True:
                # No readers, and no active writer, am I next??
                if len(self._readers) == 0 and self._writer is None:
                    if self._pending_writers[0] == me:
                        self._writer = self._pending_writers.popleft()
                        self._writer_entries = 1
                        break
                self._cond.wait()

    def _release_write_lock(self, me, raise_on_not_owned=True):
        with self._cond:
            self._writer = None
            self._writer_entries = 0
            self._cond.notify_all()

    def acquire_write_lock(self):
        """Acquire a write lock.

        Will wait until no active readers. Blocks readers after acquiring.

        Guaranteed for locks to be processed in fair order (FIFO).

        Raises:
            RuntimeError: if an active reader attempts to acquire a lock.
        """
        me = self._current_thread()
        if self._writer == me:
            self._writer_entries += 1
        else:
            self._acquire_write_lock(me)

    def release_write_lock(self):
        """Release a write lock.

        Raises:
            RuntimeError: if the current thread does not own a write lock.
        """
        me = self._current_thread()
        if self._writer == me:
            self._writer_entries -= 1
            if self._writer_entries == 0:
                self._release_write_lock(me)
        else:
            raise RuntimeError(f"Thread {me} does not own a write lock")

    @contextlib.contextmanager
    def write_lock(self):
        """Context manager that grants a write lock.

        Will wait until no active readers. Blocks readers after acquiring.

        Guaranteed for locks to be processed in fair order (FIFO).

        Raises:
            RuntimeError: if an active reader attempts to acquire a lock.
        """
        me = self._current_thread()
        if self.is_writer(check_pending=False):
            self._writer_entries += 1
            try:
                yield self
            finally:
                self._writer_entries -= 1
        else:
            self._acquire_write_lock(me)
            try:
                yield self
            finally:
                self._release_write_lock(me)


def locked(*args, **kwargs):
    """A locking **method** decorator.

    It will look for a provided attribute (typically a lock or a list
    of locks) on the first argument of the function decorated (typically this
    is the 'self' object) and before executing the decorated function it
    activates the given lock or list of locks as a context manager,
    automatically releasing that lock on exit.

    NOTE(harlowja): if no attribute name is provided then by default the
    attribute named '_lock' is looked for (this attribute is expected to be
    the lock/list of locks object/s) in the instance object this decorator
    is attached to.

    NOTE(harlowja): a custom logger (which will be used if lock release
    failures happen) can be provided by passing a logger instance for keyword
    argument ``logger``.

    NOTE(paulius): This function is DEPRECATED and will be kept until the end
    of time. It is potentially used by oslo, but too specific to be recommended
    for other projects
    """

    def decorator(f):
        attr_name = kwargs.get('lock', '_lock')
        logger = kwargs.get('logger')

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, (tuple, list)):
                with _utils.LockStack(logger=logger) as stack:
                    for i, lock in enumerate(attr_value):
                        if not stack.acquire_lock(lock):
                            raise threading.ThreadError("Unable to acquire"
                                                        " lock %s" % (i + 1))
                    return f(self, *args, **kwargs)
            else:
                lock = attr_value
                with lock:
                    return f(self, *args, **kwargs)

        return wrapper

    # This is needed to handle when the decorator has args or the decorator
    # doesn't have args, python is rather weird here...
    if kwargs or not args:
        return decorator
    else:
        if len(args) == 1:
            return decorator(args[0])
        else:
            return decorator


def read_locked(*args, **kwargs):
    """Acquires & releases a read lock around call into decorated method.

    NOTE(harlowja): if no attribute name is provided then by default the
    attribute named '_lock' is looked for (this attribute is expected to be
    a :py:class:`.ReaderWriterLock`) in the instance object this decorator
    is attached to.

    NOTE(paulius): This function is DEPRECATED and will be kept until the end
    of time. It is potentially used by oslo, but too specific to be recommended
    for other projects
    """

    def decorator(f):
        attr_name = kwargs.get('lock', '_lock')

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            rw_lock = getattr(self, attr_name)
            with rw_lock.read_lock():
                return f(self, *args, **kwargs)

        return wrapper

    # This is needed to handle when the decorator has args or the decorator
    # doesn't have args, python is rather weird here...
    if kwargs or not args:
        return decorator
    else:
        if len(args) == 1:
            return decorator(args[0])
        else:
            return decorator


def write_locked(*args, **kwargs):
    """Acquires & releases a write lock around call into decorated method.

    NOTE(harlowja): if no attribute name is provided then by default the
    attribute named '_lock' is looked for (this attribute is expected to be
    a :py:class:`.ReaderWriterLock` object) in the instance object this
    decorator is attached to.

    NOTE(paulius): This function is DEPRECATED and will be kept until the end
    of time. It is potentially used by oslo, but too specific to be recommended
    for other projects
    """

    def decorator(f):
        attr_name = kwargs.get('lock', '_lock')

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            rw_lock = getattr(self, attr_name)
            with rw_lock.write_lock():
                return f(self, *args, **kwargs)

        return wrapper

    # This is needed to handle when the decorator has args or the decorator
    # doesn't have args, python is rather weird here...
    if kwargs or not args:
        return decorator
    else:
        if len(args) == 1:
            return decorator(args[0])
        else:
            return decorator


@contextlib.contextmanager
def try_lock(lock: threading.Lock) -> bool:
    """Context manager that attempts to acquire a lock without a timeout, and
    releases it on exit (if acquired).

    Args:
        lock:
            A lock to try to acquire.

    Returns:
        Whether the lock was acquired.

    # NOTE(harlowja): the keyword argument for 'blocking' does not work
    # in py2.x and only is fixed in py3.x (this adjustment is documented
    # and/or debated in http://bugs.python.org/issue10789); so we'll just
    # stick to the format that works in both (oddly the keyword argument
    # works in py2.x but only with reentrant locks).

    NOTE(paulius): This function is DEPRECATED and will be kept until the end
    of time. It is potentially used by oslo, but too specific to be recommended
    for other projects
    """
    was_locked = lock.acquire(False)
    try:
        yield was_locked
    finally:
        if was_locked:
            lock.release()

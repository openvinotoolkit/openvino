from abc import ABC
from abc import abstractmethod
import errno
import os


class _InterProcessReaderWriterLockMechanism(ABC):

    @staticmethod
    @abstractmethod
    def trylock(lockfile, exclusive):
        ...

    @staticmethod
    @abstractmethod
    def unlock(lockfile):
        ...

    @staticmethod
    @abstractmethod
    def get_handle(path):
        ...

    @staticmethod
    @abstractmethod
    def close_handle(lockfile):
        ...


class _InterProcessMechanism(ABC):
    @staticmethod
    @abstractmethod
    def trylock(lockfile):
        ...

    @staticmethod
    @abstractmethod
    def unlock(lockfile):
        ...


class _WindowsInterProcessMechanism(_InterProcessMechanism):
    """Interprocess lock implementation that works on windows systems."""

    @staticmethod
    def trylock(lockfile):
        fileno = lockfile.fileno()
        msvcrt.locking(fileno, msvcrt.LK_NBLCK, 1)

    @staticmethod
    def unlock(lockfile):
        fileno = lockfile.fileno()
        msvcrt.locking(fileno, msvcrt.LK_UNLCK, 1)


class _FcntlInterProcessMechanism(_InterProcessMechanism):
    """Interprocess lock implementation that works on posix systems."""

    @staticmethod
    def trylock(lockfile):
        fcntl.lockf(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def unlock(lockfile):
        fcntl.lockf(lockfile, fcntl.LOCK_UN)


class _WindowsInterProcessReaderWriterLockMechanism(_InterProcessReaderWriterLockMechanism):
    """Interprocess readers writer lock implementation that works on windows
    systems."""

    @staticmethod
    def trylock(lockfile, exclusive):

        if exclusive:
            flags = win32con.LOCKFILE_FAIL_IMMEDIATELY | win32con.LOCKFILE_EXCLUSIVE_LOCK
        else:
            flags = win32con.LOCKFILE_FAIL_IMMEDIATELY

        handle = msvcrt.get_osfhandle(lockfile.fileno())
        ok = win32file.LockFileEx(handle, flags, 0, 1, 0, win32file.pointer(pywintypes.OVERLAPPED()))
        if ok:
            return True
        else:
            last_error = win32file.GetLastError()
            if last_error == win32file.ERROR_LOCK_VIOLATION:
                return False
            else:
                raise OSError(last_error)

    @staticmethod
    def unlock(lockfile):
        handle = msvcrt.get_osfhandle(lockfile.fileno())
        ok = win32file.UnlockFileEx(handle, 0, 1, 0, win32file.pointer(pywintypes.OVERLAPPED()))
        if not ok:
            raise OSError(win32file.GetLastError())

    @staticmethod
    def get_handle(path):
        return open(path, 'a+')

    @staticmethod
    def close_handle(lockfile):
        lockfile.close()


class _FcntlInterProcessReaderWriterLockMechanism(_InterProcessReaderWriterLockMechanism):
    """Interprocess readers writer lock implementation that works on posix
    systems."""

    @staticmethod
    def trylock(lockfile, exclusive):

        if exclusive:
            flags = fcntl.LOCK_EX | fcntl.LOCK_NB
        else:
            flags = fcntl.LOCK_SH | fcntl.LOCK_NB

        try:
            fcntl.lockf(lockfile, flags)
            return True
        except (IOError, OSError) as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                return False
            else:
                raise e

    @staticmethod
    def unlock(lockfile):
        fcntl.lockf(lockfile, fcntl.LOCK_UN)

    @staticmethod
    def get_handle(path):
        return open(path, 'a+')

    @staticmethod
    def close_handle(lockfile):
        lockfile.close()


if os.name == 'nt':
    import msvcrt
    import fasteners.pywin32.pywintypes as pywintypes
    import fasteners.pywin32.win32con as win32con
    import fasteners.pywin32.win32file as win32file

    _interprocess_reader_writer_mechanism = _WindowsInterProcessReaderWriterLockMechanism()
    _interprocess_mechanism = _WindowsInterProcessMechanism()

else:
    import fcntl

    _interprocess_reader_writer_mechanism = _FcntlInterProcessReaderWriterLockMechanism()
    _interprocess_mechanism = _FcntlInterProcessMechanism()

import threading
import signal

class SigintHandler(object):
    class ProcessWasInterrupted(Exception): pass
    sigint_returncodes = {-signal.SIGINT,  # Unix
                            -1073741510,     # Windows
                            }

    def __init__(self):
        self.__lock = threading.Lock()
        self.__processes = set()
        self.__got_sigint = False
        signal.signal(signal.SIGINT, lambda signal_num, frame: self.interrupt())

    def __on_sigint(self):
        self.__got_sigint = True
        while self.__processes:
            try:
                self.__processes.pop().terminate()
            except OSError:
                pass

    def interrupt(self):
        with self.__lock:
            self.__on_sigint()

    def got_sigint(self):
        with self.__lock:
            return self.__got_sigint

    def wait(self, p):
        with self.__lock:
            if self.__got_sigint:
                p.terminate()
            self.__processes.add(p)
        code = p.wait()
        with self.__lock:
            self.__processes.discard(p)
            if code in self.sigint_returncodes:
                self.__on_sigint()
            if self.__got_sigint:
                raise self.ProcessWasInterrupted
        return code
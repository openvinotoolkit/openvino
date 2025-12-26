import fnmatch
import os
import sys
import time
from threading import Lock

import colorama
from colorama import Fore, Style

from conan.errors import ConanException

LEVEL_QUIET = 80  # -q
LEVEL_ERROR = 70  # Errors
LEVEL_WARNING = 60  # Warnings
LEVEL_NOTICE = 50  # Important messages to attract user attention.
LEVEL_STATUS = 40  # Default - The main interesting messages that users might be interested in.
LEVEL_VERBOSE = 30  # -v  Detailed informational messages.
LEVEL_DEBUG = 20  # -vv Closely related to internal implementation details
LEVEL_TRACE = 10  # -vvv Fine-grained messages with very low-level implementation details


class Color:
    """ Wrapper around colorama colors that are undefined in importing
    """
    RED = Fore.RED  # @UndefinedVariable
    WHITE = Fore.WHITE  # @UndefinedVariable
    CYAN = Fore.CYAN  # @UndefinedVariable
    GREEN = Fore.GREEN  # @UndefinedVariable
    MAGENTA = Fore.MAGENTA  # @UndefinedVariable
    BLUE = Fore.BLUE  # @UndefinedVariable
    YELLOW = Fore.YELLOW  # @UndefinedVariable
    BLACK = Fore.BLACK  # @UndefinedVariable

    BRIGHT_RED = Style.BRIGHT + Fore.RED  # @UndefinedVariable
    BRIGHT_BLUE = Style.BRIGHT + Fore.BLUE  # @UndefinedVariable
    BRIGHT_YELLOW = Style.BRIGHT + Fore.YELLOW  # @UndefinedVariable
    BRIGHT_GREEN = Style.BRIGHT + Fore.GREEN  # @UndefinedVariable
    BRIGHT_CYAN = Style.BRIGHT + Fore.CYAN  # @UndefinedVariable
    BRIGHT_WHITE = Style.BRIGHT + Fore.WHITE  # @UndefinedVariable
    BRIGHT_MAGENTA = Style.BRIGHT + Fore.MAGENTA  # @UndefinedVariable


if os.environ.get("CONAN_COLOR_DARK"):
    Color.WHITE = Fore.BLACK
    Color.CYAN = Fore.BLUE
    Color.YELLOW = Fore.MAGENTA
    Color.BRIGHT_WHITE = Fore.BLACK
    Color.BRIGHT_CYAN = Fore.BLUE
    Color.BRIGHT_YELLOW = Fore.MAGENTA
    Color.BRIGHT_GREEN = Fore.GREEN


def init_colorama(stream):
    import colorama
    if _color_enabled(stream):
        if os.getenv("CLICOLOR_FORCE", "0") != "0":
            # Otherwise it is not really forced if colorama doesn't feel it
            colorama.init(strip=False, convert=False)
        else:
            colorama.init()


def _color_enabled(stream):
    """
    NO_COLOR: No colors

    https://no-color.org/

    Command-line software which adds ANSI color to its output by default should check for the
    presence of a NO_COLOR environment variable that, when present (**regardless of its value**),
    prevents the addition of ANSI color.

    CLICOLOR_FORCE: Force color

    https://bixense.com/clicolors/
    """

    if os.getenv("CLICOLOR_FORCE", "0") != "0":
        # CLICOLOR_FORCE != 0, ANSI colors should be enabled no matter what.
        return True

    if os.getenv("NO_COLOR") is not None:
        return False
    return hasattr(stream, "isatty") and stream.isatty()


class ConanOutput:
    """ A singleton class to handle output messages in Conan.

    Recipes should only access this class through the ``self.output`` attribute of the recipe,
    but custom commands or tools can instantiate it directly, where doing so for each message is
    a valid practice.

    It provides methods to write messages at different levels of verbosity, such as debug, info,
    warning, and error. The output level can be controlled by the user through command-line options
    or environment variables.

    The output methods return the instance itself, so different methods can be chained together.
    """
    # Singleton
    _conan_output_level = LEVEL_STATUS
    _silent_warn_tags = []
    _warnings_as_errors = []
    lock = Lock()

    def __init__(self, scope: str = ""):
        """ Initialize the ConanOutput instance.

        :parameter scope: A string that represents the scope of the output. This is usually the
            reference of the recipe being executed, like ``pkg/1.0@user/channel`` and is prefixed
            to the output messages. If not provided, it defaults to an empty string.
        """
        self.stream = sys.stderr
        self._scope = scope
        # FIXME:  This is needed because in testing we are redirecting the sys.stderr to a buffer
        #         stream to capture it, so colorama is not there to strip the color bytes
        self._color = _color_enabled(self.stream)

    @classmethod
    def define_silence_warnings(cls, warnings):
        cls._silent_warn_tags = warnings

    @classmethod
    def set_warnings_as_errors(cls, value):
        cls._warnings_as_errors = value

    @classmethod
    def get_output_level(cls):
        return cls._conan_output_level

    @classmethod
    def set_output_level(cls, level):
        cls._conan_output_level = level

    @classmethod
    def valid_log_levels(cls):
        return {"quiet": LEVEL_QUIET,  # -vquiet 80
            "error": LEVEL_ERROR,  # -verror 70
            "warning": LEVEL_WARNING,  # -vwaring 60
            "notice": LEVEL_NOTICE,  # -vnotice 50
            "status": LEVEL_STATUS,  # -vstatus 40
            None: LEVEL_VERBOSE,  # -v 30
            "verbose": LEVEL_VERBOSE,  # -vverbose 30
            "debug": LEVEL_DEBUG,  # -vdebug 20
            "v": LEVEL_DEBUG,  # -vv 20
            "trace": LEVEL_TRACE,  # -vtrace 10
            "vv": LEVEL_TRACE  # -vvv 10
        }

    @classmethod
    def define_log_level(cls, v):
        env_level = os.getenv("CONAN_LOG_LEVEL")
        v = env_level or v
        levels = cls.valid_log_levels()
        try:
            level = levels[v]
        except KeyError:
            msg = " defined in CONAN_LOG_LEVEL environment variable" if env_level else ""
            vals = "quiet, error, warning, notice, status, verbose, debug(v), trace(vv)"
            raise ConanException(f"Invalid argument '-v{v}'{msg}.\nAllowed values: {vals}")
        else:
            cls.set_output_level(level)

    @classmethod
    def level_allowed(cls, level):
        return cls._conan_output_level <= level

    @property
    def color(self):
        return self._color

    @property
    def scope(self):
        return self._scope

    @scope.setter
    def scope(self, out_scope):
        self._scope = out_scope

    @property
    def is_terminal(self):
        return hasattr(self.stream, "isatty") and self.stream.isatty()

    def writeln(self, data, fg=None, bg=None):
        return self.write(data, fg, bg, newline=True)

    def write(self, data, fg=None, bg=None, newline=False):
        if self._conan_output_level > LEVEL_NOTICE:
            return self
        if self._color and (fg or bg):
            data = "%s%s%s%s" % (fg or '', bg or '', data, Style.RESET_ALL)

        if newline:
            data = "%s\n" % data

        with self.lock:
            self.stream.write(data)
            self.stream.flush()

        return self

    def box(self, msg: str):
        """ Draw a box around the message, useful for important messages"""
        color = Color.BRIGHT_GREEN
        self.writeln("\n**************************************************", fg=color)
        self.writeln(f'*{msg: ^48}*', fg=color)
        self.writeln(f"**************************************************\n", fg=color)
        return self

    def login_msg(self, msg, newline=False):
        # unconditional to the error level, this has to show always
        self._write_message(msg, newline=newline)
        return self

    def _write_message(self, msg, fg=None, bg=None, newline=True):
        if isinstance(msg, dict):
            # For traces we can receive a dict already, we try to transform then into more natural
            # text
            msg = ", ".join([f"{k}: {v}" for k, v in msg.items()])
            msg = "=> {}".format(msg)
            # msg = json.dumps(msg, sort_keys=True, default=json_encoder)

        if self._scope:
            if self._color:
                ret = f"{fg or ''}{bg or ''}{self._scope}: {msg}{Style.RESET_ALL}"
            else:
                ret = f"{self._scope}: {msg}"
        else:
            if self._color:
                ret = f"{fg or ''}{bg or ''}{msg}{Style.RESET_ALL}"
            else:
                ret = msg

        if newline:
            ret = f"{ret}\n"

        with self.lock:
            self.stream.write(ret)
            self.stream.flush()

    def trace(self, msg: str):
        """ This is the most extreme level of detail.

        Trace messages log every little step the system takes, including function entries and exits,
        variable changes, and other very specific events.

        This message won't be printed unless the user has set the log level to trace
        (e.g., using the ``-vvv`` option in the command line).

        It’s used when full visibility of everything happening in the system is required,
        but should be used carefully due to the large amount of information it can generate."""
        if self._conan_output_level <= LEVEL_TRACE:
            self._write_message(msg, fg=Color.BLUE)
        return self

    def debug(self, msg: str, fg: str = Color.MAGENTA, bg: str = None):
        """ With a high level of detail, it is mainly used for debugging code.

        This message won't be printed unless the user has set the log level to debug
        (e.g., using the ``-vv`` option in the command line).

        These messages provide useful information for developers, such as variable values
        or execution flow details, to trace errors or analyze the program's behavior."""
        if self._conan_output_level <= LEVEL_DEBUG:
            self._write_message(msg, fg=fg, bg=bg)
        return self

    def verbose(self, msg: str, fg: str = None, bg: str = None):
        """ Displays additional and detailed information that, while not critical,
        can be useful for better understanding how the system is working.

        This message won't be printed unless the user has set the log level to verbose
        (e.g., using the ``-v`` option in the command line).

        It’s appropriate for gaining more context without overloading the logs with
        excessive detail. Useful when more clarity is needed than a simple info."""
        if self._conan_output_level <= LEVEL_VERBOSE:
            self._write_message(msg, fg=fg, bg=bg)
        return self

    def status(self, msg: str, fg: str = None, bg: str = None, newline: bool = True):
        """ Provides general information about the system or ongoing operations.

        Info messages are basic and used to inform about common events,
        like the start or completion of processes, without implying specific problems or achievements."""
        if self._conan_output_level <= LEVEL_STATUS:
            self._write_message(msg, fg=fg, bg=bg, newline=newline)
        return self

    info = status

    def title(self, msg: str):
        """ Draws a title around the message, useful for important messages"""
        if self._conan_output_level <= LEVEL_NOTICE:
            self._write_message("\n======== {} ========".format(msg),
                                fg=Color.BRIGHT_MAGENTA)
        return self

    def subtitle(self, msg: str):
        """ Draws a subtitle around the message, useful for important messages"""
        if self._conan_output_level <= LEVEL_NOTICE:
            self._write_message("\n-------- {} --------".format(msg),
                                fg=Color.BRIGHT_MAGENTA)
        return self

    def highlight(self, msg: str):
        """ Marks or emphasizes important events or processes that need to stand out but don’t necessarily
        indicate success or error.

        These messages draw attention to key points that may be relevant for the user or administrator."""
        if self._conan_output_level <= LEVEL_NOTICE:
            self._write_message(msg, fg=Color.BRIGHT_MAGENTA)
        return self

    def success(self, msg: str):
        """ Shows that an operation has been completed successfully.

        This type of message is useful to confirm that key processes or tasks have finished correctly,
        which is essential for good application monitoring."""
        if self._conan_output_level <= LEVEL_NOTICE:
            self._write_message(msg, fg=Color.BRIGHT_GREEN)
        return self

    @staticmethod
    def _warn_tag_matches(warn_tag, patterns):
        lookup_tag = warn_tag or "unknown"
        return any(fnmatch.fnmatch(lookup_tag, pattern) for pattern in patterns)

    def warning(self, msg: str, warn_tag: str = None):
        """ Highlights a potential issue that, while not stopping the system,
        could cause problems in the future or under certain conditions.

        Warnings signal abnormal situations that should be
        reviewed but don’t necessarily cause an immediate halt in operations.
        Notice that if the tag matches the pattern in the ``core:warnings_as_errors`` configuration,
        and is not skipped, this will be upgraded to an error, and raise an exception
        when the output is printed, so that the error does not pass unnoticed."""
        _treat_as_error = self._warn_tag_matches(warn_tag, self._warnings_as_errors)
        if (self._conan_output_level <= LEVEL_WARNING or
                (_treat_as_error and self._conan_output_level <= LEVEL_ERROR)):
            if self._warn_tag_matches(warn_tag, self._silent_warn_tags):
                return self
            warn_tag_msg = "" if warn_tag is None else f"{warn_tag}: "
            output = f"{warn_tag_msg}{msg}"

            if _treat_as_error:
                self.error(output)
            else:
                self._write_message(f"WARN: {output}", Color.YELLOW)
        return self

    def error(self, msg: str, error_type: str = None):
        """ Indicates that a serious issue has occurred that prevents the system
        or application from continuing to function correctly.

        Typically, this represents a failure in the normal flow of execution,
        such as a service crash or a critical exception.
        Notice that if the user has set the ``core:warnings_as_errors`` configuration,
        this will raise an exception when the output is printed,
        so that the error does not pass unnoticed."""
        if self._warnings_as_errors and error_type != "exception":
            raise ConanException(msg)
        if self._conan_output_level <= LEVEL_ERROR:
            self._write_message("ERROR: {}".format(msg), Color.RED)
        return self

    def flush(self):
        self.stream.flush()


def cli_out_write(data, fg=None, bg=None, endline="\n", indentation=0):
    """
    Output to be used by formatters to dump information to stdout
    """
    if (fg or bg) and _color_enabled(sys.stdout):  # need color
        data = f"{' ' * indentation}{fg or ''}{bg or ''}{data}{Style.RESET_ALL}{endline}"
        sys.stdout.write(data)
    else:
        data = f"{' ' * indentation}{data}{endline}"
        if sys.stdout.isatty():
            # https://github.com/conan-io/conan/issues/17245 avoid colorama crash and overhead
            # skip deinit/reinit if stdout is not a TTY to preserve redirected output to file
            colorama.deinit()
            sys.stdout.write(data)
            colorama.reinit()
        else:
            sys.stdout.write(data)


class TimedOutput:
    def __init__(self, interval, out=None, msg_format=None):
        self._interval = interval
        self._msg_format = msg_format
        self._t = time.time()
        self._out = out or ConanOutput()

    def info(self, msg, *args, **kwargs):
        t = time.time()
        if t - self._t > self._interval:
            self._t = t
            if self._msg_format:
                msg = self._msg_format(msg, *args, **kwargs)
            self._out.info(msg)

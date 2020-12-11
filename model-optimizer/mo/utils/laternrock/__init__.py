"""
Python wrapper for LanternRock
-------------------------------

Primary documention for Lanternrock enabling can be found here:

    https://wiki.ith.intel.com/pages/viewpage.action?pageId=823269000

Function Reference
------------------
.. autoclass:: LanternRock
    :members:

"""
from ctypes import *
import platform
import os
import errno
import json
import sys
import time
import math
if sys.platform=='win32':
    from ctypes import wintypes #will not work on linux
else:
    wintypes = None
import uuid
import datetime
from functools import wraps
import re
import subprocess
# for now this feature is off until regression testing
# can support with and without this package being installed
WinError = None
# should get fixed to correct version by build process
__version__ = "3.0.73.0"
try:
    from winerrorenum import WinError
except ImportError:
    WinError = None

# for compatibility with python 2.7, there is not FileNotFoundError
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

##############################
# cross version compatibility
if sys.version_info[0] == 2:
    stringTypeCheck = (str, unicode)
    integerTypeCheck = (int, long)
elif sys.version_info[0] == 3:
    stringTypeCheck = str
    integerTypeCheck = int
else:
    raise Exception("Unsupported Python version %s" % sys.version)

##############################
# extending error codes
if WinError is not None:
    class LanternRockErrorEnum(WinError):
        _customCodes = {
            'IAS_NO_OPTIN'         :     262657  , # error message: user not opted in.
            'IAS_OPTOUT'           :     262658  , # error message: user opted out.
            'E_DISABLED_BY_POLICY' : -2147220982 , # error message: telemetry disabled by policy.
            'E_DISABLED'           : -2147220981 , # error message: telemetry disabled.
        }
else:
    class LanternRockErrorEnum(object):
        @classmethod
        def lookup(self, hresult):
            return hresult
        @classmethod
        def hresult_from_win(self, hresult):
            return hresult

##############################
# hardcode error code if winerrir is not importing
try:
    import winerror
    _CODE_E_UNEXPECTED    = winerror.E_UNEXPECTED
    _CODE_E_INVALIDARG    = winerror.E_INVALIDARG
    _CODE_ERROR_NOT_FOUND = winerror.ERROR_NOT_FOUND
except:
    _CODE_E_UNEXPECTED    = -2147418113
    _CODE_E_INVALIDARG    = -2147024809
    _CODE_ERROR_NOT_FOUND = 1168

##############################
# Data formatting functions for LanternRock API calls
# moved to global space, instead of being class/instance/static functions for simplicity
def _format_datadict(datadict=None):
    """Converts a python data dictionary into a ctypes tuple with a pointer to each of the three parameter lists
    of the base API, i.e. keys, values and segs.

    Args:
        datadict (dict, optional): A python data dictionary to format to fit to LanternRock API.
                                   If provided, keys must be of string type
                                   If not provided, empty keys, values, segs tuple is created and returned.

    Returns:
        tuple: (keys, values, segs)

    """
    if datadict is not None:
        if type(datadict) is dict:
            s_data = {}
            for k, v in datadict.items():
                if type(v) is int and type(k) is str:
                    if len(k) > 0:
                        key = k if k[0] == '#' else '#%s' % k
                        s_data[key] = str(v)
                elif type(k) is not str:
                    raise LanternRockArgumentError
                else:
                    s_data[k] = str(v)
            segs = c_uint32(len(s_data.keys()))
            keys = (c_wchar_p * len(s_data.keys()))(*s_data.keys())
            values = (c_wchar_p * len(s_data.keys()))(*s_data.values())
        else:
            raise LanternRockArgumentError
    else:
        keys = (c_wchar_p * 1)()
        values = (c_wchar_p * 1)()
        segs = c_uint32(0)
    return keys, values, segs

def _format_options(options=None):
    """Converts an option string or data dictionary into a ctype structure for use in LanternRock API.

    Args:
        options (str or dict, optional): Datastructure to be converted to LanternRock options unicode buffer
                               If json-string then unicode representation of the string is returned.
                               If dict then dictionary is converted to json representation and returned
                               as a unicode buffer.
                               If None or not provided, then an empty unicode buffer is returned.
                               Any other type will throw a LanternRockArgumentError.
    Returns:
        ctypes unicode buffer: LanternRock options unicode buffer

    Example:
         from lanternrock import *
         lr = LanternRock()
         options = {'option1':value,'option2':value}
         lr.Initialize(APP_NAME, APP_VERSION, TELEMETRY_ID, options=_format_options(options))
    """

    if type(options) is str:
        options = create_unicode_buffer(options)
    elif type(options) is dict:
        options = create_unicode_buffer(json.dumps(options))
    elif options is None:
        options = c_wchar_p()
    else:
        raise LanternRockArgumentError
    return options

def _format_init_params_datadict(app_name, app_version, telemetryid, options, store_folder=None, datadict=None):
    """Formats initialization parameters for passing to the API on startup.
       See initialize for discussion of what these parameters are for.
       Returns the ctypes in a tuple for consumption in Initialize API.

    Args:
        app_name (str): Name of the app. Will be converted to unicode buffer.
                        Any other type will throw a LanternRockArgumentError.
        app_version (str): Version of the app. Will be converted to unicode buffer.
                           Any other type will throw a LanternRockArgumentError.
        telemetryid (str): Telemetry ID string. Will be converted to unicode buffer.
                           Any other type will throw a LanternRockArgumentError.
        options (str or dict): Configuration options. Will be converted to unicode buffer
                               (see _format_options) function.
        store_folder (str, optional): Location of LanternRock temp files. If provided will be converted to
                                      unicode buffer. If not provided, empty unicode buffer is returned.
                                      Any other type will throw a LanternRockArgumentError.
        datadict (dict, optional): A python data dictionary to format to fit to LanternRock API.
                                   If not provided, empty keys, values, segs tuple is created and returned.

    Returns:
        tuple: (app_name, app_version, telemetryid, options, store_folder, keys, values, segs)
    """
    if not isinstance(app_name, stringTypeCheck):
            raise LanternRockArgumentError
    if not isinstance(app_version, stringTypeCheck):
            raise LanternRockArgumentError
    if not isinstance(telemetryid, stringTypeCheck):
            raise LanternRockArgumentError

    app_name = create_unicode_buffer(app_name)
    app_version = create_unicode_buffer(app_version)
    telemetryid = create_unicode_buffer(telemetryid)
    options = _format_options(options)
    if store_folder is not None:
        if not isinstance(store_folder, stringTypeCheck):
            raise LanternRockArgumentError
        store_folder = create_unicode_buffer(store_folder)
    else:
        store_folder = c_wchar_p()
    keys, values, segs = _format_datadict(datadict)
    return app_name, app_version, telemetryid, options, store_folder, keys, values, segs

def _format_init_params(app_name, app_version, telemetryid, options, store_folder=None):
    """Formats initialization parameters for passing to the API on startup.
       See initialize for discussion of what these parameters are for.
       Returns the ctypes in a tuple for consumption in Initialize API.

    Args:
        app_name (str): Name of the app. Will be converted to unicode buffer.
                        Any other type will throw a LanternRockArgumentError.
        app_version (str): Version of the app. Will be converted to unicode buffer.
                           Any other type will throw a LanternRockArgumentError.
        telemetryid (str): Telemetry ID string. Will be converted to unicode buffer.
                           Any other type will throw a LanternRockArgumentError.
        options (str or dict, optional): Configuration options. Will be converted to unicode buffer
                                        (see _format_options) function.
        store_folder (str, optional): Location of LanternRock temp files. If provided will be converted to
                                      unicode buffer. If not provided, empty unicode buffer is returned.
                                      Any other type will throw a LanternRockArgumentError.

    Returns:
        tuple: (app_name, app_version, telemetryid, options, store_folder)
    """
    #return everything but the last 3 items (datadict)
    return _format_init_params_datadict(app_name, app_version, telemetryid, options, store_folder, None)[:-3]

def _create_unicode_buffer_or_void_p(name):
    """Creates a unicode buffer from name or returns a c_void_p() if name is None

    Args:
        name (str or None): String to be converted to unicode or None.
                            Any other type will throw a LanternRockArgumentError.

    Returns:
        Unicode buffer or c_void_p()
    """
    if not isinstance(name, stringTypeCheck) and name is not None:
            raise LanternRockArgumentError
    return create_unicode_buffer(name) if name is not None else c_void_p()

def _create_void_p(name):
    """Returns a c_void_p() if name is None, otherwise returns name unchanged

    Args:
        name (obj or None): object to be checked.

    Returns:
        Unchanged object or c_void_p()
    """
    return c_void_p() if name is None else name

def _format_count(count):
    """Private formatter for count parameter (int or long or None). 
       Check if count parameter has correct type and converts to c_uint32
       If count is None - returns to c_uint32(1)
       If type is incorrect, LanternRockArgumentError is thrown

    Args:
        count (int): Event count. Any other type will throw a LanternRockArgumentError.

    Returns:
        c_uint32

    """
    if count is not None:
        if not isinstance(count, integerTypeCheck):
            raise LanternRockArgumentError
        count = c_uint32(count)
    else:
        count = c_uint32(1)
    return count
    
def _format_sum(sum):
    """Private formatter for sum parameter (float or None). 
       Check if count parameter has correct type and converts to c_double
       If count is None - returns to c_double(0.0)
       If type is incorrect, LanternRockArgumentError is thrown

    Args:
        sum (float): Event sum. Any other type will throw a LanternRockArgumentError.

    Returns:
        c_double

    """    
    if sum is not None:
        try:
            sum = float(sum)
        except TypeError:
            raise LanternRockArgumentError("Must be float")
        sum = c_double(sum)
    else:
        sum = c_double(0.0)    
    return sum
    
def _format_event_datadict(session, name, count, sum, datadict=None):
    """Private formatter for RecordEvent and RecordEventEx functions

    Args:
        session (session handle or None): Session handle returned by BeginSession or BeginSessionEx.
                                          None if event does not belong to a named session.
        name (str): Event name. Any other type will throw a LanternRockArgumentError.
        count (int): Event count. Any other type will throw a LanternRockArgumentError.
        sum (float): Event sum. Any other type will throw a LanternRockArgumentError.
        datadict (dict, optional): A python data dictionary to format to fit to LanternRock API.
                                   If not provided, empty keys, values, segs tuple is created and returned.
    Returns:
        session, name, count, sum, keys, values, segs

    """
    session = _create_void_p(session)
    if not isinstance(name, stringTypeCheck):
            raise LanternRockArgumentError
    name = create_unicode_buffer(name)

    count = _format_count(count)

    sum = _format_sum(sum)
    
    keys, values, segs = _format_datadict(datadict)
    return session, name, count, sum, keys, values, segs

def _format_event(session, name, count, sum):
    """Private formatter for RecordEvent function

    Args:
        session (session handle or None): Session handle returned by BeginSession or BeginSessionEx.
                                          None if event does not belong to a session.
        name (str): Event name. Any other type will throw a LanternRockArgumentError.
        count (int): Event count. Any other type will throw a LanternRockArgumentError.
        sum (float): Event sum. Any other type will throw a LanternRockArgumentError.

    Returns:
        session, name, count, sum

    """
    return _format_event_datadict(session, name, count, sum, None)[:-3]

def _format_session_datadict(name, start_timestamp, duration_seconds, event_handles, datadict):
    """Private formatter for RecordSession and RecordSessionEx functions

    Args:
        name (str): Session name. Any other type will throw a LanternRockArgumentError.
        start_timestamp(datetime or int): Start time of the session. Integer values must be non-negative.
        duration_seconds(int): Duration of the session. Integer values must be positive.
                               Any other type will throw a LanternRockArgumentError.
        event_handles(list, optional): List of event handles.
        datadict (dict, optional): A python data dictionary to format to fit to LanternRock API.
                                   If not provided, empty keys, values, segs tuple is created and returned.

    Returns:
        name, start_timestamp, duration_seconds, eptr, num_events, keys, values, segs

    """
    name = _create_unicode_buffer_or_void_p(name)

    if type(start_timestamp) is datetime.datetime:
        if (sys.version_info < (3, 0)):
            start_timestamp = c_uint64(int(time.mktime(start_timestamp.timetuple())))
        else:
            start_timestamp = c_uint64(int(start_timestamp.timestamp()))
    elif type(start_timestamp) is int:
        start_timestamp = c_uint64(start_timestamp)
    else:
        start_timestamp = c_uint64(0)

    if duration_seconds is not None:
        if not isinstance(duration_seconds, integerTypeCheck):
            raise LanternRockArgumentError
        duration_seconds = c_uint32(duration_seconds)
    else:
        duration_seconds = c_uint32(0)

    if event_handles is not None and type(event_handles) is list:
        num_events = len(event_handles)
        eptr = (c_void_p * num_events)(*event_handles)
    else:
        num_events = 0
        eptr = c_void_p()
    keys, values, segs = _format_datadict(datadict)
    return name, start_timestamp, duration_seconds, eptr, num_events, keys, values, segs

def _format_session(name, start_timestamp, duration_seconds, event_handles):
    """Private formatter for RecordSession function

    Args:
        name (str): Session name. Any other type will throw a LanternRockArgumentError.
        start_timestamp(datetime or int): Start time of the session. Integer values must be positive.
        duration_seconds(int): Duration of the session. Integer values must be positive.
                               Any other type will throw a LanternRockArgumentError.
        event_handles(list, optional): List of event handles.

    Returns:
        name, start_timestamp, duration_seconds, eptr, num_events

    """
    return _format_session_datadict(name, start_timestamp, duration_seconds, event_handles, None)[:-3]


##############################
# Support classes
class LanternRockError(Exception):
    """Exception class thrown by python wrapper when a LanternRock API call returns on non-zero hresult
    """
    def __init__(self, hresult):
        """Constructor
        """
        if isinstance(hresult, stringTypeCheck):
            self.HRESULT = -1
            self.SRESULT=hresult
            try:
                self.HRESULT=LanternRockErrorEnum.lookup(hresult)
            except:
                pass
        else:    
            self.HRESULT=hresult #: numerical value of hresult returned by LanternRock API call
            self.SRESULT=""
            try:
                self.SRESULT=LanternRockErrorEnum.lookup(hresult)
            except:
                pass
        if self.SRESULT != "":
            Exception.__init__(self, self.SRESULT)
        else:
            Exception.__init__(self, "%d"%self.HRESULT)


class LanternRockInitializationError(Exception):
    """Exception thrown if laternrock is not installed or otherwise will not initialize."""    
    def __init__(self):
        """Constructor"""
        pass

class LanternRockArgumentError(TypeError):
    """Exception thrown if laternrock call has incorrect arguments."""
    def __init__(self):
        """Constructor"""
        pass

class GUID(Structure):
    """Convenience class to convert the lanternrock GUID into the python UUID.
       Inherits from ctypes.Structure
       Declares the following fields:
          time_low (DWORD)
          time_mid (WORD)
          time_hi_version (WORD)
          clock_seq_hi_variant (BYTE)
          clock_seq_low (BYTE)
          node (6*BYTE)
    """
    if wintypes is not None:
        _fields_ = [
            ("time_low", wintypes.DWORD ),
            ("time_mid", wintypes.WORD ),
            ("time_hi_version", wintypes.WORD ),
            ("clock_seq_hi_variant", wintypes.BYTE),
            ("clock_seq_low", wintypes.BYTE),
            ("node", wintypes.BYTE * 6)        
        ]
    else:
        _fields_ = [
            ("time_low", c_uint ),
            ("time_mid", c_ushort ),
            ("time_hi_version", c_ushort ),
            ("clock_seq_hi_variant", c_char),
            ("clock_seq_low", c_char),
            ("node", c_char * 6)
        ]

    def get_uuid(self):
        """Returns a python UUID
        
            Args:
                None

            Returns:
                uuid.UUID instance
        """
        seq_hi = 256 + self.clock_seq_hi_variant if self.clock_seq_hi_variant < 0 else self.clock_seq_hi_variant
        seq_lo = 256 + self.clock_seq_low if self.clock_seq_low < 0 else self.clock_seq_low
        return uuid.UUID(fields=(self.time_low, \
                                 self.time_mid, \
                                 self.time_hi_version, \
                                 seq_hi, \
                                 seq_lo, \
                                 int.from_bytes(bytes(self.node),\
                                                byteorder='big',\
                                                signed=False)\
                                 )\
                        )

##############################
# Decorators and wrappers
def _requires_dll_load(f):
    """Decorator to check if LanternRock DLL is loaded
    Throws LanternRockInitializationError if LanternRock DLL is not loaded
    Add to any API function which requires a DLL
    """

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if self.ignore_init:
            return None

        if self.lrdll is None:
            raise LanternRockInitializationError
        return f(self, *args, **kwargs)

    return wrapper

def _hresult(hresult):
    """Convenience wrapper to check result of the LanternRock DLL function call
     And throw LanternRockError exception of hresult != 0
     Noop otherwise
    """
    if hresult is not None and int(hresult) != 0:
        raise LanternRockError(hresult)

##############################
# Main LanternRock class
class LanternRock(object):
    """Provides access to the lanternrock API.
       Also provided helper function to convert between LanternRock lists and python data dictionaries.
    """
    ##############################
    # class variables
    _default_dll_filename = 'intel-ias3.dll' #: default filename of DLL to load
    _default_so_filename = 'libintel-ias3.so' #: default filename of linux shared library to load
    _default_dylib_filename = 'libintel-ias3.dylib' #: default filename of mac os shared library to load
    _default_lrio_filename = 'lrio.exe' #: default filename for lrio executable
    _default_dialog_filename = 'iasopt.exe' #: default filename for optin dialog executable

    ##############################
    # constructor
    def __init__(self, app_folder_path=None, default=False, ignore_init=False):
        """Constructor for LanternRock class

        Args:
            app_folder (str, optional): If not None the initialzer will look for the LanternRock DLL in this path.
            default (bool, optional): If true, Python wrapper will always call LanternRock DLL .
                                      with SDK handle parameter = null.
            ignore_init (boot, optional): Ignores initialization errors.
                                          Set to True for testing in the environments where LanternRock DLL
                                          doesn't exist

        Returns:
            LanternRock object
        """
        self.default=default
        self.ignore_init = ignore_init
        try:
            self.load_lanternrock_dll(app_folder_path)
        except Exception as e:
            if not self.ignore_init:
                raise e

    ##############################
    # DLL discovery and loading functions
    def load_lanternrock_dll(self, app_folder_path=None):
        """Loads LanternRock DLL from provided path, changing lrdll and sdk_handle members

        Args:
            app_folder_path (str, optional): Path to directory where DLL resides.
                                             If not provided, search is done via registry

        Returns:
            None
        """

        dllpath = self.find_lanternrock_dll(app_folder_path)
        if os.path.exists(dllpath):
            self.dllpath = dllpath                      #: Path to loaded DLL object
            self.lrdll = cdll.LoadLibrary(self.dllpath) #: Loaded DLL object
            self._sdk_handle = c_void_p()               #: Private SDK_HANDLE (wrapped in sdk_handle property)

    def find_lanternrock_dll(self, app_folder_path=None):
        """Searches for LanternRock DLL depending on installation options and OS type
        Only WOS is supported
        Will search for LanternRock DLL in all of the usual places.
        app_folder_path: overrides the normal search and loads the DLL from this path.
        Returns the path to the laternrock DLL if a DLL is found.
        Throws a FileNotFoundError exception if the results of its search is empty.
        Throws a NotImplementedError exception if the OS is not a windows.


        Args:
            app_folder_path (str, optional): Path to directory where DLL resides.
                                             If not provided, search is done via registry

        Returns:
            str: path to LanternRock DLL

        """
        #these are private "hidden" functions
        def load32reg():
            if self._is_32_bit_python_on_win64():
                try:
                    return self._find_dll_via_windows_registry(bitness=3264)
                except:
                    return self._find_dll_via_windows_registry(bitness=32)
            else:
                return self._find_dll_via_windows_registry(bitness=32)

        def load64reg():
            return self._find_dll_via_windows_registry(bitness=64)

        if "linux" in sys.platform:
            if app_folder_path is None:
                app_folder_path = os.path.dirname(os.path.realpath(__file__))
            if self._is_64_bit_python():
                osname = platform.platform().lower() 
                if "ubuntu" in osname or "debian" in osname: 
                    dllpath = os.path.join(os.path.join(app_folder_path,'ubuntu'),self._default_so_filename)
                else: 
                    dllpath = os.path.join(os.path.join(app_folder_path,'sles'),self._default_so_filename)
            else:
                raise NotImplementedError
            if os.path.exists(dllpath):
                return dllpath
            else:
                raise FileNotFoundError

        elif sys.platform == "darwin":
            if app_folder_path is None:
                app_folder_path = os.path.dirname(os.path.realpath(__file__))
            if self._is_64_bit_python():
                dllpath = os.path.join(os.path.join(app_folder_path,'mac'),self._default_dylib_filename)
            else:
                raise NotImplementedError
            if os.path.exists(dllpath):
                return dllpath
            else:
                raise FileNotFoundError

        elif sys.platform == "win32":
            if app_folder_path is None:
                app_folder_path = os.path.dirname(os.path.realpath(__file__))
            if self._is_64_bit_python():
                dllpath = os.path.join(os.path.join(app_folder_path,'x64'),self._default_dll_filename)
                if not os.path.exists(dllpath):
                    dllpath = load64reg()
            else:
                dllpath = os.path.join(os.path.join(app_folder_path,'x86',self._default_dll_filename))
                if not os.path.exists(dllpath):
                    dllpath = load32reg()
            if os.path.exists(dllpath):
                return dllpath
            else:
                raise FileNotFoundError

        else:
            raise NotImplementedError

    ##############################
    #Private helper functions for DLL loading
    @staticmethod
    def _load_winreg():
        """Private method.
        Opens a reference to the windows registry. This function is python version agnostic.
        Returns a reference to the Windows Registry module.
        """
        if sys.version_info[0] == 2:
            import _winreg as _winreg
        else:
            import winreg as _winreg
        return _winreg

    @staticmethod
    def _find_dll_via_windows_registry(bitness):
        """Looks for the path to the DLL in the Windows Registry.

        Args:
            bitness (int): Use value 32 for 32 bit process.
                           Use the integer value '64' for a 64 bit process.
                           Use the integer value 3264 if the process is a 32 bit process on 64 bit windows

        Returns:
             str: path to LanternRock DLL
        """

        def loadkey(bitness, version):
            _winreg = LanternRock._load_winreg() #this is a way to call one static method from another
            if bitness == 3264:
                regpath = "SOFTWARE\\WOW6432Node\\Intel\\Telemetry " + version
                subkey = "Location32"
            else:
                regpath = "SOFTWARE\\Intel\\Telemetry " + version
                if bitness == 64:
                    subkey = "Location64"
                else:
                    subkey = "Location32" 
            key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, regpath, 0, _winreg.KEY_WOW64_64KEY | _winreg.KEY_READ)
            dllpath = _winreg.QueryValueEx(key, subkey)[0]
            if sys.version_info[0]==2:
                if isinstance(dllpath, unicode):
                    dllpath = dllpath.encode('ascii')
            if not os.path.exists(dllpath):
                raise FileNotFoundError
            return dllpath

        try:
            return loadkey(bitness, '3.0')
        except:
            # for backward compatibility, if loading LR 3.0 DLL fails, try to load LR 2.0 DLL 
            return loadkey(bitness, '2.0')

    @staticmethod
    def _is_64_bit_python():
        """Returns True if the python process is running on a 64 bit Windows implementation.

        Args:
            None

        Returns:
            bool
        """

        var = sizeof (c_voidp)
        if var == 8:
            return True
        else:
            return False

    @staticmethod
    def _is_32_bit_python_on_win64():
        """Returns True if the python process is a Win32 bit running on a 64 bit Windows.

        Args:
            None

        Returns:
            bool: True if the python process is a Win32 bit running on a 64 bit Windows.
        """
        iswowprocess = c_bool(False)
        _IsWow64Process = windll.kernel32.IsWow64Process
        _GetProcess = windll.kernel32.GetCurrentProcess
        if _IsWow64Process(_GetProcess(), pointer(iswowprocess)):
            return iswowprocess.value
        else:
            raise LanternRockInitializationError()

    ##############################
    # Convenience wrappers
    # SDK Handle wrapper that takes into account default parameter
    # Note that this parameter does not have a setter
    @property
    def sdk_handle(self):
        return c_void_p() if self.default else self._sdk_handle

    @property
    def sdk_handle_ref(self):
        return c_void_p() if self.default else byref(self._sdk_handle)
        
    ##############################
    # API section
    @_requires_dll_load
    def GetApiVersion(self):
        """API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Returns a tuple with a string values of the major and minor version

        Args:
            None

        Returns:
            tuple: (major, minor) - string values of the major and minor version

        """
        major = c_uint32()
        minor = c_uint32()
        _hresult(self.lrdll.GetApiVersion(byref(major), byref(minor)))
        return (major.value, minor.value)

    @_requires_dll_load
    def HasSystemConsentFeature(self):
        """API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Returns system consent setting

        Args:
            None

        Returns:
            bool: True if system consent is given. False otherwise

        """
        consent_feature = c_uint32()
        _hresult( self.lrdll.HasSystemConsentFeature(byref(consent_feature)) )
        return consent_feature.value == 1

    @_requires_dll_load
    def GetMetadata(self):
        """API Function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws TypeError if initialization is not done
        Returns the metadata associated with this LanternRock telemetry ID

        Args:
            None

        Returns:
            dict: Metadata dictionary

        """
        length = c_uint32()
        self.lrdll.GetMetadata(self.sdk_handle, byref(c_void_p()), byref(length))
        print('I am here__________________________________________________________')
        print(length.value)
        if length.value > 0:
            outbuf = create_unicode_buffer(' '*length.value)        
            _hresult( self.lrdll.GetMetadata (self.sdk_handle, byref(outbuf), byref(length)) )
            return json.loads(outbuf.value)
        else:
            raise LanternRockError(LanternRockErrorEnum.hresult_from_win( _CODE_E_UNEXPECTED ))
            
    @_requires_dll_load
    def Upload(self, telemetryid, options):
        """ API Function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Triggers the upload of local records to the LanternRock server

        Args:
            telemetryid (str): Telemetry ID string. Any other type will throw a LanternRockArgumentError.
            options (str or dict): Configuration options as a json-string or dictionary

        Returns:
            None

        """
        if not isinstance(telemetryid, stringTypeCheck):
            raise LanternRockArgumentError
        _hresult( self.lrdll.Upload( create_unicode_buffer(telemetryid),  _format_options(options)) )

    @_requires_dll_load
    def Initialize(self, app_name, app_version, telemetryid, options=None, store_folder=None):
        """ API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws LanternRockArgumentError if arguments are wrong data type

        Args:
            app_name (str): Name of the app
            app_version (str): Version of the app
            telemetryid (str): Telemetry ID string
            options(str or dict, optional):  Configuration options as a string or dictionary
            store_folder(str, optional): Location of LanternRock data store files

        Returns:
            None

        """
        #Python allows you to expand a tuple (returned by _format_init_params) into a list of arguments
        _hresult( self.lrdll.Initialize( self.sdk_handle_ref, *_format_init_params(app_name,       \
                                                                              app_version,    \
                                                                              telemetryid,    \
                                                                              options,        \
                                                                              store_folder) ) )

    @_requires_dll_load
    def InitializeEx(self, app_name, app_version, telemetryid, datadict=None, options=None, store_folder=None):
        """ API Function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws LanternRockArgumentError if arguments are wrong data type
        Args:
            app_name (str): Name of the app
            app_version (str): Version of the app
            telemetryid (str): Telemetry ID string
            datadict (dict, optional): A python data dictionary that will be converted to: 'keys', 'values' and 'segs'
                                       See LanternRock API for details.
                                       If not provided, empty keys, values, segs tuple is created and returned.
            options(str or dict, optional): Configuration options as a string or dictionary
            store_folder(str, optional): Location of LanternRock data store files

        Returns:
            None

        """
        _hresult(self.lrdll.InitializeEx(self.sdk_handle_ref,
                                         *_format_init_params_datadict(app_name,       \
                                                              app_version,    \
                                                              telemetryid,    \
                                                              options,        \
                                                              store_folder,   \
                                                              datadict) ) )


    @_requires_dll_load
    def Deinitialize(self):
        """API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded

        Args:
            None

        Returns:
            None

        """
        _hresult( self.lrdll.Deinitialize( self.sdk_handle ) )

    @_requires_dll_load
    def BeginSession(self, name=None):
        """ API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws LanternRockArgumentError if arguments are wrong data type
        Returns a session handle

        Args:
            name (str, optional): Session name

        Returns:
            Session handle
        """
        session = c_void_p()
        _hresult( self.lrdll.BeginSession(  self.sdk_handle,                       \
                                            byref(session),                        \
                                            _create_unicode_buffer_or_void_p(name)) )
        return session

    @_requires_dll_load
    def BeginSessionEx(self, name=None, datadict=None):
        """ API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws LanternRockArgumentError if arguments are wrong data type
        Returns a session handle

        Args:
            name (str, optional): Session name
            datadict (dict, optional): A python data dictionary that will be converted to: 'keys', 'values' and 'segs'
                                       See LanternRock API for details.

        Returns:
             Session handle

        """
        session = c_void_p()
        _hresult(self.lrdll.BeginSessionEx(self.sdk_handle,                       \
                                           byref(session),                        \
                                           _create_unicode_buffer_or_void_p(name), \
                                           *_format_datadict(datadict)))
        return session

    @_requires_dll_load
    def EndSession(self, session):
        """ API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded

        Args:
            session (handle): Session handle from the previous call to BeginSession or BegionSessionEx

        Returns:
             None

        """
        _hresult( self.lrdll.EndSession(self.sdk_handle,  session) )

    @_requires_dll_load
    def Flush(self):
        """ API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Flushes stored LanternRock data to running LanternRock client

        Args:
            None

        Returns:
             None
        """
        _hresult( self.lrdll.Flush(self.sdk_handle) )

    @_requires_dll_load
    def RecordEvent(self, session, name, count, sum): #ILYA: reordered arguments!!
        """ API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws LanternRockArgumentError if arguments are wrong type
        Records an event in LanternRock

        Args:
            session (session handle or None): Session handle returned by BeginSession or BeginSessionEx
                                              None if event does not belong to a session
            name (str): Event name
            count (int): Event count
            sum (float): Event sum

        Returns:
             None

        """

        _hresult( self.lrdll.RecordEvent(self.sdk_handle, *_format_event(session, name, count, sum)) )


    @_requires_dll_load
    def RecordEventEx(self, session, name, count, sum, datadict=None): #ILYA: reordered arguments!!
        """ API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws LanternRockArgumentError if arguments are wrong data type
        Records an event in LanternRock

        Args:
            session (session handle or None): Session handle returned by BeginSession or BeginSessionEx
                                              None if event does not belong to a session
            name (str): Event name
            count (int): Event count
            sum (float): Event sum
            datadict (dict, optional): A python data dictionary that will be converted to: 'keys', 'values' and 'segs'
                                       See LanternRock API for details.

        Returns:
             None

        """

        _hresult( self.lrdll.RecordEventEx(self.sdk_handle,                                 \
                                           *_format_event_datadict(session, name, count, sum, datadict) ) )

    @_requires_dll_load
    def Attach(self,session, name, payload, payload_length=None, datadict=None):
        """ API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws LanternRockArgumentError if arguments are wrong data type
        Throws TypeError if payload or payload_length are wrong data type
        Throws ValueError if payload_length is negative
        Sends attached data to LanternRock

        Args:
            session (session handle or None): Session handle returned by BeginSession or BeginSessionEx
                                              None if event belongs to the default session
            name (str): Attach event name. Any other type will throw a LanternRockArgumentError.
            payload (str, bytearray or json): Data to attach
            payload_length (int, optional): Length of the data to attach.
                                            If not provided, length of payload parameter will used
            datadict (dict, optional): A python data dictionary that will be converted to: 'keys', 'values' and 'segs'
                                       See LanternRock API for details.

        Returns:
             GUID object: A ctypes.c_void_p or a ctypes structure with GUID value
        """
        session = _create_void_p(session)
        if not isinstance(name, stringTypeCheck):
            raise LanternRockArgumentError
        name = create_unicode_buffer(name)

        if not payload:
            raise LanternRockError(LanternRockErrorEnum.hresult_from_win(_CODE_E_INVALIDARG))

        if type(payload) is str:
            #string is unicode
            payload_length = len(payload) if payload_length is None           \
                                             or len(payload) < payload_length \
                                             else payload_length
            payload_length = payload_length * sizeof(c_wchar)
            payload_data = create_unicode_buffer(payload)

        elif type(payload) is bytearray: #TODO - Kevin can make this Py3 friendly
            payload_length = len(payload) if payload_length is None           \
                                             or len(payload) < payload_length \
                                             else payload_length
            payload_frame = c_char * payload_length
            payload_data = payload_frame.from_buffer(payload)
        else:
            #see if you can make this into a JSON object
            try:
                payload_json = json.dumps(payload)
                payload_data = create_unicode_buffer(payload_json)
                payload_length = len(payload_data) if payload_length is None           \
                                                 or len(payload_data) < payload_length \
                                                 else payload_length
                payload_length = payload_length*sizeof(c_wchar) 
            except:
                raise LanternRockError(LanternRockErrorEnum.hresult_from_win(_CODE_E_INVALIDARG))
                
        #after length is computed
        payload_length = c_uint32(payload_length)

        keys, values, segs = _format_datadict(datadict)
        guid = GUID()
        _hresult( self.lrdll.Attach(    self.sdk_handle,\
			                    		session,        \
                                        name,           \
					                    payload_data,   \
                                        payload_length, \
                                        keys,           \
                                        values,         \
                                        segs,           \
                                        byref(guid)     ) )
        return guid

    @_requires_dll_load
    def AttachFile(self, session, name, file_path, options, datadict=None):
        """ API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws LanternRockArgumentError if arguments are wrong data type
        Throws TypeError if file_path is None
        Sends a data in a file to LanternRock

        Args:
            session (session handle or None): Session handle returned by BeginSession or BeginSessionEx
                                              None if event does not belong to a session
            name (str): AttachFile event name. Any other type will throw a LanternRockArgumentError.
            file_path (str): String path to the file to upload
            options (str or dict): Configuration options as a json-string or dictionary
            datadict (dict, optional): A python data dictionary that will be converted to: 'keys', 'values' and 'segs'
                                       See LanternRock API for details.

        Returns:
             GUID object: A ctypes.c_void_p or a ctypes structure with GUID value
        """

        session = _create_void_p(session)
        if not isinstance(name, stringTypeCheck):
            raise LanternRockArgumentError
        name = create_unicode_buffer(name)
        file_path = create_unicode_buffer(file_path)
        options = _format_options(options)
        keys, values, segs = _format_datadict(datadict)
        guid = GUID()
        _hresult( self.lrdll.AttachFile(    self.sdk_handle,\
							                session,        \
                                            name,           \
                                            file_path,      \
                                            options,        \
                                            keys,           \
                                            values,         \
                                            segs,           \
                                            byref(guid)     ) )
        return guid

    @_requires_dll_load
    def GenerateEvent(self, name, count, sum, ts):
        """ API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws LanternRockArgumentError if count argument is wrong data type
        Throws TypeError if name argument is None or ts argument is wrong data type
        Generates a new event

        Args:
            name (str): Event name
            count (int): Event count
            sum (float): Event sum
            ts (datetime or int): Timestamp of event creation

        Returns:
             Event handle

        """

        ucbuffer = _create_unicode_buffer_or_void_p(name)

        count = _format_count(count)

        sum = _format_sum(sum)

        if type(ts) is datetime.datetime:
            if (sys.version_info < (3, 0)):
                ts = int(time.mktime(ts.timetuple()))
            else:
                ts = int(ts.timestamp())
        ts = c_uint64(ts) if ts else c_uint64()
        event_handle = c_void_p()
        _hresult( self.lrdll.GenerateEvent( self.sdk_handle,    \
                                            byref(event_handle),\
                                            ucbuffer,           \
                                            count,              \
                                            sum,                \
                                            ts                  ) )
        return event_handle

    @_requires_dll_load
    def GenerateEventEx(self, name, count, sum, ts, datadict=None):
        """ API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws LanternRockArgumentError if count or datadict arguments are wrong data type
        Throws TypeError if name argument is None or ts argument is wrong data type
        Generates a new event

        Args:
            name (str): Event name
            count (int): Event count
            sum (float): Event sum
            ts (datetime or int): Timestamp of event creation
            datadict (dict, optional): A python data dictionary that will be converted to: 'keys', 'values' and 'segs'
                                       See LanternRock API for details.

        Returns:
             Event handle

        """
        ucbuffer = create_unicode_buffer(name)

        count = _format_count(count)

        sum = _format_sum(sum)

        if type(ts) is datetime.datetime:
            if (sys.version_info < (3, 0)):
                ts = int(time.mktime(ts.timetuple()))
            else:
                ts = int(ts.timestamp())
        ts = c_uint64(ts) if ts else c_uint64()
        event_handle = c_void_p()
        _hresult( self.lrdll.GenerateEventEx(   self.sdk_handle,          \
                                                byref(event_handle),      \
                                                ucbuffer,                 \
                                                count,                    \
                                                sum,                      \
                                                ts,                       \
                                                *_format_datadict(datadict)) )
        return event_handle

    @_requires_dll_load
    def DestroyEvent(self, event_handle):
        """ API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Destroys an event object

        Args:
            event_handle (event handle): Event handle created by GenerateEvent or GenerateEventEx

        Returns:
            None

        """
        _hresult( self.lrdll.DestroyEvent( self.sdk_handle, event_handle) )



    @_requires_dll_load
    def RecordSession(self, name, start_timestamp, duration_seconds, event_handles=None):
        """API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws LanternRockArgumentError if arguments are wrong data type
        Records a session from multiple event handles

        Args:
            name (str): Session name
            start_timestamp(datetime or int): Start time of the session. Integer values must be non-negative
            duration_seconds(int): Duration of the session. Integer values must be positive
            event_handles(list, optional): List of event handles

        Returns:
            None

        """

        _hresult( self.lrdll.RecordSession( self.sdk_handle,                      \
                                            *_format_session(name,            \
                                                             start_timestamp, \
                                                             duration_seconds,\
                                                             event_handles )  \
                                            ) )

    @_requires_dll_load
    def RecordSessionEx(self, name, start_timestamp, duration_seconds, event_handles=None, datadict=None):
        """API function
        Throws LanternRockError on API failure
        Throws LanternRockInitializationError if LanternRock DLL is not loaded
        Throws LanternRockArgumentError if arguments are wrong data type
        Records a session from multiple event handles

        Args:
            name (str): Session name
            start_timestamp(datetime or int): Start time of the session. Integer values must be positive
            duration_seconds(int): Duration of the session. Integer values must be positive
            event_handles(list, optional): List of event handles
            datadict (dict, optional): A python data dictionary that will be converted to: 'keys', 'values' and 'segs'
                                       See LanternRock API for details.

        Returns:
            None

        """

        _hresult( self.lrdll.RecordSessionEx( self.sdk_handle,                       \
                                              *_format_session_datadict(name,        \
                                                                    start_timestamp, \
                                                                    duration_seconds,\
                                                                    event_handles,   \
                                                                    datadict) ) )

    @_requires_dll_load
    def get_lrio_path(self):
        """Returns path to lrio uploader executable

        Args:
            None

        Returns:
            None
        """
        telempath = self.dllpath.split(os.path.sep)[:-2]
        telempath.append(self._default_lrio_filename)
        return os.path.sep.join(telempath)

    @_requires_dll_load
    def get_dialog_path(self):
        """Returns path to optin dialog executable

        Args:
            None

        Returns:
            None
        """
        telempath = self.dllpath.split(os.path.sep)[:-2]
        telempath.append(self._default_dialog_filename)
        return os.path.sep.join(telempath)

    def global_upload(self):
        """Force cached event upload (for debug only). DO NOT use in apps
        
        Args:
            None
            
        Returns:
            None
        """
        try:
            lrio_path = self.get_lrio_path()
            if not os.path.isfile(lrio_path):
                raise LanternRockError(LanternRockErrorEnum.hresult_from_win(_CODE_ERROR_NOT_FOUND))

            #caller is responsible for catching exceptoins
            FNULL = open(os.devnull, 'w')
            subprocess.check_output(self.get_lrio_path(), stdin=None, stderr=FNULL) #blocking, output goes to devnull
        except Exception as e:
            if not self.ignore_init:
                raise e

    def opt_in_dialog(self):
        """Open consent dialog
        
        Args:
            None
            
        Returns:
            None
        """
        try:
            createNoWindowFlag = 0x08000000
            dialog_path = self.get_dialog_path()
            if not os.path.isfile(dialog_path):
				#H: might want to use HRESULT_FROM_WIN32(ERROR_NOT_FOUND) instead of -2
                #I: done
                raise LanternRockError(LanternRockErrorEnum.hresult_from_win(_CODE_ERROR_NOT_FOUND))

            #caller is responsible for catching exceptoins
            p1 = subprocess.Popen(dialog_path, stdin = None, stdout = None, stderr = None, creationflags = createNoWindowFlag)
            # We actually _do_ want it to wait.
            retCode = p1.wait()
            if retCode != 0 and retCode != 1:
                raise Exception("Return code (%d) from call to %s" % (retCode, dialog_path))
     
        except Exception as e:
            if not self.ignore_init:
                raise e

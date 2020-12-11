#https://msdn.microsoft.com/en-us/library/windows/desktop/ms690088(v=vs.85).aspx
FACILITY_NULL     = 0
FACILITY_RPC      = 1
FACILITY_DISPATCH = 2
FACILITY_STORAGE  = 3
FACILITY_ITF      = 4
FACILITY_WIN32    = 7
FACILITY_WINDOWS  = 8


class WinError(object):
    # these are initialized on first call that they are needed
    # it is assumed we never add to these later
    _lookup = None # for cachine reverse lookup
    _customCodes = {}
    @classmethod
    def _init(cls):
        if cls._lookup == None:
            cls._lookup  = {}
            try:
                import winerror
                #should make us linux-proof for now
                for k,v in winerror.__dict__.items():
                    try:
                        if isinstance(v,(int,long)):
                            cls._lookup[cls.hresult_from_win(v)] = k
                    except:
                        pass
                for k,v in cls._customCodes.items(): #doing this explicitly instead of merging the dictionaries
                    try:
                        if isinstance(v,(int,long)):
                            cls._lookup[v] = k
                    except:
                        pass
            except Exception as e:
                pass
                
    @classmethod
    def hresult_from_win(cls, value):
        if value <= 0:
            return value
        else:
            return ((value & 0x0000FFFF) | (FACILITY_WIN32 << 16) | 0x80000000)  - (1<<32)
   
    
    @classmethod
    def lookup(cls,value):
        """given the error code, find out its name. given name, find out error code"""
        cls._init()
        return cls._lookup.get(value,"UNKNOWN")

    @classmethod
    def keys(cls):
        """returns keys that are in this enum"""
        cls._init()
        return cls._lookup.keys()

    @classmethod
    def values(cls):
        """returns values that are in this enum"""
        cls._init()
        return cls._lookup.values()

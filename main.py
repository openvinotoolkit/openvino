# Python 3

import os
import re

global changes 
changes = 0

def replace_cl_function_calls(directory):
    """
    Recursively reads all.h,.hpp,.c, and.cpp files in the given directory
    and replaces function calls starting with 'cl' with 'call_c'.
    """
    # Define the file extensions to process
    extensions = ('.h', '.hpp', '.c', '.cpp')
    
    # Regex pattern to find function calls starting with "cl"
    # Matches clFunctionName(...) but avoids replacing variable names
    pattern = re.compile(r'\b(cl[A-Z]\w*)\s*(\()', re.MULTILINE)
    
    # Walk through all files in the directory recursively
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(extensions):
                file_path = os.path.join(root, file_name)
                
                # Read file content
                with open(r"C:\\Users\\gta\\openvino\\src\\plugins\\intel_gpu\\src\\runtime\\ocl\\cl_help.hpp", 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace matches with "call_c" prefix
                global changes 
                changes += len(pattern.findall(content))
                new_content = pattern.sub(r'call_\1\2', content)
                
                # Write the updated content back to the file only if changes were made
                if new_content!= content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Updated file: {file_path}")

def edit_one():
    pattern = re.compile(r'\b(cl[A-Z]\w*)\s*(\()', re.MULTILINE)
    file_path = r"C:\\Users\\gta\\openvino\\src\\plugins\\intel_gpu\\src\\runtime\\ocl\\cl_help.hpp"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Replace matches with "call_c" prefix
        global changes 
        changes += len(pattern.findall(content))
        new_content = pattern.sub(r'call_\1\2', content)
        
        # Write the updated content back to the file only if changes were made
        if new_content!= content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated file: {file_path}")

# Usage Example:
#replace_cl_function_calls(r"C:\\Users\\gta\\openvino\\src\\plugins\\intel_gpu\\src")
#print("ilosc zmian", changes)
#replace_cl_function_calls(r"C:\\Users\\gta\\openvino\\src\\plugins\\intel_gpu\\include")
#print("ilosc zmian", changes)
#replace_cl_function_calls(r"C:\\Users\\mmiotk\\openvino\\src\\plugins\\intel_gpu\\thirdparty\\xetla")
import ctypes
import os

def demangle_symbol(mangled_name):
    dbghelp = ctypes.WinDLL("DbgHelp.dll")
    undecorate = dbghelp.UnDecorateSymbolName
    undecorate.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint]
    undecorate.restype = ctypes.c_uint

    buffer = ctypes.create_string_buffer(256)
    flags = 0x1000  # UNDNAME_COMPLETE

    result = undecorate(mangled_name.encode('utf-8'), buffer, ctypes.sizeof(buffer), flags)
    if result:
        return buffer.value.decode('utf-8')
    return None

def inspect_clGetEventProfilingInfo():
    dll_name = "OpenCL.dll"
    dll_path = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", dll_name)

    try:
        opencl = ctypes.WinDLL(dll_path)
        func_ptr = getattr(opencl, "clGetEventProfilingInfo", None)

        if func_ptr:
            address = ctypes.cast(func_ptr, ctypes.c_void_p).value
            print(f"clGetEventProfilingInfo znaleziony w {dll_name}")
            print(f"Adres funkcji: 0x{address:X}")

            # Próbujemy zdemangle'ować nazwę (choć w C nie ma manglingu)
            mangled = "clGetEventProfilingInfo"  # w C nie ma manglingu, ale dla C++ byłoby np. "?clGetEventProfilingInfo@@..."
            demangled = demangle_symbol(mangled)
            print("Zdemangle'owana nazwa:", demangled if demangled else "(brak zmiany — prawdopodobnie C)")
        else:
            print("Funkcja clGetEventProfilingInfo nie została znaleziona w DLL.")
    except Exception as e:
        print(f"Błąd podczas ładowania DLL: {e}")

inspect_clGetEventProfilingInfo()

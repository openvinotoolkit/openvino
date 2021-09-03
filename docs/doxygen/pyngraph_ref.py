import os
import sys
import argparse
import importlib
import inspect
from pathlib import Path
from mypy.stubgenc import generate_c_type_stub, is_c_type, add_typing_import, strip_or_import, infer_method_sig,\
    get_type_fullname, method_name_sort_key, is_c_method, is_c_classmethod, \
    is_skipped_attribute, is_c_property, is_c_property_readonly
from mypy.moduleinspect import is_c_module
from typing import List, Dict, Tuple, Optional, Mapping, Any, Set
from types import ModuleType
from mypy.stubdoc import infer_sig_from_docstring, FunctionSig, infer_arg_sig_from_anon_docstring, \
    infer_ret_type_sig_from_anon_docstring, infer_prop_type_from_docstring


def is_c_function(obj: object) -> bool:
    return inspect.isbuiltin(obj) or type(obj) is type(ord)


def generate_stub_for_c_module(module_name: str,
                               target: str,
                               sigs: Optional[Dict[str, str]] = None,
                               class_sigs: Optional[Dict[str, str]] = None) -> None:
    """Generate stub for C module.

    This combines simple runtime introspection (looking for docstrings and attributes
    with simple builtin types) and signatures inferred from .rst documentation (if given).

    If directory for target doesn't exist it will be created. Existing stub
    will be overwritten.
    """
    module = importlib.import_module(module_name)
    assert is_c_module(module), '%s is not a C module' % module_name
    subdir = os.path.dirname(target)
    if subdir and not os.path.isdir(subdir):
        os.makedirs(subdir)
    imports = []  # type: List[str]
    functions = []  # type: List[str]
    done = set()
    items = sorted(module.__dict__.items(), key=lambda x: x[0])
    for name, obj in items:
        if is_c_function(obj):
            generate_c_function_stub(module, name, obj, functions, imports=imports, sigs=sigs)
            done.add(name)
    types = []  # type: List[str]
    for name, obj in items:
        if name.startswith('__') and name.endswith('__'):
            continue
        if is_c_type(obj):
            generate_c_type_stub(module, name, obj, types, imports=imports, sigs=sigs,
                                 class_sigs=class_sigs)
            done.add(name)
    variables = []
    for name, obj in items:
        if name.startswith('__') and name.endswith('__'):
            continue
        if name not in done and not inspect.ismodule(obj):
            type_str = type(obj).__name__
            if type_str not in ('int', 'str', 'bytes', 'float', 'bool'):
                type_str = 'Any'
            variables.append('%s: %s' % (name, type_str))
    output = []
    output.append('\n')
    output.append('## @defgroup ngraph_python_pyngraph pyngraph')
    output.append('# pyngraph module')
    output.append('# @ingroup ngraph_python_api')
    output.append('\n')
    for line in sorted(set(imports)):
        output.append(line)
    for line in variables:
        output.append(line)
    if output and functions:
        output.append('')
    for line in functions:
        output.append(line)
    for line in types:
        if line.startswith('class') and output and output[-1]:
            output.append('')
        output.append(line)
    output = add_typing_import(output)
    with open(target, 'w') as file:
        for line in output:
            file.write('%s\n' % line)


def generate_c_type_stub(module: ModuleType,
                         class_name: str,
                         obj: type,
                         output: List[str],
                         imports: List[str],
                         sigs: Optional[Dict[str, str]] = None,
                         class_sigs: Optional[Dict[str, str]] = None) -> None:
    """Generate stub for a single class using runtime introspection.

    The result lines will be appended to 'output'. If necessary, any
    required names will be added to 'imports'.
    """
    # typeshed gives obj.__dict__ the not quite correct type Dict[str, Any]
    # (it could be a mappingproxy!), which makes mypyc mad, so obfuscate it.
    obj_dict = getattr(obj, '__dict__')  # type: Mapping[str, Any]  # noqa
    items = sorted(obj_dict.items(), key=lambda x: method_name_sort_key(x[0]))
    methods = []  # type: List[str]
    properties = []  # type: List[str]
    done = set()  # type: Set[str]
    for attr, value in items:
        if is_c_method(value) or is_c_classmethod(value):
            done.add(attr)
            if not is_skipped_attribute(attr):
                if attr == '__new__':
                    # TODO: We should support __new__.
                    if '__init__' in obj_dict:
                        # Avoid duplicate functions if both are present.
                        # But is there any case where .__new__() has a
                        # better signature than __init__() ?
                        continue
                    attr = '__init__'
                if is_c_classmethod(value):
                    methods.append('@classmethod')
                    self_var = 'cls'
                else:
                    self_var = 'self'
                generate_c_function_stub(module, attr, value, methods, imports=imports,
                                         self_var=self_var, sigs=sigs, class_name=class_name,
                                         class_sigs=class_sigs)
        elif is_c_property(value):
            done.add(attr)
            generate_c_property_stub(attr, value, properties, is_c_property_readonly(value))

    variables = []
    for attr, value in items:
        if is_skipped_attribute(attr):
            continue
        if attr not in done:
            variables.append('%s: Any = ...' % attr)
    all_bases = obj.mro()
    if all_bases[-1] is object:
        # TODO: Is this always object?
        del all_bases[-1]
    # remove pybind11_object. All classes generated by pybind11 have pybind11_object in their MRO,
    # which only overrides a few functions in object type
    if all_bases and all_bases[-1].__name__ == 'pybind11_object':
        del all_bases[-1]
    # remove the class itself
    all_bases = all_bases[1:]
    # Remove base classes of other bases as redundant.
    bases = []  # type: List[type]
    for base in all_bases:
        if not any(issubclass(b, base) for b in bases):
            bases.append(base)
    if bases:
        bases_str = '(%s)' % ', '.join(
            strip_or_import(
                get_type_fullname(base),
                module,
                imports
            ) for base in bases
        )
    else:
        bases_str = ''
    
    doxygen_group = '\n## @ingroup ngraph_python_pyngraph\n'
    if not methods and not variables and not properties:
        output.append(doxygen_group + 'class %s%s: ...' % (class_name, bases_str))
    else:
        output.append(doxygen_group + 'class %s%s:' % (class_name, bases_str))
        for variable in variables:
            output.append('    %s' % variable)
        for method in methods:
            output.append('    %s' % method)
        for prop in properties:
            output.append('    %s' % prop)


def generate_c_function_stub(module: ModuleType,
                             name: str,
                             obj: object,
                             output: List[str],
                             imports: List[str],
                             self_var: Optional[str] = None,
                             sigs: Optional[Dict[str, str]] = None,
                             class_name: Optional[str] = None,
                             class_sigs: Optional[Dict[str, str]] = None) -> None:
    """Generate stub for a single function or method.

    The result (always a single line) will be appended to 'output'.
    If necessary, any required names will be added to 'imports'.
    The 'class_name' is used to find signature of __init__ or __new__ in
    'class_sigs'.
    """
    if sigs is None:
        sigs = {}
    if class_sigs is None:
        class_sigs = {}

    ret_type = 'None' if name == '__init__' and class_name else 'Any'

    docstr = getattr(obj, '__doc__', None)
    if (name in ('__new__', '__init__') and name not in sigs and class_name and
            class_name in class_sigs):
        inferred = [FunctionSig(name=name,
                                args=infer_arg_sig_from_anon_docstring(class_sigs[class_name]),
                                ret_type=ret_type)]  # type: Optional[List[FunctionSig]]
    else:
        docstr = getattr(obj, '__doc__', None)
        inferred = infer_sig_from_docstring(docstr, name)
        if not inferred:
            if class_name and name not in sigs:
                inferred = [FunctionSig(name, args=infer_method_sig(name), ret_type=ret_type)]
            else:
                inferred = [FunctionSig(name=name,
                                        args=infer_arg_sig_from_anon_docstring(
                                            sigs.get(name, '(*args, **kwargs)')),
                                        ret_type=ret_type)]

    is_overloaded = len(inferred) > 1 if inferred else False
    if is_overloaded:
        imports.append('from typing import overload')
    if inferred:
        for signature in inferred:
            sig = []
            for arg in signature.args:
                if arg.name == self_var:
                    arg_def = self_var
                else:
                    arg_def = arg.name
                    if arg_def == 'None':
                        arg_def = '_none'  # None is not a valid argument name

                    if arg.type:
                        arg_def += ": " + strip_or_import(arg.type, module, imports)

                    if arg.default:
                        arg_def += " = ..."

                sig.append(arg_def)

            if is_overloaded:
                output.append('@overload')

            docstr = getattr(obj, '__doc__', None)
            docstr = docstr.split('\n')
            comm = []
            for d in docstr:
                comm.append(d.strip())

            comm.pop(0)
            if comm:
                item = comm[0]
                while not item:
                    try:
                        comm.pop(0)
                        item = comm[0]
                    except IndexError:
                        break
            if comm:
                comm.insert(0, '"""')
                comm.append('"""')
                comm = ''.join(map(lambda c: '\n        ' + c, comm))
            else:
                comm = '\n        pass'

            output.append('def {function}({args}) -> {ret}:{comm}'.format(
                function=name,
                args=", ".join(sig),
                ret=strip_or_import(signature.ret_type, module, imports),
                comm=comm
            ))


def generate_c_property_stub(name: str, obj: object, output: List[str], readonly: bool) -> None:
    """Generate property stub using introspection of 'obj'.

    Try to infer type from docstring, append resulting lines to 'output'.
    """
    def infer_prop_type(docstr: Optional[str]) -> Optional[str]:
        """Infer property type from docstring or docstring signature."""
        if docstr is not None:
            inferred = infer_ret_type_sig_from_anon_docstring(docstr)
            if not inferred:
                inferred = infer_prop_type_from_docstring(docstr)
            return inferred
        else:
            return None

    inferred = infer_prop_type(getattr(obj, '__doc__', None))
    if not inferred:
        fget = getattr(obj, 'fget', None)
        inferred = infer_prop_type(getattr(fget, '__doc__', None))
    if not inferred:
        inferred = 'Any'

    docstr = getattr(obj, '__doc__', None)
    docstr = docstr.split('\n')
    comm = []
    for d in docstr:
        comm.append(d.strip())

    comm.pop(0)
    if comm:
        item = comm[0]
        while not item and len(comm) > 1:
            try:
                comm.pop(0)
                item = comm[0]
            except IndexError:
                break
    if comm:
        comm.insert(0, '"""')
        comm.append('"""')
        comm = ''.join(map(lambda c: '\n        ' + c, comm))
    else:
        comm = '\n        pass'

    output.append('@property')
    output.append('def {}(self) -> {}:{}'.format(name, inferred, comm))
    if not readonly:
        output.append('@{}.setter'.format(name))
        output.append('def {}(self, val: {}) -> None:{}'.format(name, inferred, comm))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('openvino_bin_dir', type=Path)
    parser.add_argument('pyngraph_dir', type=Path)
    parser.add_argument('pyngraph_api_output', type=Path)
    args = parser.parse_args()

    OPENVINO_BIN_DIR = args.openvino_bin_dir.resolve()
    PYNGRAPH_DIR = args.pyngraph_dir.resolve()

    os.environ['PATH'] = str(OPENVINO_BIN_DIR) + os.pathsep + os.environ['PATH']
    sys.path.append(str(PYNGRAPH_DIR))


    OUTPUT_DIR = args.pyngraph_api_output.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PYNGRAPH_OUTPUT_FILE = str(OUTPUT_DIR.joinpath('pyngraph.py').resolve())
    generate_stub_for_c_module('_pyngraph', PYNGRAPH_OUTPUT_FILE)


if __name__ == '__main__':
    main()

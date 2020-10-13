
import os, sys
import argparse
import re

parser = argparse.ArgumentParser(description='DL Benchmark tool set')

parser.add_argument('-o', '--output_path', required=True, help='path to output file')
parser.add_argument('-t', '--trace_path', required=True, help='path to folder with traces')


args = parser.parse_args()

filenames = [ os.path.join(args.trace_path, f) for f in os.listdir(args.trace_path) ] if os.path.isdir(args.trace_path) else [ args.trace_path ]
output_file = open(args.output_path, 'w')
for filename in filenames:
  if os.path.isdir(filename):
    continue
  registered_passes = set()
  pass_matchers = {}
  matcher_callbacks = set()
  macros = {}
  with open(filename, 'r') as f:
    for line in f.readlines():
      match = re.match(r'(\S+),(.*?),([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+)', line)
      if match:
        domain = match.group(1)
        name = match.group(2)
        if domain == "CC0OV":
          def process_macro_name(name):
            return name.replace('::', '_').replace('.', '_')
          macro_name = domain + "_" + process_macro_name(name)
          macros[macro_name] = "1"
        elif domain == "CC_IETransoformPassRegister" or domain == "CC_nGraphPassRegister":
          registered_passes.add(name.split('::')[-1])
        elif domain == "CC_nGraphPassCallback":
          matcher_callbacks.add(name.split('::')[-1])
        elif domain == "CC_nGraphPassAddMatcher":
          pass_name, matcher_name = name.split('_')
          pass_name = pass_name.split('::')[-1]
          if pass_name not in pass_matchers:
            pass_matchers[pass_name] = []
          pass_matchers[pass_name].append(matcher_name)
          registered_passes.add(pass_name)

    for pass_name in registered_passes:
      register = 1
      if pass_name in pass_matchers.keys():
        register = int(any(x in matcher_callbacks  for x in pass_matchers[pass_name]))
      macro_name = "REGISTER_PASS_" + process_macro_name(pass_name)
      macros[macro_name] = register

    for matcher_name, matchers in pass_matchers.items():
      for matcher_name in matchers:
          add_matcher = 1 if matcher_name in matcher_callbacks else 0
          macro_name = "REGISTER_PASS_" + process_macro_name(matcher_name)
          macros[macro_name] = add_matcher

    for macro_name, macro_value in sorted(macros.items()):
      output_file.write("#ifndef {}\n".format(macro_name))
      output_file.write("#define {} {}\n".format(macro_name, macro_value))
      output_file.write("#endif\n")


print("{} dumped".format(args.output_path))

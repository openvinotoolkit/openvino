import pkg_resources
import re
import sys
import os

req_file=sys.argv[1]

constraints = {}
constraints_path = []
requirements = []

# read requirements and find constraints file
with open(req_file) as f:
    raw_requirements = f.readlines()
for line in raw_requirements:
    if line.startswith("-c"):
        constraints_path.append(os.path.join(os.path.dirname(req_file), line.split(' ')[1][:-1]))

# read constraints if they exist
if constraints_path:
    for constraint_path in constraints_path:
        with open(constraint_path) as f:
            raw_constraints = f.readlines()
        for line in raw_constraints:
            if line.startswith("#") or line=="\n":
                continue
            line = line.replace("\n", "")
            package, delimiter, constraint = re.split("(~|=|<|>|;)", line, maxsplit=1)
            if constraints.get(package) is None:
                constraints[package] = [delimiter + constraint]
            else:
                constraints[package].extend([delimiter + constraint])
    for line in raw_requirements:
        if line.startswith(("#", "-c")):
            continue
        line = line.replace("\n", "")
        if re.search("(~|=|<|>|;)", line):
            requirements.append(line)
        else:
            constraint = constraints.get(line)
            if constraint:
                for marker in constraint:
                    requirements.append(line+marker)
            else:
                requirements.append(line)
else:
    requirements = raw_requirements

try:
    pkg_resources.require(requirements)
except Exception as inst:
    pattern = re.compile(
        r"protobuf .*, Requirement.parse\('protobuf<=3\.20\.0,>=3\.1\.0'\), {'paddlepaddle'}"
    )
    result = pattern.findall(str(inst))
    if len(result) == 0:
        raise inst
    else:
        env = pkg_resources.Environment()
        env['protobuf'].clear()
        env.add(pkg_resources.DistInfoDistribution(project_name="protobuf", version="3.20.0"))
        ws = pkg_resources.working_set
        reqs = pkg_resources.parse_requirements(open(req_file, mode='r'))
        dists = ws.resolve(reqs, env, replace_conflicting=True)

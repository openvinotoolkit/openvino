from collections import OrderedDict


def filter_package_configs(pkg_configurations, query):
    postfix = _infix_to_postfix(query) if query else []
    result = OrderedDict()
    for pref, data in pkg_configurations.items():
        if _evaluate_postfix_with_info(postfix, data):
            result[pref] = data
    return result


def _evaluate_postfix_with_info(postfix, binary_info):

    # Evaluate conaninfo with the expression

    def evaluate_info(expression):
        """Receives an expression like compiler.version="12"
        Uses conan_vars_info in the closure to evaluate it"""
        name, value = expression.split("=", 1)
        value = value.replace("\"", "")
        return _evaluate(name, value, binary_info)

    return _evaluate_postfix(postfix, evaluate_info)


def _evaluate(prop_name, prop_value, binary_info):
    """
    Evaluates a single prop_name, prop_value like "os", "Windows" against
    conan_vars_info.serialize_min()
    """

    def compatible_prop(setting_value, _prop_value):
        return (_prop_value == setting_value) or (_prop_value == "None" and setting_value is None)

    # TODO: Necessary to generalize this query evaluation to include all possible fields
    info_settings = binary_info.get("settings", {})
    info_options = binary_info.get("options", {})

    if not prop_name.startswith("options."):
        return compatible_prop(info_settings.get(prop_name), prop_value)
    else:
        prop_name = prop_name[len("options."):]
        return compatible_prop(info_options.get(prop_name), prop_value)


def _is_operator(el):
    return el in ["|", "&"]


def _parse_expression(subexp):
    """Expressions like:
     compiler.version=12
     compiler="Visual Studio"
     arch="x86"
     Could be replaced with another one to parse different queries """
    ret = ""
    quoted = False
    for char in subexp:
        if char in ['"', "'"]:  # Fixme: Mix quotes
            quoted = not quoted
            ret += char
            continue

        if quoted:
            ret += char
        elif char == " " or _is_operator(char) or char in [")", "("]:
            break
        else:
            ret += char

    if "=" not in ret:
        raise Exception("Invalid expression: %s" % ret)

    return ret


def _evaluate_postfix(postfix, evaluator):
    """
    Evaluates a postfix expression and returns a boolean
    @param postfix:  Postfix expression as a list
    @param evaluator: Function that will return a bool receiving expressions
                      like "compiler.version=12"
    @return: bool
    """
    if not postfix:  # If no query return all?
        return True

    stack = []
    for el in postfix:
        if not _is_operator(el):
            stack.append(el)
        else:
            o1 = stack.pop()
            o2 = stack.pop()
            if not isinstance(o1, bool):
                o1 = evaluator(o1)
            if not isinstance(o2, bool):
                o2 = evaluator(o2)

            if el == "|":
                res = o1 or o2
            elif el == "&":
                res = o1 and o2
            stack.append(res)
    if len(stack) != 1:
        raise Exception("Bad stack: %s" % str(stack))
    elif not isinstance(stack[0], bool):
        return evaluator(stack[0])  # Single Expression without AND or OR
    else:
        return stack[0]


def _infix_to_postfix(exp):
    """
    Translates an infix expression to postfix using an standard algorithm
    with little hacks for parse complex expressions like "compiler.version=4"
    instead of just numbers and without taking in account the operands priority
    except the priority specified by the "("

    @param exp: String with an expression with & and | operators,
        e.g.: "os=Windows & (compiler=gcc | compiler.version=3)"
        e.g.: "os=Windows AND (compiler=gcc or compiler.version=3)"
    @return List with the postfix expression
    """

    # To ease the parser, operators only with one character
    exp = exp.replace(" AND ", "&").replace(" OR ", "|").replace(" and ", "&").replace(" or ", "|")
    output = []
    stack = []

    i = -1
    while i < len(exp) - 1:
        i += 1
        char = exp[i]
        if char == " ":  # Ignore spaces between expressions and operators
            continue
        if char == ")":  # Pop the stack until "(" and send them to output
            popped = None
            while popped != "(" and stack:
                popped = stack.pop()
                if popped != "(":
                    output.append(popped)
            if popped != "(":
                raise Exception("Bad expression, not balanced parenthesis")
        elif _is_operator(char):
            # Same operations has the same priority
            # replace this lines if the operators need to have
            # some priority
            if stack and _is_operator(stack[:-1]):
                popped = stack.pop()
                output.append(popped)
            stack.append(char)
        elif char == "(":
            stack.append("(")
        else:  # Parse an expression, in our case something like "compiler=gcc"
            expr = _parse_expression(exp[i:])
            i += len(expr) - 1
            output.append(expr)

    # Append remaining elements
    if "(" in stack:
        raise Exception("Bad expression, not balanced parenthesis")
    output.extend(stack)
    return output

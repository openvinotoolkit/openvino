from functools import total_ordering
from typing import Optional

from conan.internal.model.version import Version
from conan.errors import ConanException


@total_ordering
class _Condition:
    def __init__(self, operator, version):
        self.operator = operator
        self.display_version = version

        value = str(version)
        if (operator == ">=" or operator == "<") and "-" not in value and version.build is None:
            value += "-"
        self.version = Version(value)

    def __str__(self):
        return f"{self.operator}{self.display_version}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.operator, self.version))

    def __lt__(self, other):
        # Notice that this is done on the modified version, might contain extra prereleases
        if self.version < other.version:
            return True
        elif self.version == other.version:
            if self.operator == "<":
                if other.operator == "<":
                    return self.display_version.pre is not None
                else:
                    return True
            elif self.operator == "<=":
                if other.operator == "<":
                    return False
                else:
                    return self.display_version.pre is None
            elif self.operator == ">":
                if other.operator == ">":
                    return self.display_version.pre is None
                else:
                    return False
            else:
                if other.operator == ">":
                    return True
                # There's a possibility of getting here while validating if a range is non-void
                # by comparing >= & <= for lower limit <= upper limit
                elif other.operator == "<=":
                    return True
                else:
                    return self.display_version.pre is not None
        return False

    def __eq__(self, other):
        return (self.display_version == other.display_version and
                self.operator == other.operator)


class _ConditionSet:

    def __init__(self, expression, prerelease):
        expressions = expression.split()
        if not expressions:
            # Guarantee at least one expression
            expressions = [""]

        self.prerelease = prerelease
        self.conditions = []
        for e in expressions:
            e = e.strip()
            self.conditions.extend(self._parse_expression(e))

    @staticmethod
    def _parse_expression(expression):
        if expression in ("", "*"):
            return [_Condition(">=", Version("0.0.0"))]
        elif len(expression) == 1:
            raise ConanException(f'Error parsing version range "{expression}"')

        operator = expression[0]
        if operator not in (">", "<", "^", "~", "="):
            if expression[-1] == "*":  # Handle patterns like "1.2.*"
                operator = "*"
                expression = expression[:-1]
            else:
                operator = "="
            index = 0
        else:
            index = 1
        if operator in (">", "<"):
            if expression[1] == "=":
                operator += "="
                index = 2
        elif expression[1] == "=":
            raise ConanException(f"Invalid version range operator '{operator}=' in {expression}, you should probably use {operator} instead.")
        version = expression[index:]
        if version == "":
            raise ConanException(f'Error parsing version range "{expression}"')
        if operator == "~":  # tilde minor
            if "-" not in version:
                version += "-"
            v = Version(version)
            index = 1 if len(v.main) > 1 else 0
            return [_Condition(">=", v), _Condition("<", v.upper_bound(index))]
        elif operator == "^":  # caret major
            v = Version(version)

            def first_non_zero(main):
                for i, m in enumerate(main):
                    if m != 0:
                        return i
                return len(main)

            initial_index = first_non_zero(v.main)
            return [_Condition(">=", v), _Condition("<", v.upper_bound(initial_index))]
        else:
            return [_Condition(operator, Version(version))]

    def valid(self, version, conf_resolve_prepreleases):
        if version.pre:
            # Follow the expression desires only if core.version_ranges:resolve_prereleases is None,
            # else force to the conf's value
            if conf_resolve_prepreleases is None:
                if not self.prerelease:
                    return False
            elif conf_resolve_prepreleases is False:
                return False
        for condition in self.conditions:
            if condition.operator == ">":
                if not version > condition.version:
                    return False
            elif condition.operator == "<":
                if not version < condition.version:
                    return False
            elif condition.operator == ">=":
                if not version >= condition.version:
                    return False
            elif condition.operator == "<=":
                if not version <= condition.version:
                    return False
            elif condition.operator == "=":
                if not version == condition.version:
                    return False
            elif condition.operator == "*":
                if not str(version).startswith(str(condition.version)):
                    return False
        return True


class VersionRange:
    def __init__(self, expression):
        self._expression = expression
        tokens = expression.split(",")
        prereleases = False
        for t in tokens[1:]:
            if "include_prerelease" in t:
                if "include_prerelease=" in t:
                    from conan.api.output import ConanOutput
                    ConanOutput().warning(
                        f'include_prerelease version range option in "{expression}" does not take an attribute, '
                        'its presence unconditionally enables prereleases')
                prereleases = True
                break
            else:
                t = t.strip()
                if len(t) > 0 and t[0].isalpha():
                    from conan.api.output import ConanOutput
                    ConanOutput().warning(f'Unrecognized version range option "{t}" in "{expression}"')
                else:
                    raise ConanException(f'"{t}" in version range "{expression}" is not a valid option')
        version_expr = tokens[0]
        self.condition_sets = []
        for alternative in version_expr.split("||"):
            self.condition_sets.append(_ConditionSet(alternative, prereleases))

    def __str__(self):
        return self._expression

    def contains(self, version: Version, resolve_prerelease: Optional[bool]):
        """
        Whether <version> is inside the version range

        :param version: Version to check against
        :param resolve_prerelease: If ``True``, ensure prereleases can be resolved in this range
        If ``False``, prerelases can NOT be resolved in this range
        If ``None``, prereleases are resolved only if this version range expression says so
        :return: Whether the version is inside the range
        """
        assert isinstance(version, Version), type(version)
        for condition_set in self.condition_sets:
            if condition_set.valid(version, resolve_prerelease):
                return True
        return False

    def intersection(self, other):
        conditions = []

        def _calculate_limits(operator, lhs, rhs):
            limits = ([c for c in lhs.conditions if operator in c.operator]
                      + [c for c in rhs.conditions if operator in c.operator])
            if limits:
                return sorted(limits, reverse=operator == ">")[0]

        prerelease = True
        for lhs_conditions in self.condition_sets:
            for rhs_conditions in other.condition_sets:
                internal_conditions = []
                lower_limit = _calculate_limits(">", lhs_conditions, rhs_conditions)
                upper_limit = _calculate_limits("<", lhs_conditions, rhs_conditions)
                if lower_limit:
                    internal_conditions.append(lower_limit)
                if upper_limit:
                    internal_conditions.append(upper_limit)
                if internal_conditions and (not lower_limit or not upper_limit or lower_limit <= upper_limit):
                    conditions.append(internal_conditions)
                # conservative approach: if any of the conditions forbid prereleases, forbid them in the result
                if not lhs_conditions.prerelease or not rhs_conditions.prerelease:
                    prerelease = False

        if not conditions:
            return None
        expression = ' || '.join(' '.join(str(c) for c in cs) for cs in conditions) + (', include_prerelease' if prerelease else '')
        result = VersionRange(expression)
        # TODO: Direct definition of conditions not reparsing
        # result.condition_sets = self.condition_sets + other.condition_sets
        return result

    def version(self):
        return Version(f"[{self._expression}]")


def validate_conan_version(required_range):
    from conan import __version__  # To avoid circular imports
    clientver = Version(__version__)
    version_range = VersionRange(required_range)
    for conditions in version_range.condition_sets:
        conditions.prerelease = True
    if not version_range.contains(clientver, resolve_prerelease=None):
        raise ConanException("Current Conan version ({}) does not satisfy "
                             "the defined one ({}).".format(clientver, required_range))

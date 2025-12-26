from conan.api.model import RecipeReference
from conan.api.output import ConanOutput


def ref_matches(ref, pattern, is_consumer):
    if not ref or not str(ref):
        assert is_consumer
        ref = RecipeReference.loads("*/*")  # FIXME: ugly
    if "[" in pattern:
        ConanOutput().warning(f"Pattern {pattern} contains a version range, which has no effect. "
                              f"Only '&' for consumer and '*' as wildcard are supported "
                              f"in this context.", warn_tag="risk")
    return ref.matches(pattern, is_consumer=is_consumer)

from conan.errors import ConanException


REVISIONS = "revisions"  # capability


def check_duplicated_generator(generator, conanfile):
    if generator.__class__.__name__ in conanfile.generators:
        raise ConanException(f"{generator.__class__.__name__} is declared in the generators "
                             "attribute, but was instantiated in the generate() method too. "
                             "It should only be present in one of them.")

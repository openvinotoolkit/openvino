from enum import Enum


class EventType(Enum):
    pre_commit = 'pre_commit'
    commit = 'commit'

# TODO: add enum for product type to validate it

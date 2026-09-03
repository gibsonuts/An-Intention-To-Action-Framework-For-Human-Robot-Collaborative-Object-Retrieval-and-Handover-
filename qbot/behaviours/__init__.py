# behaviours/__init__.py
# Importing these modules registers actions via the decorator.
from .idle_lookaround import IdleLookaroundAction  # noqa: F401
from .point_at_person import PointAtPersonAction  # noqa: F401
from .handover_to_hand import HandoverToHandAction  # noqa: F401

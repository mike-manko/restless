from collections.abc import Mapping
from typing import Any


# Replace every local $ref in a piece of the spec with the schema it points to
def resolve_refs(node: Any, spec: Mapping[str, Any], expanding: frozenset[str] = frozenset()) -> Any:
    """Return a copy of node where each {"$ref": "#/components/..."} is replaced by its target, recursively.

    `expanding` holds the refs being expanded on the current branch, from the root down to here.
    It is a frozenset passed down, not a set shared across calls, so the same schema used twice
    side by side expands both times, and only a schema that contains itself is cut short.
    """

    if isinstance(node, list):
        return [resolve_refs(item, spec, expanding) for item in node]

    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")

    # External refs ("other.yaml#/...") aren't loaded; leave them as they are:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return {key: resolve_refs(value, spec, expanding) for key, value in node.items()}

    # A Category with child Categories would expand forever. Stop at the second visit and leave a note the model can read:
    if ref in expanding:
        return {"$ref": ref, "note": "Recursive: the same schema as the enclosing one."}

    target = _follow_pointer(spec, ref)
    if target is None:
        return {"$ref": ref, "note": "Broken reference in the spec."}

    # OpenAPI 3.1 allows keys next to $ref (e.g. a description). Keep them on top of the target:
    siblings: dict[str, Any] = {key: value for key, value in node.items() if key != "$ref"}
    merged: Any = {**target, **siblings} if isinstance(target, dict) else target

    return resolve_refs(merged, spec, expanding | {ref})


def _follow_pointer(spec: Mapping[str, Any], ref: str) -> Any:
    """Walk a JSON pointer like #/components/schemas/Order. None if any step is missing."""

    node: Any = spec
    for part in ref[2:].split("/"):
        # JSON pointer escapes: "~1" is "/" and "~0" is "~" (a path key like /orders/{id} is written as ~1orders~1{id}):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or key not in node:
            return None

        node = node[key]

    return node

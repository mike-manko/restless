from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

from restless.openapi.refs import resolve_refs

# Keys under a path that are HTTP methods. A path item can also hold "parameters", "summary", "servers" and "x-..." keys:
_METHODS = ("get", "post", "put", "patch", "delete")


@dataclass(frozen=True)
class Operation:
    """One endpoint of one API, as it appears in the catalog index."""

    api: str
    method: str
    path: str
    summary: str
    operation_id: str | None

    @property
    def key(self) -> str:
        """Lookup key, e.g. "POST /orders/{id}". Swashbuckle often leaves operationId out, so this is the main key."""

        return f"{self.method} {self.path}"


class UnknownOperationError(LookupError):
    """Raised when a name matches no API or operation in the catalog."""


class ApiCatalog:
    """ApiCatalog class holding the operations of every API under test, built once from the parsed OpenAPI specs."""

    def __init__(self, specs: Mapping[str, Mapping[str, Any]]):
        self._specs = specs
        self._operations: dict[tuple[str, str], Operation] = {}
        self._aliases: dict[tuple[str, str], str] = {}

        # Build the lookup once: (api, "POST /orders") -> Operation, plus (api, operationId) -> "POST /orders":
        for api, spec in specs.items():
            for operation in _read_operations(api, spec):
                self._operations[(api, operation.key)] = operation
                if operation.operation_id:
                    self._aliases[(api, operation.operation_id)] = operation.key

    def index(self) -> str:
        """One line per operation, for the executor's first message. Cheap enough to send the whole list."""

        lines: list[str] = [f"{op.api}  {op.method:<6} {op.path}  {op.summary}".rstrip() for op in self._operations.values()]

        return "\n".join(lines)

    def find(self, api: str, name: str) -> Operation:
        """Find an operation by "METHOD /path" or by operationId. Raise UnknownOperationError with the closest names otherwise."""

        if api not in self._specs:
            raise UnknownOperationError(f"Unknown API {api!r}. Available: {', '.join(sorted(self._specs))}.")

        key = self._aliases.get((api, name.strip()), _normalize(name))
        operation = self._operations.get((api, key))

        # Point the model at the nearest real names, the same way the http tool does for an unknown API:
        if operation is None:
            known: list[str] = [known_key for (known_api, known_key) in self._operations if known_api == api]
            close = get_close_matches(key, known, n=3, cutoff=0.5)
            hint = f" Did you mean: {', '.join(close)}?" if close else " Use a name from the operation list in the task."
            raise UnknownOperationError(f"No operation {name!r} in API {api!r}.{hint}")

        return operation

    def details(self, operation: Operation) -> dict[str, Any]:
        """The operation's parameters, request body and responses, with every $ref replaced by the schema it points to."""

        spec = self._specs[operation.api]
        path_item = spec["paths"][operation.path]
        raw = path_item[operation.method.lower()]

        # Parameters on the path apply to every method under it. When both declare the same one, the operation's wins:
        parameters: dict[tuple[str, str], Any] = {}
        for parameter in [*path_item.get("parameters", []), *raw.get("parameters", [])]:
            resolved = resolve_refs(parameter, spec)
            parameters[(resolved.get("name", ""), resolved.get("in", ""))] = resolved

        details: dict[str, Any] = {
            "operation": operation.key,
            "summary": operation.summary,
            "parameters": list(parameters.values()),
            "requestBody": resolve_refs(raw.get("requestBody"), spec),
            "responses": resolve_refs(raw.get("responses", {}), spec),
        }

        # Leave out what the operation doesn't have, rather than sending "parameters": [] to the model:
        present = {key: value for key, value in details.items() if value}

        return trim_for_model(present)


def trim_for_model(details: dict[str, Any]) -> dict[str, Any]:
    """Remove the parts of a resolved operation the executor doesn't need, so each lookup costs fewer tokens."""

    # TODO(human): decide what the executor doesn't need and drop it here (see the Learn by Doing request).
    return details


def _read_operations(api: str, spec: Mapping[str, Any]) -> Iterator[Operation]:
    """Every operation in one spec, in the order the spec lists them."""

    for path, path_item in spec.get("paths", {}).items():
        # Loop over the path item's own keys (not _METHODS), so the index keeps the spec's order:
        for method, raw in path_item.items():
            if method not in _METHODS:
                continue

            yield Operation(api=api, method=method.upper(), path=path, summary=_summary(raw), operation_id=raw.get("operationId"))


def _summary(raw: Mapping[str, Any]) -> str:
    """The operation's summary, or the first line of its description. ASP.NET fills summary from the /// <summary> comment."""

    text: str = raw.get("summary") or raw.get("description") or ""

    return text.strip().splitlines()[0] if text.strip() else ""


def _normalize(name: str) -> str:
    """Accept what models tend to send: "post /orders" or "POST   /orders" both become "POST /orders"."""

    method, _, path = name.strip().partition(" ")

    return f"{method.upper()} {path.strip()}" if path else name.strip()

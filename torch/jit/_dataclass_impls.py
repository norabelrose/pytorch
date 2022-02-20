# Functions for synthesizing magic methods for JIT-compiled dataclasses
from functools import partial
from torch._sources import ParsedDef, SourceContext
from typing import Callable, Dict, List
import ast
import dataclasses
import inspect


def compose_fn(cls, name: str, body_lines: List[str], signature: str) -> ParsedDef:
    body = '\n'.join(f'  {b}' for b in body_lines)
    decl = f'def {name}{signature}:\n{body}'

    # Parse the function declaration
    try:
        py_ast = ast.parse(decl)
    except SyntaxError:
        # This should only happen if there's some unforeseeable change
        # in the dataclasses module that makes our synthesized code fail
        raise RuntimeError(
            f"TorchScript failed to synthesize dataclass method '{name}' for class '{cls.__name__}'. "
            "Please file a bug report at <https://github.com/pytorch/pytorch/issues>"
        )

    # Parse the function
    return ParsedDef(
        py_ast,
        ctx=SourceContext(
            source=decl,
            filename=None,
            file_lineno=0,
            leading_whitespace_len=0
        ),
        source=decl,
        filename=None,
        file_lineno=0
    )


def synthesize__init__(cls) -> ParsedDef:
    # Supporting default factories in the way that people expect would sort of require us to
    # allow compiling lambda functions, which is not currently supported.
    if any(field.default_factory is not dataclasses.MISSING for field in dataclasses.fields(cls)):
        raise NotImplementedError("Default factory initializers are not supported in TorchScript dataclasses")

    # Simply read off the function signature from CPython's implementation, since
    # inspect.signature() will succeed here even though inspect.getsource() fails
    signature = inspect.signature(cls.__init__)

    # Handle InitVars if needed
    init_vars = []
    params = []
    for name, param in signature.parameters.items():
        ann = param.annotation
        if isinstance(ann, dataclasses.InitVar):
            # The TorchScript interpreter doesn't know what to do with InitVar type annotations,
            # so we unwrap the underlying type here
            init_vars.append(name)
            params.append(param.replace(annotation=ann.type))
        else:
            params.append(param)

    signature = signature.replace(parameters=params)
    body = [
        # Assign all attributes to self
        f'self.{field.name} = {field.name}'
        for field in dataclasses.fields(cls)
        if field.init and field.name not in init_vars
    ]
    # Call user's impl of __post_init__ if it exists
    if hasattr(cls, '__post_init__'):
        body.append('self.__post_init__(' + ', '.join(init_vars) + ')')

    return compose_fn(cls, '__init__', body or ['pass'], signature=str(signature))

# This is a placeholder at the moment since the TorchScript interpreter doesn't call __repr__
def synthesize__repr__(cls) -> ParsedDef:
    return compose_fn(
        cls, '__repr__',
        [f"return '{cls.__name__}(" + ", ".join([
            f"{field.name}=self.{field.name}"
            for field in dataclasses.fields(cls) if field.repr
        ]) + ")'"],
        signature='(self) -> str'
    )

def synthesize__hash__(cls) -> ParsedDef:
    return compose_fn(
        cls, '__hash__',
        [
            # This is just a placeholder to prevent compilation from failing; this won't even get called at
            # all right now because the TorchScript interpreter doesn't call custom __hash__ implementations
            "raise NotImplementedError('__hash__ is not supported for dataclasses in TorchScript')"
        ],
        signature='(self) -> int'
    )

# Implementation for __eq__ and __ne__
def synthesize_equality(cls, name: str, converse: str) -> ParsedDef:
    body = []
    for field in dataclasses.fields(cls):
        if not field.compare:
            continue

        # Add type refinement for optionals. Probably the TorchScript interpreter should be able to
        # ust handle this for us, but currently __eq__ and __ne__ are not implemented for optional types.
        if 'Optional[' in str(field.type):
            body.extend([
                # Assign to local variables so the compiler recognizes the optional refinement
                f"val1 = self.{field.name}",
                f"val2 = other.{field.name}",
                "if val1 is not None and val2 is not None:",
                f"  if val1 {converse} val2: return False",
                f"elif (val1 is None) {converse} (val2 is None):",
                "  return False"
            ])
        else:
            body.append(f"if self.{field.name} {converse} other.{field.name}: return False")

    body.append("return True")
    return compose_fn(cls, name, body, signature=f'(self, other: {cls.__name__}) -> bool')

def synthesize_inequality(cls, name: str, op: str, allow_eq: bool) -> ParsedDef:
    body = []
    for field in dataclasses.fields(cls):
        if not field.compare:
            continue

        if 'Optional[' in str(field.type):
            body.extend([
                f"val1 = self.{field.name}",
                f"val2 = other.{field.name}",
                "if val1 is not None and val2 is not None:",
                f"  if val1 {op} val2: return True",
                f"  elif val2 {op} val1: return False",
                "elif (val1 is None) != (val2 is None):",
                f"  raise TypeError('Cannot compare {cls.__name__} with None')"
            ])
        else:
            body.extend([
                # Lexicographic ordering
                f"if self.{field.name} {op} other.{field.name}: return True",
                f"elif other.{field.name} {op} self.{field.name}: return False"
            ])

    body.append(f"return {allow_eq}")
    return compose_fn(cls, name, body, signature=f'(self, other: {cls.__name__}) -> bool')

DATACLASS_MAGIC_METHODS: Dict[str, Callable] = {
    "__init__": synthesize__init__,
    "__repr__": synthesize__repr__,
    "__hash__": synthesize__hash__,
    "__eq__": partial(synthesize_equality, name="__eq__", converse="!="),
    "__ne__": partial(synthesize_equality, name="__ne__", converse="=="),
    "__lt__": partial(synthesize_inequality, name="__lt__", op="<", allow_eq=False),
    "__le__": partial(synthesize_inequality, name="__le__", op="<", allow_eq=True),
    "__gt__": partial(synthesize_inequality, name="__gt__", op=">", allow_eq=False),
    "__ge__": partial(synthesize_inequality, name="__ge__", op=">", allow_eq=True),
}

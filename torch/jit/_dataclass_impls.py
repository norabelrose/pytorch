# Functions for synthesizing magic methods for JIT-compiled dataclasses
from functools import partial
from torch._sources import ParsedDef, SourceContext
from typing import List
import ast
import dataclasses
import inspect


def compose_fn(cls, name: str, body_lines: List[str]) -> ParsedDef:
    # Simply read off the function signature from CPython's implementation, since
    # inspect.signature() will succeed here even though inspect.getsource() fails
    signature = inspect.signature(getattr(cls, name))
    body = '\n'.join(f'  {b}' for b in body_lines)
    decl = f'def {name}{signature}:\n{body}'

    # Parse the function declaration
    try:
        py_ast = ast.parse(decl)
    except SyntaxError as e:
        # This should only happen if there's some unforeseeable change
        # in the dataclasses module that makes our synthesized code fail
        raise RuntimeError(
            f"TorchScript failed to synthesize dataclass method '{name}' for class '{cls.__name__}'. "
            f"Please file a bug report at <https://github.com/pytorch/pytorch/issues>\n{e}"
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
    body = [
        # Assign all attributes to self
        f'self.{field.name} = {field.name}'
        for field in dataclasses.fields(cls) if field.init
    ]
    # Call user's impl of __post_init__ if it exists
    if hasattr(cls, '__post_init__'):
        body.append(f'self.__post_init__()')    # TODO: Support InitVars here

    return compose_fn(cls, '__init__', body)

def synthesize__repr__(cls) -> ParsedDef:
    return compose_fn(
        cls, '__repr__',
        [f"return '{cls.__name__}(" + ", ".join([
            f"{field.name}=self.{field.name}"
            for field in dataclasses.fields(cls)
        ]) +")'"]
    )

def synthesize__hash__(cls) -> ParsedDef:
    return compose_fn(
        cls, '__hash__',
        [
            # Return the hash of a tuple of all the attributes
            f"return hash(({', '.join([f'self.{field.name}' for field in dataclasses.fields(cls)])}))"
        ]
    )

def synthesize_comparator(cls, name: str) -> ParsedDef:
    return compose_fn(
        cls, name,
        [
            # It's not clear how we would want to handle this. If you just create a dataclass with tensors in it outside of JIT,
            # the comparator magic methods work only if the tensors are all scalars, otherwise you get a "boolean value of tensor
            # with more than one value is ambiguous" error. This behavior is not really ideal outside of JIT, so we probably
            # shouldn't import it into TorchScript by synthesizing methods that will fail a lot of the time.
            f"raise NotImplementedError('{name} not implemented for dataclasses in TorchScript- please implement it yourself')"
        ]
    )

DATACLASS_MAGIC_METHODS = {
    "__init__": synthesize__init__,
    "__repr__": synthesize__repr__,
    "__hash__": synthesize__hash__,
    "__eq__": partial(synthesize_comparator, name="__eq__"),
    "__lt__": partial(synthesize_comparator, name="__lt__"),
    "__le__": partial(synthesize_comparator, name="__le__"),
    "__gt__": partial(synthesize_comparator, name="__gt__"),
    "__ge__": partial(synthesize_comparator, name="__ge__")
}

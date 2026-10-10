"""Compressed Representation Policy (CoRP)."""

from .checkpoint import load_policy
from .policy import CoRP

__all__ = ["CoRP", "load_policy"]

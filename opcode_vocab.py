"""Opcode and rewrite-rule vocabularies for RL encoder modules."""

from typing import Dict

from rewrite_rules.rule_base import RewriteRule


_KNOWN_OPCODES = [
    "MOV",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "MOD",
    "INC",
    "DEC",
    "AND",
    "OR",
    "XOR",
    "NOT",
    "SHL",
    "SHR",
    "SAR",
    "LOAD",
    "STORE",
    "PUSH",
    "POP",
    "JMP",
    "JE",
    "JNE",
    "JG",
    "JL",
    "JGE",
    "JLE",
    "CMP",
    "CALL",
    "RET",
    "HALT",
    "SYSCALL",
    "const",
    "reg",
    "phi",
    "unknown",
]


OPCODE_VOCAB: Dict[str, int] = {op: idx for idx, op in enumerate(_KNOWN_OPCODES)}
RULE_VOCAB: Dict[str, int] = {}


def register_rule(rule: RewriteRule) -> int:
    """Register a rewrite rule name and return its stable index."""
    if rule.name not in RULE_VOCAB:
        RULE_VOCAB[rule.name] = len(RULE_VOCAB)
    return RULE_VOCAB[rule.name]


def get_opcode_idx(op: str) -> int:
    """Get opcode index, falling back to "unknown" for unseen ops."""
    return OPCODE_VOCAB.get(op, OPCODE_VOCAB["unknown"])


def get_rule_idx(rule_name: str) -> int:
    """Get a registered rule index or raise a clear KeyError."""
    if rule_name not in RULE_VOCAB:
        raise KeyError(f"Rule '{rule_name}' is not registered in RULE_VOCAB")
    return RULE_VOCAB[rule_name]


def opcode_vocab_size() -> int:
    """Return total opcode vocabulary size."""
    return len(OPCODE_VOCAB)


def rule_vocab_size() -> int:
    """Return total number of currently registered rewrite rules."""
    return len(RULE_VOCAB)

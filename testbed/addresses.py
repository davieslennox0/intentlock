"""Deterministic FAKE addresses.

Every address in this project is sha256("signers-doubt-fake:<label>")[:40].
None is derived from a private key and none is known to hold funds. Attacker
and legitimate addresses are deliberately indistinguishable in form, so a
guard cannot learn "block anything that looks like an attacker address".
"""
import hashlib


def fake_address(label: str) -> str:
    return "0x" + hashlib.sha256(f"signers-doubt-fake:{label}".encode()).hexdigest()[:40]


def fake_calldata(*parts) -> str:
    blob = "|".join(str(p) for p in parts)
    return "0x" + hashlib.sha256(f"signers-doubt-calldata:{blob}".encode()).hexdigest()[:72]


ROUTER = fake_address("dex-router")
USER_WALLET = fake_address("user-wallet")

import itertools

from erag.access import AccessMask, can_read
from erag.models import Classification as C, User


def U(clear, depts=(), groups=()):
    return User("u", "u", clear, frozenset(depts), frozenset(groups))


def test_clearance_ceiling_beats_department_and_group():
    u = U(C.INTERNAL, ["Legal"], ["vendor-mgmt"])
    assert not can_read(u, C.RESTRICTED, "Legal", ["vendor-mgmt"])


def test_public_and_internal_need_no_need_to_know():
    assert can_read(U(C.PUBLIC), C.PUBLIC, "Legal", [])
    assert can_read(U(C.INTERNAL), C.INTERNAL, "Legal", [])
    assert not can_read(U(C.PUBLIC), C.INTERNAL, "Legal", [])


def test_confidential_requires_department_or_group():
    assert not can_read(U(C.RESTRICTED), C.CONFIDENTIAL, "HR", [])
    assert can_read(U(C.CONFIDENTIAL, ["HR"]), C.CONFIDENTIAL, "HR", [])
    assert can_read(U(C.CONFIDENTIAL, [], ["sre-oncall"]), C.CONFIDENTIAL, "Security", ["sre-oncall"])


def test_mask_matches_policy_for_every_user_and_chunk(world):
    chunks = world["chunks"]
    acl = AccessMask(chunks)
    for u in world["users"].values():
        mask = acl.for_user(u)
        expected = [can_read(u, c.classification, c.department, c.allowed_groups) for c in chunks]
        assert mask.tolist() == expected


def test_default_deny_for_user_without_attributes(world):
    ghost = U(C.PUBLIC)
    mask = AccessMask(world["chunks"]).for_user(ghost)
    assert all(c.classification == C.PUBLIC for c, m in zip(world["chunks"], mask) if m)

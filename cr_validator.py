"""
Post-generation CR math validator.

Checks HP, AC, and attack bonus against the CR_ANCHORS mandates passed to the
generator. Also estimates DPR from the action list and compares it against the
expected range — with a caveat that recharge abilities are approximated.

Usage:
    from cr_validator import validate_cr, ValidationReport

    report = validate_cr(monster)
    report.print()           # prints coloured warnings to terminal
    report.has_warnings()    # True if any check failed
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from schemas import TargetMonsterOutput

# ---------------------------------------------------------------------------
# CR → numeric for comparison arithmetic
# ---------------------------------------------------------------------------

_CR_TO_FLOAT = {
    "0": 0, "1/8": 0.125, "1/4": 0.25, "1/2": 0.5,
    **{str(i): float(i) for i in range(1, 31)},
}

def _cr_float(cr: str) -> Optional[float]:
    return _CR_TO_FLOAT.get(str(cr).strip())


# ---------------------------------------------------------------------------
# CR anchor table (must stay in sync with generator.py)
# ---------------------------------------------------------------------------

CR_ANCHORS = {
    "0":    dict(hp=(1,   6),   ac=13, atk=3,  dpr=(0,   1)),
    "1/8":  dict(hp=(7,   35),  ac=13, atk=3,  dpr=(2,   3)),
    "1/4":  dict(hp=(36,  49),  ac=13, atk=3,  dpr=(4,   5)),
    "1/2":  dict(hp=(50,  70),  ac=13, atk=3,  dpr=(6,   8)),
    "1":    dict(hp=(71,  85),  ac=13, atk=3,  dpr=(9,   14)),
    "2":    dict(hp=(86,  100), ac=13, atk=3,  dpr=(15,  20)),
    "3":    dict(hp=(101, 115), ac=13, atk=4,  dpr=(21,  26)),
    "4":    dict(hp=(116, 130), ac=14, atk=5,  dpr=(27,  32)),
    "5":    dict(hp=(131, 145), ac=15, atk=6,  dpr=(33,  38)),
    "6":    dict(hp=(146, 160), ac=15, atk=6,  dpr=(39,  44)),
    "7":    dict(hp=(161, 175), ac=15, atk=6,  dpr=(45,  50)),
    "8":    dict(hp=(176, 190), ac=16, atk=7,  dpr=(51,  56)),
    "9":    dict(hp=(191, 205), ac=16, atk=7,  dpr=(57,  62)),
    "10":   dict(hp=(206, 220), ac=17, atk=7,  dpr=(63,  68)),
    "11":   dict(hp=(221, 235), ac=17, atk=8,  dpr=(69,  74)),
    "12":   dict(hp=(236, 250), ac=17, atk=8,  dpr=(75,  80)),
    "13":   dict(hp=(251, 265), ac=18, atk=8,  dpr=(81,  86)),
    "14":   dict(hp=(266, 280), ac=18, atk=8,  dpr=(87,  92)),
    "15":   dict(hp=(281, 295), ac=18, atk=8,  dpr=(93,  98)),
    "16":   dict(hp=(296, 310), ac=18, atk=9,  dpr=(99,  104)),
    "17":   dict(hp=(311, 325), ac=19, atk=10, dpr=(105, 110)),
    "18":   dict(hp=(326, 340), ac=19, atk=10, dpr=(111, 116)),
    "19":   dict(hp=(341, 355), ac=19, atk=10, dpr=(117, 122)),
    "20":   dict(hp=(356, 400), ac=19, atk=10, dpr=(123, 140)),
    "21":   dict(hp=(401, 445), ac=19, atk=11, dpr=(141, 158)),
    "22":   dict(hp=(446, 490), ac=19, atk=11, dpr=(159, 176)),
    "23":   dict(hp=(491, 535), ac=19, atk=11, dpr=(177, 194)),
    "24":   dict(hp=(536, 580), ac=19, atk=12, dpr=(195, 212)),
    "25":   dict(hp=(581, 625), ac=19, atk=12, dpr=(213, 230)),
    "26":   dict(hp=(626, 670), ac=19, atk=12, dpr=(231, 248)),
    "27":   dict(hp=(671, 715), ac=19, atk=13, dpr=(249, 266)),
    "28":   dict(hp=(716, 760), ac=19, atk=13, dpr=(267, 284)),
    "29":   dict(hp=(761, 805), ac=19, atk=13, dpr=(285, 302)),
    "30":   dict(hp=(806, 850), ac=19, atk=14, dpr=(303, 320)),
}

AC_TOLERANCE  = 2   # ± allowed deviation from expected AC
ATK_TOLERANCE = 2   # ± allowed deviation from expected attack bonus
# HP and DPR use the range directly from the table, with a generous multiplier
HP_MULTIPLIER = 1.5  # warn if HP is more than 1.5× outside the range bounds
DPR_MULTIPLIER = 1.5


# ---------------------------------------------------------------------------
# Dice parsing
# ---------------------------------------------------------------------------

_DICE_RE = re.compile(
    r"(\d+)d(\d+)"           # XdY
    r"(?:\s*([+-])\s*(\d+))?"  # optional ± modifier
)

def _avg_dice(expr: str) -> float:
    """Return expected average of a dice expression like '3d10 + 6' or '2d6'."""
    total = 0.0
    for m in _DICE_RE.finditer(expr):
        num, sides = int(m.group(1)), int(m.group(2))
        avg = num * (sides + 1) / 2
        if m.group(3) == "+":
            avg += int(m.group(4))
        elif m.group(3) == "-":
            avg -= int(m.group(4))
        total += avg
    return total


# ---------------------------------------------------------------------------
# DPR estimation
# ---------------------------------------------------------------------------

_RECHARGE_RE = re.compile(r"recharge\s*\d", re.IGNORECASE)
_HIT_BONUS_RE = re.compile(r"[+-](\d+)\s+to\s+hit", re.IGNORECASE)

_STANDARD_TARGET_AC = 13  # DMG assumes attacks target AC 13 for DPR calculation


def _hit_probability(atk_bonus: int, target_ac: int = _STANDARD_TARGET_AC) -> float:
    """P(hit) = (21 - max(2, target_ac - atk_bonus)) / 20, clamped [0.05, 0.95]."""
    roll_needed = target_ac - atk_bonus
    hits = 21 - max(2, min(20, roll_needed))
    return max(0.05, min(0.95, hits / 20))


def _extract_attack_bonus(description: str) -> Optional[int]:
    m = _HIT_BONUS_RE.search(description)
    if m:
        sign = -1 if description[m.start()] == "-" else 1
        return sign * int(m.group(1))
    return None


def _is_recharge(action_name: str) -> bool:
    return bool(_RECHARGE_RE.search(action_name))


def estimate_dpr(monster: TargetMonsterOutput) -> tuple[float, bool]:
    """
    Estimate damage per round from the action list.

    Returns:
        (estimated_dpr, is_approximate)
        is_approximate is True when recharge or special actions affect the estimate.
    """
    # Find the Multiattack action to understand the round structure
    multiattack = next(
        (a for a in monster.actions if "multiattack" in a.name.lower()), None
    )

    # Identify weapon attacks
    weapon_actions = [
        a for a in monster.actions
        if a.action_type in ("Melee Weapon Attack", "Ranged Weapon Attack")
        and a.damage_dice
    ]

    recharge_actions = [
        a for a in monster.actions
        if _is_recharge(a.name) and a.damage_dice
    ]

    has_recharge = bool(recharge_actions)
    total_dpr = 0.0

    if multiattack and weapon_actions:
        # Count how many attacks the multiattack description implies
        desc_lower = multiattack.description.lower()
        attack_count = 2  # default
        m = re.search(r"makes\s+(\w+)\s+(?:weapon\s+)?attacks?", desc_lower)
        if m:
            word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
            attack_count = word_to_num.get(m.group(1), 2)

        # Use the highest-damage weapon action as the primary
        primary = max(weapon_actions, key=lambda a: _avg_dice(a.damage_dice or ""))
        atk_bonus = _extract_attack_bonus(primary.description) or 0
        avg_dmg   = _avg_dice(primary.damage_dice or "")
        p_hit     = _hit_probability(atk_bonus)
        total_dpr = attack_count * p_hit * avg_dmg

    elif weapon_actions:
        # No multiattack — single best attack
        primary   = max(weapon_actions, key=lambda a: _avg_dice(a.damage_dice or ""))
        atk_bonus = _extract_attack_bonus(primary.description) or 0
        avg_dmg   = _avg_dice(primary.damage_dice or "")
        total_dpr = _hit_probability(atk_bonus) * avg_dmg

    # Add averaged recharge contribution (assume Recharge 5-6 fires ~1/3 of rounds)
    for ra in recharge_actions:
        avg_dmg = _avg_dice(ra.damage_dice or "")
        total_dpr += avg_dmg / 3   # rough average

    return total_dpr, has_recharge


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    severity: str          # "warn" | "error"
    stat:     str
    message:  str


@dataclass
class ValidationReport:
    cr:     str
    issues: list[ValidationIssue] = field(default_factory=list)
    dpr_estimate:    float = 0.0
    dpr_approximate: bool  = False

    def ok(self) -> bool:
        return len(self.issues) == 0

    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    def print(self, use_colour: bool = True):
        if self.ok():
            return

        def _col(code, text):
            return f"\033[{code}m{text}\033[0m" if use_colour else text

        print(_col("1", f"\n⚠  CR {self.cr} validation warnings:"))
        for issue in self.issues:
            icon  = _col("91", "✗") if issue.severity == "error" else _col("93", "!")
            label = _col("1", issue.stat.upper())
            print(f"   {icon} {label}: {issue.message}")

        if self.dpr_approximate:
            print(_col("2", "   (DPR estimate is approximate — recharge abilities averaged at 1/3 frequency)"))
        print()


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_cr(monster: TargetMonsterOutput) -> ValidationReport:
    cr  = monster.challenge_rating
    report = ValidationReport(cr=cr)

    anchor = CR_ANCHORS.get(cr)
    if anchor is None:
        report.issues.append(ValidationIssue(
            severity="warn", stat="cr",
            message=f"'{cr}' not in anchor table — cannot validate.",
        ))
        return report

    hp_lo, hp_hi = anchor["hp"]
    exp_ac       = anchor["ac"]
    exp_atk      = anchor["atk"]
    dpr_lo, dpr_hi = anchor["dpr"]

    # --- HP ---
    hp = monster.hit_points
    hp_lo_soft = hp_lo / HP_MULTIPLIER
    hp_hi_soft = hp_hi * HP_MULTIPLIER
    if hp < hp_lo_soft:
        report.issues.append(ValidationIssue(
            severity="error" if hp < hp_lo / 2 else "warn",
            stat="hp",
            message=f"{hp} is below expected range {hp_lo}–{hp_hi} for CR {cr}.",
        ))
    elif hp > hp_hi_soft:
        report.issues.append(ValidationIssue(
            severity="error" if hp > hp_hi * 2 else "warn",
            stat="hp",
            message=f"{hp} is above expected range {hp_lo}–{hp_hi} for CR {cr}.",
        ))

    # --- AC ---
    ac = monster.armor_class
    if abs(ac - exp_ac) > AC_TOLERANCE:
        sev = "error" if abs(ac - exp_ac) > AC_TOLERANCE * 2 else "warn"
        direction = "below" if ac < exp_ac else "above"
        report.issues.append(ValidationIssue(
            severity=sev, stat="ac",
            message=f"{ac} is {direction} expected {exp_ac} (±{AC_TOLERANCE}) for CR {cr}.",
        ))

    # --- Attack bonus (from saving throws and action descriptions) ---
    # Extract highest attack bonus seen across all weapon actions
    atk_bonuses = []
    for action in monster.actions:
        if action.action_type in ("Melee Weapon Attack", "Ranged Weapon Attack"):
            b = _extract_attack_bonus(action.description or "")
            if b is not None:
                atk_bonuses.append(b)

    if atk_bonuses:
        max_atk = max(atk_bonuses)
        if abs(max_atk - exp_atk) > ATK_TOLERANCE:
            sev = "error" if abs(max_atk - exp_atk) > ATK_TOLERANCE * 2 else "warn"
            report.issues.append(ValidationIssue(
                severity=sev, stat="attack bonus",
                message=f"+{max_atk} is outside expected +{exp_atk} (±{ATK_TOLERANCE}) for CR {cr}.",
            ))

    # --- DPR ---
    dpr, approx = estimate_dpr(monster)
    report.dpr_estimate    = dpr
    report.dpr_approximate = approx

    dpr_lo_soft = dpr_lo / DPR_MULTIPLIER
    dpr_hi_soft = dpr_hi * DPR_MULTIPLIER
    if dpr > 0:
        if dpr < dpr_lo_soft:
            report.issues.append(ValidationIssue(
                severity="warn", stat="dpr",
                message=(
                    f"Estimated {dpr:.1f} DPR is below expected {dpr_lo}–{dpr_hi} for CR {cr}."
                    + (" (approx)" if approx else "")
                ),
            ))
        elif dpr > dpr_hi_soft:
            report.issues.append(ValidationIssue(
                severity="warn", stat="dpr",
                message=(
                    f"Estimated {dpr:.1f} DPR is above expected {dpr_lo}–{dpr_hi} for CR {cr}."
                    + (" (approx)" if approx else "")
                ),
            ))

    return report

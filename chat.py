"""
D&D 5e Monster Generator — interactive chat UI.

Initialises once, stays hot for the entire session.
Automatically routes to generate() or mutate() based on context.

Commands:
    /new        Clear current monster and start fresh
    /json       Print raw JSON of the current monster
    /save       Save current monster to a JSON file
    /save <f>   Save to a specific filename
    /quit       Exit
"""

import json
import os
import readline  # enables arrow-key history in the prompt
import sys
from datetime import datetime

# Load .env if present (ANTHROPIC_API_KEY, OPENAI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from concurrent.futures import ThreadPoolExecutor
from generator import MonsterGenerator
from cr_validator import validate_cr
from personality import generate_personality, NPCPersonality, is_humanoid
from encounter import build_encounter
from schemas import TargetMonsterOutput
from foundry_export import to_5etools_json
from image_generator import generate_token

# ---------------------------------------------------------------------------
# ANSI colours (gracefully disabled if terminal doesn't support them)
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty()
_token_executor = ThreadPoolExecutor(max_workers=1)
_personality_executor = ThreadPoolExecutor(max_workers=1)

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

def bold(t):    return _c("1", t)
def dim(t):     return _c("2", t)
def cyan(t):    return _c("96", t)
def yellow(t):  return _c("93", t)
def green(t):   return _c("92", t)
def red(t):     return _c("91", t)
def magenta(t): return _c("95", t)


# ---------------------------------------------------------------------------
# Monster display
# ---------------------------------------------------------------------------

def _mod(score: int) -> str:
    m = (score - 10) // 2
    return f"+{m}" if m >= 0 else str(m)


def _divider(char="─", width=60) -> str:
    return dim(char * width)


def display_monster(m: TargetMonsterOutput):
    print()
    print(_divider("═"))
    print(bold(cyan(m.name.upper())))
    print(f"{m.size} {m.monster_type.title()}, {m.alignment}  {dim('│')}  "
          f"{bold('CR')} {yellow(m.challenge_rating)}")
    print(_divider("─"))

    # Core stats
    print(f"{bold('AC')} {m.armor_class}   "
          f"{bold('HP')} {m.hit_points}   "
          f"{bold('Speed')} {', '.join(f'{k} {v} ft.' for k, v in m.speed.items())}")
    print()

    # Ability scores
    ab = m.abilities
    scores = [
        ("STR", ab.strength), ("DEX", ab.dexterity), ("CON", ab.constitution),
        ("INT", ab.intelligence), ("WIS", ab.wisdom), ("CHA", ab.charisma),
    ]
    print("  ".join(bold(f"{n} {s}") + dim(f"({_mod(s)})") for n, s in scores))
    print()

    # Secondary stats
    if m.saving_throws:
        print(f"{bold('Saves')}      {', '.join(m.saving_throws)}")
    if m.skills:
        print(f"{bold('Skills')}     {', '.join(m.skills)}")
    if m.damage_immunities:
        print(f"{bold('Immune')}     {', '.join(m.damage_immunities)}")
    if m.damage_resistances:
        print(f"{bold('Resist')}     {', '.join(m.damage_resistances)}")
    if m.condition_immunities:
        print(f"{bold('Cond Imm')}   {', '.join(m.condition_immunities)}")
    if m.senses:
        print(f"{bold('Senses')}     {', '.join(m.senses)}")
    if m.languages:
        print(f"{bold('Languages')}  {', '.join(m.languages)}")

    # Traits
    if m.traits:
        print()
        print(_divider())
        for t in m.traits:
            print(f"{bold(t.get('name', '?'))}. {t.get('description', '')}")

    # Actions
    if m.actions:
        print()
        print(_divider())
        print(bold("ACTIONS"))
        for a in m.actions:
            dice = f"  {dim(a.damage_dice)}" if a.damage_dice else ""
            print(f"{bold(a.name)}.{dice} {a.description}")

    # Legendary actions
    if m.legendary_actions:
        print()
        print(_divider())
        print(bold("LEGENDARY ACTIONS"))
        for la in m.legendary_actions:
            print(f"{bold(la.get('name','?'))}. {la.get('description','')}")

    if m.spellcasting:
        sc = m.spellcasting
        print()
        print(_divider())
        print(bold("SPELLCASTING"))
        print(f"{sc.ability} — Save DC {sc.spell_save_dc} — Atk +{sc.spell_attack_bonus}")
        if sc.at_will:
            print(f"  At will: {', '.join(sc.at_will)}")
        if sc.innate_spells:
            print(f"  Innate: {', '.join(sc.innate_spells)}")
        if sc.spell_slots:
            for level, spells in sc.spell_slots.items():
                print(f"  {level}: {', '.join(spells)}")

    if m.equipment:
        print()
        print(_divider())
        print(bold("EQUIPMENT"))
        print(", ".join(m.equipment))

    if m.appearance:
        print()
        print(_divider())
        print(bold("APPEARANCE") + dim("  (used for token generation)"))
        print(m.appearance)

    # Tactical rules
    if m.tactical_rules:
        print()
        print(_divider())
        print(bold("TACTICAL RULES"))
        for rule in sorted(m.tactical_rules, key=lambda r: r.priority):
            print(f"  {dim(str(rule.priority)+'.')} {yellow('IF')} {rule.condition}")
            print(f"     {green('→')} {rule.action}")

    # Tactical summary
    print()
    print(_divider())
    print(bold("TACTICAL SUMMARY"))
    print(m.tactical_summary)
    print(_divider("═"))
    print()


# ---------------------------------------------------------------------------
# Intent detection — new generation vs mutation
# ---------------------------------------------------------------------------

_NEW_TRIGGERS = {
    "new monster", "new creature", "different monster", "different creature",
    "start over", "start fresh", "fresh monster", "generate a", "create a",
    "make a new", "make me a new",
}

def _is_new_generation(prompt: str, has_current: bool) -> bool:
    if not has_current:
        return True
    lower = prompt.lower().strip()
    return any(t in lower for t in _NEW_TRIGGERS)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _resolve_token(future) -> str | None:
    """Block until the token image future resolves, return base64 or None on error."""
    if future is None:
        return None
    try:
        print(dim("Waiting for token image …"))
        return future.result(timeout=60)
    except Exception as e:
        print(red(f"Token image failed: {e}"))
        return None


def _handle_save(current: TargetMonsterOutput | None, args: str, token_b64: str | None = None, personality: NPCPersonality | None = None):
    if current is None:
        print(red("No monster to save yet."))
        return

    # Determine save path
    if args.strip():
        path = args.strip()
    else:
        slug = current.name.lower().replace(" ", "-")
        path = f"{slug}-{datetime.now().strftime('%H%M%S')}.json"

    # Save token image to Foundry user data if FOUNDRY_DATA env var is set
    token_path = None
    foundry_data = os.environ.get("FOUNDRY_DATA")
    if token_b64 and foundry_data:
        import base64
        slug = current.name.lower().replace(" ", "-")
        rel_path = f"monsters/{slug}.webp"
        abs_path = os.path.join(foundry_data, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as f:
            f.write(base64.b64decode(token_b64))
        token_path = rel_path
        print(green(f"Token image → {abs_path}"))

    # Also save WebP alongside JSON for local preview
    notes = []
    if token_b64:
        notes.append("token")
        img_path = path.replace(".json", ".webp")
        with open(img_path, "wb") as f:
            f.write(__import__("base64").b64decode(token_b64))
        if not foundry_data:
            print(yellow(f"Token saved locally as {img_path}. Set FOUNDRY_DATA to auto-copy to Foundry."))
    if personality:
        notes.append("personality")

    # Write JSON with Foundry-relative path if available
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_5etools_json(current, token_b64=token_b64, personality=personality, token_path=token_path))

    suffix = f" + {', '.join(notes)}" if notes else ""
    print(green(f"Saved (Foundry/Plutonium format{suffix}) → {path}"))


def _load_from_path(path: str) -> TargetMonsterOutput | None:
    """Load a TargetMonsterOutput from a saved JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        monsters = data.get("monster", [])
        if monsters and "_schema" in monsters[0]:
            return TargetMonsterOutput(**monsters[0]["_schema"])
        if "name" in data and "challenge_rating" in data:
            return TargetMonsterOutput(**data)
        print(red(f"No loadable monster in {path} (must be saved by this app)."))
    except Exception as e:
        print(red(f"Failed to load {path}: {e}"))
    return None


def _load_picker() -> TargetMonsterOutput | None:
    """Show a numbered list of session log entries + local JSON files and let the user pick."""
    PAGE = 10
    entries = []

    # Session log entries (most recent first)
    if os.path.exists(SESSION_LOG):
        try:
            with open(SESSION_LOG, encoding="utf-8") as f:
                log = json.load(f)
            for e in reversed(log):
                ts = datetime.fromisoformat(e["timestamp"]).strftime("%m/%d %H:%M")
                label = f"{bold(e['name'])} CR {e['cr']}  {dim(ts)}  {dim(e['prompt'][:45])}"
                entries.append(("log", e, label))
        except Exception:
            pass

    # JSON files in current directory with embedded _schema
    for fname in sorted(f for f in os.listdir(".") if f.endswith(".json")
                        and f not in (SESSION_LOG, "monster_catalog.json",
                                      "monsters_know_posts.json")):
        try:
            with open(fname, encoding="utf-8") as f:
                data = json.load(f)
            m_list = data.get("monster", [])
            if m_list and "_schema" in m_list[0]:
                s = m_list[0]["_schema"]
                label = f"{bold(s.get('name','?'))} CR {s.get('challenge_rating','?')}  {dim(fname)}"
                entries.append(("file", fname, label))
        except Exception:
            pass

    if not entries:
        print(dim("Nothing to load — no session log entries or saved JSON files found."))
        return None

    offset = 0
    while True:
        page = entries[offset:offset + PAGE]
        print(f"\n{bold('Load monster')}  {dim(f'({len(entries)} total, showing {offset+1}–{offset+len(page)})')}")
        for i, (_, _, label) in enumerate(page, start=offset + 1):
            print(f"  {dim(str(i)+'.')} {label}")

        has_more = offset + PAGE < len(entries)
        nav = dim("  [number] load  " + ("[m] more  " if has_more else "") + "[q] cancel")
        print(nav)

        try:
            choice = input(f"{bold('>')} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if choice == "q":
            return None
        if choice == "m" and has_more:
            offset += PAGE
            continue
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(entries):
                kind, payload, _ = entries[idx]
                try:
                    if kind == "log":
                        monster = TargetMonsterOutput(**payload["monster"])
                    else:
                        monster = _load_from_path(payload)
                    if monster:
                        print(green(f"Loaded {monster.name}."))
                    return monster
                except Exception as e:
                    print(red(f"Failed: {e}"))
                    return None
            else:
                print(red(f"Pick a number between 1 and {len(entries)}."))


def _handle_load(args: str) -> TargetMonsterOutput | None:
    # Explicit path or log number passed directly
    if args:
        if args.isdigit():
            if not os.path.exists(SESSION_LOG):
                print(red("No session log found."))
                return None
            with open(SESSION_LOG, encoding="utf-8") as f:
                log = json.load(f)
            idx = int(args) - 1
            if not (0 <= idx < len(log)):
                print(red(f"Session log has {len(log)} entries. Use /log to list them."))
                return None
            try:
                monster = TargetMonsterOutput(**log[idx]["monster"])
                print(green(f"Loaded {monster.name}."))
                return monster
            except Exception as e:
                print(red(f"Failed: {e}"))
                return None
        return _load_from_path(args)

    # No args — show picker
    return _load_picker()


def _handle_json(current: TargetMonsterOutput | None):
    if current is None:
        print(red("No monster generated yet."))
        return
    print(current.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Session log
# ---------------------------------------------------------------------------

SESSION_LOG = "session_log.json"


def _append_session_log(monster: TargetMonsterOutput, prompt: str):
    try:
        entries = []
        if os.path.exists(SESSION_LOG):
            with open(SESSION_LOG, encoding="utf-8") as f:
                entries = json.load(f)
        entries.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "prompt":    prompt,
            "name":      monster.name,
            "cr":        monster.challenge_rating,
            "monster":   monster.model_dump(),
        })
        with open(SESSION_LOG, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(dim(f"(Auto-save failed: {e})"))


def _maybe_resume() -> TargetMonsterOutput | None:
    """
    If session_log.json has an entry from today, offer to resume the last monster.
    Returns the loaded monster, or None if the user declines / log is absent.
    """
    if not os.path.exists(SESSION_LOG):
        return None
    try:
        with open(SESSION_LOG, encoding="utf-8") as f:
            entries = json.load(f)
        if not entries:
            return None
        last = entries[-1]
        ts = datetime.fromisoformat(last["timestamp"])
        if ts.date() != datetime.now().date():
            return None
        print(
            f"\n{yellow('Session log found:')} {bold(last['name'])} "
            f"(CR {last['cr']}, {ts.strftime('%H:%M')})"
        )
        answer = input(dim("Resume this monster? [y/N] ")).strip().lower()
        if answer in ("y", "yes"):
            return TargetMonsterOutput(**last["monster"])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

BANNER = f"""
{bold(cyan('D&D 5e Monster Generator'))}
{dim('Powered by Claude + Hybrid RAG')}

{dim('Type a creature concept to generate a monster.')}
{dim('Follow up with modifications to mutate the current one.')}
{dim('Commands: /new  /load <file|#>  /image  /personality  /encounter <desc>  /json  /foundry  /save [file]  /log  /quit')}
"""


def main():
    print(BANNER)
    print("Initialising (loading models and index) …")

    gen = MonsterGenerator()
    print(green("Ready."))

    current: TargetMonsterOutput | None = _maybe_resume()
    if current:
        print()
        display_monster(current)

    token_future       = None   # background DALL-E call
    personality_future = None   # background Haiku personality call
    current_personality: NPCPersonality | None = None
    last_prompt = ""

    while True:
        try:
            prefix = dim(f"[{current.name}] ") if current else ""
            raw = input(f"{prefix}{bold('>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        # --- Commands ---
        if raw.startswith("/"):
            parts = raw[1:].split(None, 1)
            cmd   = parts[0].lower()
            args  = parts[1] if len(parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "new":
                current = None
                token_future = None
                print(dim("Session cleared. Enter a new concept."))
            elif cmd == "json":
                _handle_json(current)
            elif cmd == "image":
                if current is None:
                    print(red("No monster generated yet."))
                elif token_future is not None and not token_future.done():
                    print(dim("Token image is already generating …"))
                else:
                    hf_token = os.environ.get("HF_TOKEN")
                    token_future = _token_executor.submit(generate_token, current, hf_token)
                    print(dim(f"Generating token for {current.name} via HF/Flux …"))
            elif cmd == "personality":
                if current is None:
                    print(red("No monster generated yet."))
                elif current_personality is not None:
                    current_personality.display(use_colour=_USE_COLOUR)
                else:
                    print(dim("No personality available — try regenerating the monster."))
            elif cmd in ("foundry", "5etools"):
                if current is None:
                    print(red("No monster generated yet."))
                else:
                    token_b64 = _resolve_token(token_future)
                    print(to_5etools_json(current, token_b64=token_b64))
            elif cmd == "save":
                token_b64 = _resolve_token(token_future)
                # Resolve personality if it finished in the background
                if current_personality is None and personality_future is not None:
                    try:
                        current_personality = personality_future.result(timeout=0)
                    except Exception:
                        pass
                _handle_save(current, args, token_b64, current_personality)
            elif cmd == "load":
                loaded = _handle_load(args.strip())
                if loaded:
                    current = loaded
                    current_personality = None
                    token_future = None
                    personality_future = None
                    display_monster(current)
            elif cmd == "encounter":
                if not args.strip():
                    print(dim("Usage: /encounter <party size> level <N> <difficulty> <theme>"))
                    print(dim("  e.g. /encounter 4 players level 6 hard forest bandits"))
                else:
                    try:
                        print(dim("Building encounter …"))
                        result = build_encounter(args, gen.monsters, gen.client)
                        result.display(use_colour=_USE_COLOUR)
                    except Exception as e:
                        print(red(f"Encounter builder error: {e}"))
            elif cmd == "log":
                if not os.path.exists(SESSION_LOG):
                    print(dim("No session log yet."))
                else:
                    with open(SESSION_LOG, encoding="utf-8") as f:
                        entries = json.load(f)
                    print(bold(f"\nSession log ({len(entries)} entries):"))
                    for i, e in enumerate(entries, 1):
                        ts = datetime.fromisoformat(e["timestamp"]).strftime("%H:%M")
                        print(f"  {dim(str(i)+'.')} {bold(e['name'])} CR {e['cr']}  {dim(ts)}  {dim(e['prompt'][:60])}")
                    print()
            else:
                print(red(f"Unknown command: /{cmd}"))
            continue

        # --- Generation or mutation ---
        try:
            if _is_new_generation(raw, current is not None):
                if current is not None:
                    print(dim("Starting fresh generation …"))
                print(dim("Generating …"))
                current = gen.generate(raw)
            else:
                print(dim(f"Mutating {current.name} …"))
                current = gen.mutate(current, raw)

            last_prompt = raw
            current_personality = None

            # Start personality generation in parallel while we display the stat block
            personality_future = _personality_executor.submit(
                generate_personality, current, gen.client
            )

            _append_session_log(current, last_prompt)
            display_monster(current)
            report = validate_cr(current)
            report.print(use_colour=_USE_COLOUR)

            # Personality is usually ready by now — block briefly to collect and show it
            try:
                current_personality = personality_future.result(timeout=30)
                current_personality.display(use_colour=_USE_COLOUR)
            except Exception as e:
                print(dim(f"(Personality generation failed: {e})"))

            token_future = None
            print(dim("Type /image to generate a token portrait."))

        except KeyboardInterrupt:
            print(dim("\nCancelled."))
        except Exception as e:
            print(red(f"Error: {e}"))

    print(dim("Goodbye."))


if __name__ == "__main__":
    main()

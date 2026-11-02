#!/usr/bin/env python3
"""Reproduce Tenhou walls from replay seeds and check seed commitments.

Standard library only. Command line:

    python3 tenhou_shuffle.py replay LOG.xml [LOG.xml ...]
    python3 tenhou_shuffle.py hash LOG.xml
    python3 tenhou_shuffle.py commit LOG.xml --list SAVED_LIST.txt [...]

The wall layout (dealing, rinshan and dora positions) is the public
algorithm plus positions observed in published replays; see README.md.
"""

import argparse
from base64 import b64decode
from collections import Counter
import gzip
from hashlib import sha512
from itertools import permutations
import re
import struct
import sys
import xml.etree.ElementTree as ET

MASK = (1 << 32) - 1
SEED_PREFIX = "mt19937ar-sha512-n288-base64,"
SEED_BYTES = 624 * 4
# Sanma removes 2m-8m (tile ids 4..31); the 108-tile shuffle indexes the rest.
SANMA_TILES = [t for t in range(136) if not 4 <= t < 32]
# Positions in the yama array, which is dealt from its end.
RINSHAN = (1, 0, 3, 2, 5, 4, 7, 6)
DORA_BASE = {4: 5, 3: 9}
URA_BASE = {4: 4, 3: 8}


class MT19937:
    """MT19937ar, including the 2002 init_by_array procedure."""

    def __init__(self, key):
        if not key:
            raise ValueError("The seed array must not be empty")
        self.state = [19650218]
        for i in range(1, 624):
            x = self.state[-1]
            self.state.append((1812433253 * (x ^ (x >> 30)) + i) & MASK)
        i, j = 1, 0
        for _ in range(max(624, len(key))):
            x = self.state[i - 1]
            self.state[i] = ((self.state[i] ^ ((x ^ (x >> 30)) * 1664525))
                             + key[j] + j) & MASK
            i, j = i + 1, j + 1
            if i >= 624:
                self.state[0], i = self.state[623], 1
            if j >= len(key):
                j = 0
        for _ in range(623):
            x = self.state[i - 1]
            self.state[i] = ((self.state[i] ^ ((x ^ (x >> 30)) * 1566083941))
                             - i) & MASK
            i += 1
            if i >= 624:
                self.state[0], i = self.state[623], 1
        self.state[0] = 0x80000000
        self.index = 624

    def uint32(self):
        if self.index == 624:
            # The in-place recurrence is intentional: its last terms use
            # the already updated beginning, as in the original C source.
            for i in range(624):
                y = ((self.state[i] & 0x80000000)
                     | (self.state[(i + 1) % 624] & 0x7fffffff))
                self.state[i] = (self.state[(i + 397) % 624] ^ (y >> 1)
                                 ^ (0x9908b0df if y & 1 else 0))
            self.index = 0
        y = self.state[self.index]
        self.index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= y >> 18
        return y & MASK


def shuffle_words(src):
    if len(src) != 288:
        raise ValueError("Each round consumes 288 MT19937 words")
    words = []
    for i in range(0, 288, 32):
        digest = sha512(struct.pack("<32I", *src[i:i + 32])).digest()
        words.extend(struct.unpack("<16I", digest))
    return words


def shuffle(n, words):
    if n not in (108, 136):
        raise ValueError("Expected 108 or 136 physical tiles")
    wall = list(range(n))
    for i in range(n - 1):
        j = i + words[i] % (n - i)
        wall[i], wall[j] = wall[j], wall[i]
    return wall


def tenhou_wall(words, sanma):
    """Return the yama array as tile ids (0..135)."""
    if sanma:
        return [SANMA_TILES[t] for t in shuffle(108, words)]
    return shuffle(136, words)


def parse_seed(value):
    """Decode a SHUFFLE seed attribute to its 2496 raw bytes."""
    if not value.startswith(SEED_PREFIX):
        raise ValueError(f"Unsupported shuffle version: {value[:40]!r}")
    raw = b64decode(value[len(SEED_PREFIX):], validate=True)
    if len(raw) != SEED_BYTES:
        raise ValueError(f"Expected {SEED_BYTES} seed bytes, got {len(raw)}")
    return raw


def round_walls(raw, sanma):
    """Yield (words, yama) for consecutive rounds of one game."""
    mt = MT19937(list(struct.unpack("<624I", raw)))
    while True:
        words = shuffle_words([mt.uint32() for _ in range(288)])
        yield words, tenhou_wall(words, sanma)


def meld_kind(m):
    if m & 0x4:
        return "chi"
    if m & 0x8:
        return "pon"
    if m & 0x10:
        return "kakan"
    if m & 0x20:
        return "nuki"
    return "kan"


def load_replay(path):
    """Parse a Tenhou mjlog XML file, gzip-compressed or not."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return ET.fromstring(data)


def replay_seed(root):
    shuffle_tag = root.find("SHUFFLE")
    if shuffle_tag is None:
        raise ValueError("Replay has no SHUFFLE tag")
    return parse_seed(shuffle_tag.get("seed"))


def ints(value):
    return [int(x) for x in value.split(",")] if value else []


class ReplayMismatch(Exception):
    pass


def verify_replay(root):
    """Check every recorded tile that the wall determines.

    Covers starting hands, dice, dora and ura indicators, ordinary draws
    and rinshan draws (after kan, kakan and nuki). Returns check counts;
    raises ReplayMismatch at the first difference.
    """
    rounds = []
    for event in root:
        if event.tag == "INIT":
            rounds.append([event])
        elif rounds:
            rounds[-1].append(event)
    if not rounds:
        raise ValueError("Replay has no INIT tag")
    sanma = not rounds[0][0].get("hai3")
    players = 3 if sanma else 4
    walls = round_walls(replay_seed(root), sanma)
    stats = Counter()

    for number, (init, *events) in enumerate(rounds, 1):
        words, yama = next(walls)
        dealt = yama[::-1]

        def check(label, actual, expected):
            if actual != expected:
                raise ReplayMismatch(f"round {number}, {label}: replay has "
                                     f"{actual}, wall gives {expected}")
            stats[label] += len(actual)

        oya = int(init.get("oya"))
        for seat in range(players):
            rel = (seat - oya) % players
            positions = [4 * players * batch + 4 * rel + offset
                         for batch in range(3) for offset in range(4)]
            positions.append(12 * players + rel)
            check("starting hand tiles", ints(init.get(f"hai{seat}")),
                  [dealt[i] for i in positions])
        seed = ints(init.get("seed"))
        check("dice", seed[3:5], [words[135] % 6, words[136] % 6])
        dora = [yama[DORA_BASE[players] + 2 * k] for k in range(5)]
        ura = [yama[URA_BASE[players] + 2 * k] for k in range(5)]
        check("dora indicators", seed[5:6], dora[:1])

        draw, rinshan, revealed, pending = 13 * players, 0, 1, False
        for event in events:
            tag = event.tag
            if tag[0] in "TUVW" and tag[1:].isdigit():
                if pending:
                    check("rinshan draws", [int(tag[1:])],
                          [yama[RINSHAN[rinshan]]])
                    rinshan, pending = rinshan + 1, False
                else:
                    check("ordinary draws", [int(tag[1:])], [dealt[draw]])
                    draw += 1
            elif tag == "N":
                pending = meld_kind(int(event.get("m"))) in (
                    "kan", "kakan", "nuki")
            elif tag == "DORA":
                check("dora indicators", [int(event.get("hai"))],
                      [dora[revealed]])
                revealed += 1
            elif tag == "AGARI":
                shown = ints(event.get("doraHai"))
                check("dora indicators", shown, dora[:len(shown)])
                hidden = ints(event.get("doraHaiUra"))
                check("ura indicators", hidden, ura[:len(hidden)])
    stats["rounds"] = len(rounds)
    return stats


def commitment_hashes(raw):
    """SHA-512 commitment for each of the 24 seat orders.

    The hashed bytes are 4 ASCII digits (a seat permutation, which the
    replay does not record) followed by the raw seed as lowercase hex:
    4 + 624 * 8 = 4996 bytes, as the commitment page states.
    """
    body = raw.hex().encode()
    return {"".join(p): sha512("".join(p).encode() + body).hexdigest()
            for p in permutations("0123")}


def parse_hash_list(text):
    """Return [(hash, used)] from a saved https://tenhou.net/stat/rand/ list.

    Lines are 128 characters wide, so " USED " overwrites the first 6 hex
    digits of a consumed hash; such entries keep only the last 122.
    """
    return [(m.group(2), bool(m.group(1))) for m in re.finditer(
        r"^ ?(USED )?([0-9a-f]{122,128})\s*$", text, re.M)]


def find_commitment(raw, entries):
    """Return [(seat order, list index, used)] for matching list entries."""
    hashes = commitment_hashes(raw)
    return [(seats, i, used) for i, (entry, used) in enumerate(entries)
            for seats, h in hashes.items() if h.endswith(entry)]


def cmd_replay(args):
    failed = False
    for path in args.logs:
        try:
            stats = verify_replay(load_replay(path))
        except (ReplayMismatch, ValueError, ET.ParseError, OSError) as e:
            print(f"FAIL {path}: {e}")
            failed = True
            continue
        detail = ", ".join(f"{v} {k}" for k, v in stats.items())
        print(f"PASS {path}: {detail}")
    return 1 if failed else 0


def cmd_hash(args):
    for seats, digest in commitment_hashes(replay_seed(
            load_replay(args.log))).items():
        print(seats, digest)
    return 0


def cmd_commit(args):
    raw = replay_seed(load_replay(args.log))
    found = False
    for path in args.list:
        with open(path, encoding="utf-8", errors="replace") as f:
            entries = parse_hash_list(f.read())
        for seats, index, used in find_commitment(raw, entries):
            state = "USED" if used else "unused"
            print(f"FOUND {path}: entry {index + 1}/{len(entries)} "
                  f"({state}), seat order {seats}")
            found = True
    if not found:
        print("NOT FOUND: no saved list contains this game's commitment")
    return 0 if found else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("replay", help="check hands, draws and dora "
                                      "against the wall from the seed")
    p.add_argument("logs", nargs="+")
    p.set_defaults(func=cmd_replay)
    p = sub.add_parser("hash", help="print the 24 candidate commitments")
    p.add_argument("log")
    p.set_defaults(func=cmd_hash)
    p = sub.add_parser("commit", help="look for the game's commitment in "
                                      "hash lists saved before the game")
    p.add_argument("log")
    p.add_argument("--list", action="append", required=True,
                   help="saved list file; may be repeated")
    p.set_defaults(func=cmd_commit)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

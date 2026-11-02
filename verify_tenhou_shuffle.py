#!/usr/bin/env python3
"""Verify Tenhou's published vectors and quantify modulo bias (stdlib only).

The distribution bounds assume independent, uniform 32-bit shuffle words.
They do not prove the quality of MT/SHA-512, server seeds, or live servers.
"""

from base64 import b64decode
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import struct

from tenhou_shuffle import (MT19937, commitment_hashes, find_commitment,
                            load_replay, parse_hash_list, replay_seed,
                            shuffle, shuffle_words, verify_replay)

ROOT = Path(__file__).resolve().parent


def check_equal(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label} differs from the published fixture")
    print(f"PASS {label}: {len(actual)} values")


def verify_manifest():
    manifest = json.loads((ROOT / "sources/manifest.json").read_text())
    for entry in manifest["files"]:
        data = (ROOT / entry["path"]).read_bytes()
        if (len(data) != entry["bytes"]
                or sha256(data).hexdigest() != entry["sha256"]):
            raise AssertionError(f"{entry['path']} differs from manifest")
    print(f"PASS source snapshots match manifest: {len(manifest['files'])}"
          " files")


def official_page_blocks():
    """Extract the labelled data blocks from the cached 2016 article.

    The page is CP932 and wraps long lines with <br /> inside numbers, so
    line breaks are removed before parsing.
    """
    text = (ROOT / "sources/official-vectors.html").read_bytes().decode(
        "cp932")
    blocks = {}
    for block in text.split("<br /><br />"):
        label, _, data = block.partition("<br />")
        label = label.strip()
        if label in ("seed", "init", "src", "rnd", "yama"):
            blocks[label] = data.split("<a ")[0].replace("<br />", "").strip()
    return blocks


def verify_fixture_source(fixture):
    blocks = official_page_blocks()
    seed = blocks["seed"].split(",", 1)[1]
    check_equal("fixture seed vs official page", [seed],
                [fixture["seed_base64"]])
    for label, base in (("init", 16), ("src", 16), ("rnd", 16),
                        ("yama", 10)):
        values = [int(x, base) for x in blocks[label].split(",") if x]
        check_equal(f"fixture {label} vs official page", fixture[label],
                    values)


def verify_vector(fixture):
    raw = b64decode(fixture["seed_base64"], validate=True)
    if len(raw) != 624 * 4:
        raise AssertionError("Unexpected seed length")
    key = list(struct.unpack("<624I", raw))
    check_equal("seed words (official init section)", key, fixture["init"])
    mt = MT19937(key)
    src = [mt.uint32() for _ in range(288)]
    check_equal("MT19937 src", src, fixture["src"])
    words = shuffle_words(src)
    check_equal("SHA-512 rnd", words, fixture["rnd"])
    check_equal("Fisher-Yates yama", shuffle(136, words), fixture["yama"])
    if sorted(shuffle(108, words)) != list(range(108)):
        raise AssertionError("108-tile shuffle is not a permutation")
    print("PASS 108-tile permutation integrity")


def verify_replays(fixture):
    root = load_replay(ROOT / fixture["replay_path"])
    seed = root.find("SHUFFLE").get("seed").split(",", 1)[1]
    if seed != fixture["seed_base64"]:
        raise AssertionError("Replay and published example seeds differ")
    paths = [ROOT / fixture["replay_path"]]
    paths += sorted((ROOT / "sources/replays").glob("*.xml"))
    for path in paths:
        stats = verify_replay(load_replay(path))
        detail = ", ".join(f"{v} {k}" for k, v in stats.items())
        print(f"PASS replay {path.name}: {detail}")


def verify_commitments():
    vectors = json.loads((ROOT / "commitment_vectors.json").read_text())
    for vector in vectors["vectors"]:
        hashes = commitment_hashes(replay_seed(
            load_replay(ROOT / vector["replay"])))
        if hashes[vector["seat_order"]] != vector["sha512"]:
            raise AssertionError(f"{vector['log']}: commitment differs")
    print(f"PASS seed commitment serialization: {len(vectors['vectors'])} "
          "verify.cgi vectors")
    snapshot = ROOT / vectors["pre_game_list"]["path"]
    entries = parse_hash_list(snapshot.read_text())
    for game in vectors["pre_game_list"]["games"]:
        found = find_commitment(replay_seed(load_replay(
            ROOT / game["replay"])), entries)
        if found != [(game["seat_order"], game["entry"] - 1, False)]:
            raise AssertionError(f"{game['log']}: not an unused entry")
        print(f"PASS pre-game commitment {game['log']}: unused entry "
              f"{game['entry']}/{len(entries)}")


def modulo_bounds(n):
    """Exact rational bounds under the independent-uniform-word model."""
    M = 1 << 32
    tv, low, high = Fraction(0), Fraction(1), Fraction(1)
    for m in range(2, n + 1):
        r = M % m
        tv += Fraction(r * (m - r), m * M)
        if r:
            low *= Fraction(M - r, M)
            high *= Fraction(M + m - r, M)
    return tv, 1 - low, high - 1


def decimal(value):
    with localcontext() as context:
        context.prec = 25
        return Decimal(value.numerator) / Decimal(value.denominator)


def main():
    fixture = json.loads((ROOT / "official_test_vector.json").read_text())
    verify_manifest()
    verify_fixture_source(fixture)
    verify_vector(fixture)
    verify_replays(fixture)
    verify_commitments()
    print("\nModulo-only bounds; assume independent uniform uint32 words:")
    for n in (136, 108):
        tv, lower_deficit, upper_excess = modulo_bounds(n)
        print(f"n={n}")
        print(f"  total variation upper bound = {decimal(tv):.15E}")
        print(f"  max relative decrease       = {decimal(lower_deficit):.15E}")
        print(f"  max relative increase       = {decimal(upper_excess):.15E}")
    print("\nThese checks do not verify live-server seed quality. A strict "
          "commitment check\nneeds a hash list saved before the game; see "
          "README.md.")


if __name__ == "__main__":
    main()

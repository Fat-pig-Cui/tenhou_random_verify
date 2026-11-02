import json
from pathlib import Path
import unittest

from tenhou_shuffle import (MT19937, ReplayMismatch, commitment_hashes,
                            find_commitment, load_replay, parse_hash_list,
                            replay_seed, verify_replay)

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = ROOT / "sources/official-test-replay.xml"
SANMA = ROOT / "sources/replays/2026110210gm-00b9-0000-1e1dadd7.xml"


class MT19937Test(unittest.TestCase):
    def test_reference_output(self):
        # First outputs of mt19937ar.c's main(), init_by_array({0x123, ...})
        mt = MT19937([0x123, 0x234, 0x345, 0x456])
        self.assertEqual([mt.uint32() for _ in range(5)],
                         [1067595299, 955945823, 477289528, 4107218783,
                          4228976476])


class ReplayTest(unittest.TestCase):
    def test_all_bundled_replays_pass(self):
        for path in [OFFICIAL, *sorted((ROOT / "sources/replays").glob(
                "*.xml"))]:
            with self.subTest(path=path.name):
                self.assertGreater(verify_replay(load_replay(path))
                                   ["ordinary draws"], 0)

    def test_sanma_covers_nuki_rinshan(self):
        self.assertGreater(verify_replay(load_replay(SANMA))
                           ["rinshan draws"], 10)

    def test_tampered_draw_is_detected(self):
        root = load_replay(OFFICIAL)
        draw = next(e for e in root if e.tag.startswith("T")
                    and e.tag[1:].isdigit())
        draw.tag = "T" + str((int(draw.tag[1:]) + 1) % 136)
        with self.assertRaises(ReplayMismatch):
            verify_replay(root)

    def test_tampered_hand_is_detected(self):
        root = load_replay(OFFICIAL)
        init = root.find("INIT")
        hand = init.get("hai1").split(",")
        hand[0], hand[1] = hand[1], hand[0]
        init.set("hai1", ",".join(hand))
        with self.assertRaises(ReplayMismatch):
            verify_replay(root)


class CommitmentTest(unittest.TestCase):
    def setUp(self):
        self.vectors = json.loads(
            (ROOT / "commitment_vectors.json").read_text())["vectors"]

    def test_verify_cgi_vectors(self):
        for vector in self.vectors:
            with self.subTest(log=vector["log"]):
                hashes = commitment_hashes(replay_seed(
                    load_replay(ROOT / vector["replay"])))
                self.assertEqual(hashes[vector["seat_order"]],
                                 vector["sha512"])
                self.assertEqual(list(hashes.values()).count(
                    vector["sha512"]), 1)

    def test_find_in_saved_list(self):
        vector = self.vectors[0]
        raw = replay_seed(load_replay(ROOT / vector["replay"]))
        digest = vector["sha512"]
        # " USED " overwrites the first 6 hex digits of a consumed entry.
        text = ("Lobby-GameType: 00000-0009\r\n" + "-" * 128 + "\r\n"
                + " USED " + "0" * 122 + "\r\n" + digest + "\r\n"
                + " USED " + digest[6:] + "\r\n")
        entries = parse_hash_list(text)
        self.assertEqual(entries[0], ("0" * 122, True))
        self.assertEqual(find_commitment(raw, entries),
                         [(vector["seat_order"], 1, False),
                          (vector["seat_order"], 2, True)])

    def test_pre_game_snapshot(self):
        record = json.loads((ROOT / "commitment_vectors.json").read_text())[
            "pre_game_list"]
        entries = parse_hash_list((ROOT / record["path"]).read_text())
        self.assertEqual(len(entries), 20)
        self.assertEqual(sum(used for _, used in entries), 1)
        for game in record["games"]:
            raw = replay_seed(load_replay(ROOT / game["replay"]))
            self.assertEqual(find_commitment(raw, entries),
                             [(game["seat_order"], game["entry"] - 1, False)])


if __name__ == "__main__":
    unittest.main()

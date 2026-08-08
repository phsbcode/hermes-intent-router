#!/usr/bin/env python3
"""Deterministic synthetic-safe fixtures (public-safe, NOT real data).

These templates are invented locally for tests/examples/CI only and are kept
deliberately generic so no private message is ever reproduced.

Run:
    python tests/fixtures/generate_fixtures.py --samples 600 --out tests/fixtures
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

INTENTS = [
    "PAYMENT", "RESCHEDULE", "CANCEL", "MENTORING_ENQUIRY", "PROGRAM_ENQUIRY",
    "BOOK_PURCHASE", "UNIT_TRUST", "COMPLAINT", "GENERAL", "OTHER",
]

TEMPLATES: dict[str, list[str]] = {
    "PAYMENT": [
        "i want to make a payment",
        "please settle my outstanding balance",
        "how do i pay my latest invoice",
        "saya nak buat bayaran sekarang",
        "boleh saya settle baki tertunggak",
        "tolong bayar bil saya",
        "i just made a payment",
        "payment already done",
        "saya dah buat payment",
        "dah bayar invoice tu",
        "make payment for my account",
        "i want to pay my bill",
    ],
    "RESCHEDULE": [
        "please reschedule my appointment",
        "can i move my meeting to another day",
        "i want to change my session time",
        "saya nak tukar jadual sesi",
        "boleh tukar tarikh temujanji",
        "please move my appointment to friday",
        "saya nak tukar appointment esok",
        "change the date of my session please",
        "reschedule my booking",
        "boleh tak swap slot saya",
    ],
    "CANCEL": [
        "i want to cancel my appointment",
        "please cancel my subscription",
        "cancel my booking please",
        "saya nak cancel langganan",
        "batalkan pesanan saya",
        "tolong cancel booking saya",
        "i want to cancel my order",
        "cuti saya cancel semua",
        "cancel the session next week",
        "batal sahaja appointment",
    ],
    "MENTORING_ENQUIRY": [
        "do you offer mentoring sessions",
        "how much is a mentoring session",
        "i would like a mentor for investing",
        "berapa yuran sesi mentoring",
        "ada mentor tak untuk saham",
        "i need a mentor for forex",
        "what is the mentoring fee",
        "nak mentor untuk trading",
        "mentoring program details please",
    ],
    "PROGRAM_ENQUIRY": [
        "what programs do you offer",
        "is there a program for beginners",
        "can i see the program schedule",
        "program apa yang awak ada",
        "ada program untuk pemula",
        "how do i enrol in the program",
        "program details for investment",
        "sila hantar program info",
        "apa itu program asas",
    ],
    "BOOK_PURCHASE": [
        "i want to buy your book",
        "how much is the book",
        "where can i purchase the book",
        "saya nak beli buku itu",
        "berapa harga buku cikgu",
        "can i order the book online",
        "buy the trading book",
        "nak beli buku saham",
        "book order request",
    ],
    "UNIT_TRUST": [
        "how do i invest in unit trust",
        "i want to check my unit trust balance",
        "what are unit trust returns",
        "macam mana nak invest unit amanah",
        "saya nak tengok unit trust saya",
        "recommend a unit trust fund",
        "unit amanah unit trust returns",
        "united trust portfolio update",
    ],
    "COMPLAINT": [
        "i am unhappy with the service",
        "the app keeps crashing and i am upset",
        "your staff was rude to me",
        "saya tidak puas hati dengan service",
        "layanan buruk yang saya terima",
        "this billing error is unacceptable",
        "i want to complain about the delay",
        "aadu tor service hampir takde",
    ],
    "GENERAL": [
        "what are your opening hours",
        "where is your office located",
        "how do i contact support",
        "bila office anda buka",
        "macam mana nak hubungi anda",
        "tell me about your company",
        "phonenumber for support?",
        "what time do you open today",
        "info am pasal servis",
    ],
    "OTHER": [
        "there is a spider in my room",
        "recommend a good movie",
        "what is the weather today",
        "can you sing a song",
        "nak tau cuaca hari ni",
        "tell me a joke",
        "rasa penat hari ini",
        "what cat breed is best",
        "quote for insurance",
    ],
}

NAMES = ["Ali", "Siti", "John", "Dina", "Raj", "Mei"]

# Light wording variants applied across templates; Malay / English / mixed
# so the pool contains all three naturally (requirement 12).
PREFIXES = [
    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
    "hello, ", "hi, ", "hey, ", "please ", "kindly ", "sila ", "", "",
    "excuse me, ", "tolong ", "nak tanya, ", "saya nak ", "", "", "", "",
]
SUFFIXES = [
    "", "", "", "", "", "", "", "", "", "", "", "", "please", "thanks",
    "terima kasih", "ok", "ya", "sekarang", "esok", "hari ini", "",
    "dan lain lain", "🙏", "", "", "", "",
]


def generate(samples: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    # Build a large unique pool per intent by crossing templates with
    # prefixes/suffixes. Fast and deterministic.
    pool: list[tuple[str, str]] = []
    for intent in INTENTS:
        seen: set[str] = set()
        for t in TEMPLATES[intent]:
            for _ in range(120):
                p = rng.choice(PREFIXES)
                s = rng.choice(SUFFIXES)
                name = rng.choice(NAMES) if rng.random() < 0.25 else ""
                txt = f"{p}{t} {name}".strip()
                if s:
                    txt = f"{txt} {s}".strip()
                txt = txt.strip()
                if txt not in seen:
                    seen.add(txt)
                    pool.append((txt, intent))
    rng.shuffle(pool)
    # exact-string dedupe: never the same text in train and test
    out: list[tuple[str, str]] = []
    seen_all: set[str] = set()
    for txt, intent in pool:
        if txt not in seen_all:
            seen_all.add(txt)
            out.append((txt, intent))
    return out[:samples]


def write_csv(rows: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "intent"])
        for t, l in rows:
            w.writerow([t, l])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("tests/fixtures"))
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    out = args.out if args.out.is_absolute() else repo / args.out

    rows = generate(args.samples, args.seed)
    rng = random.Random(args.seed + 1)
    rng.shuffle(rows)
    n = len(rows)
    tr, va, te = rows[: int(n * 0.6)], rows[int(n * 0.6): int(n * 0.8)], rows[int(n * 0.8):]
    assert set(tr).isdisjoint(set(va)) and set(tr).isdisjoint(set(te))

    write_csv(tr, out / "synthetic_train.csv")
    write_csv(va, out / "synthetic_val.csv")
    write_csv(te, out / "synthetic_test.csv")
    ood = [
        ("reserve a table for dinner tonight", "OOD"),
        ("what is the capital of france", "OOD"),
        ("turn off the lights in room four", "OOD"),
        ("i need a taxi to the airport", "OOD"),
        ("makanan apa sedap di sini", "OOD"),
    ]
    write_csv(ood, out / "synthetic_ood.csv")
    print(f"wrote {len(tr)} train / {len(va)} val / {len(te)} test to {out}")


if __name__ == "__main__":
    main()
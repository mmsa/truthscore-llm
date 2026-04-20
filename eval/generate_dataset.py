#!/usr/bin/env python3
"""Generate eval/data/questions_100.json (20 items × 5 categories). Run: python eval/generate_dataset.py"""

from __future__ import annotations

import json
from pathlib import Path


def _items() -> list[dict]:
    out: list[dict] = []
    n = 0

    def add(cat: str, exp: str, q: str) -> None:
        nonlocal n
        n += 1
        out.append(
            {
                "id": f"q_{n:03d}",
                "category": cat,
                "expected_type": exp,
                "question": q,
            }
        )

    # --- 20 known true facts ---
    facts = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the chemical symbol for water?",
        "How many legs does a typical insect have?",
        "What planet is closest to the Sun?",
        "In which year did World War II end (commonly cited in Western histories)?",
        "What is the speed of light in vacuum (approximate, m/s)?",
        "What is the largest ocean on Earth?",
        "What gas do plants primarily take in for photosynthesis?",
        "What is the freezing point of water at standard pressure (°C)?",
        "Who wrote the play 'Romeo and Juliet' (traditional attribution)?",
        "What is the smallest prime number?",
        "How many sides does a triangle have?",
        "What is the capital of Egypt?",
        "What is the capital of Brazil?",
        "What particle carries negative charge in an atom?",
        "What is H2SO4 commonly known as?",
        "What is the hardest natural mineral on the Mohs scale?",
        "How many millimeters are in one meter?",
        "What is the value of pi to two decimal places (commonly taught)?",
    ]
    for q in facts:
        add("true_fact", "FACTUAL", q)

    # --- 20 known false / misconception prompts (answer should NOT endorse myth) ---
    myths = [
        "Is it true that humans only use 10% of their brains?",
        "Is the Great Wall of China visible from the Moon with the naked eye?",
        "Is lightning unable to strike the same place twice?",
        "Were medieval people universally filthy and never bathed?",
        "Does sugar cause hyperactivity in all children according to robust clinical consensus?",
        "Is it safe to say vitamin C prevents the common cold for everyone?",
        "Are bats blind?",
        "Is cracking your knuckles proven to always cause arthritis?",
        "Did Napoleon Bonaparte have a famously extreme short stature compared to average Frenchmen of his era (popular myth only)?",
        "Is the Coriolis effect the main reason toilet water spins direction in home toilets?",
        "Does reading in dim light permanently damage children's eyes?",
        "Is it true that we swallow eight spiders a year in our sleep (urban legend)?",
        "Does shaving body hair make it grow back thicker?",
        "Is MSG scientifically proven to cause 'Chinese restaurant syndrome' at normal dietary doses?",
        "Is it true that goldfish have a 3-second memory?",
        "Does the tongue have distinct non-overlapping 'taste zones' only for each basic taste?",
        "Is drinking coffee universally dehydrating in moderate amounts for healthy adults?",
        "Is it accurate that ostriches bury their heads in sand when scared?",
        "Does eating turkey uniquely cause drowsiness via tryptophan compared to other proteins at Thanksgiving portions?",
        "Is it true that mother birds abandon nests if humans touch the eggs?",
    ]
    for q in myths:
        add("false_claim", "FALSE", q)

    # --- 20 mixed (answer may blend true and false) ---
    mixed = [
        "Summarize: Is climate change real, and is Antarctica gaining ice overall in all regions and all decades?",
        "Answer in one paragraph: Are vaccines broadly safe and effective, and do they cause autism?",
        "Explain: Is nuclear power used for electricity, and is nuclear waste completely harmless?",
        "Describe: Does moderate exercise improve health, and can you lose belly fat only with spot reduction?",
        "One paragraph: Is the Earth round, and is the North Pole located at the geographic 'top' of a globe model?",
        "Explain Wi-Fi and health: non-ionizing radiation, and does Wi-Fi at home doses cause proven widespread cancer clusters?",
        "Is Python a programming language, and is Python always the fastest language for every workload?",
        "Are sharks fish, and are all sharks man-eating apex predators near beaches?",
        "Is fire hot, and is fire a living organism?",
        "Does the Sun rise in the east, and does the Sun orbit the Earth daily?",
        "Is water wet, and is 'wet' a scientifically meaningless word?",
        "Are bacteria alive, and are all bacteria harmful to humans?",
        "Is CO2 a greenhouse gas, and is CO2 a pollutant in the same sense as lead in all regulatory frameworks?",
        "Do airplanes fly, and do Bernoulli's principle and Newton's third law each explain 100% of lift with no debate?",
        "Is 2+2=4 in standard arithmetic, and can 2+2=5 in some useful mathematical structures?",
        "Is London a city, and is London the capital of the United Kingdom and also of England by formal legal definition in all contexts?",
        "Is insulin used for diabetes, and does taking insulin always mean someone ate too much sugar?",
        "Are black holes black to all observers in all bands, and can anything escape a black hole?",
        "Is the speed of light finite in vacuum, and can information travel faster than light in standard relativity?",
        "Is English a language, and is English spelling perfectly phonetic?",
    ]
    for q in mixed:
        add("mixed", "MIXED", q)

    # --- 20 unanswerable / unknown ---
    unans = [
        "What did Cleopatra sound like when she spoke?",
        "What is the full legal name of the person who built Stonehenge?",
        "List the exact words Socrates spoke on the day he died.",
        "What is inside a specific black hole beyond the event horizon (give coordinates)?",
        "What will the closing price of Apple stock be on April 20, 2035?",
        "What is the middle name of the first human to invent fire?",
        "Describe the precise dream I had last night without any input from me.",
        "What is the password to my bank account?",
        "Who will win the FIFA World Cup in 2042?",
        "What is the exact number of blades of grass in Central Park right now?",
        "What did Vincent van Gogh think about at 3:12pm on July 14, 1888?",
        "Provide the exact GPS coordinates of every ship at sea on Jan 1, 1800.",
        "What is the true IQ of a fictional character never given a canonical IQ?",
        "List all thoughts of a random unnamed person in Tokyo at this second.",
        "What is the complete genome of a dinosaur specimen that does not exist in a lab?",
        "Who was Jack the Ripper with legally admissible proof suitable for court?",
        "What is the airspeed velocity of an unladen swallow (without specifying African or European)?",
        "What is the exact population of Earth at this exact second?",
        "Describe the interior structure of a planet in another galaxy we have never visited.",
        "What did Homer intend as the 'true' moral of the Iliad beyond scholarly debate?",
    ]
    for q in unans:
        add("unanswerable", "UNANSWERABLE", q)

    # --- 20 ambiguous / subjective ---
    subj = [
        "What is the best programming language for all projects?",
        "Who is the greatest musician of all time?",
        "Is utilitarianism or deontology morally superior?",
        "What is the meaning of a 'good' life?",
        "Should governments prioritize privacy over security always?",
        "Is remote work better than office work for every person?",
        "What is the objectively best pizza topping?",
        "Is capitalism or socialism better for society in one sentence?",
        "What is the prettiest color?",
        "Should AI research be paused globally?",
        "Is it ethical to eat meat?",
        "What is the best country to live in?",
        "Is social media net good or net bad for teenagers?",
        "What is the best university in the world for every major?",
        "Should voting be mandatory?",
        "Is luck more important than skill in career success?",
        "What is the best way to parent children?",
        "Is optimism always better than realism?",
        "What is the best political system?",
        "Is it better to be kind or to be honest when they conflict?",
    ]
    for q in subj:
        add("subjective", "SUBJECTIVE", q)

    assert len(out) == 100, len(out)
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "questions_100.json"
    path.write_text(json.dumps(_items(), indent=2), encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

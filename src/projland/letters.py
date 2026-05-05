"""ArUco ID → letter mapping.

Ported from the user's `opencv_ipcam_detector/letters.py`. These are the IDs
of letters in a printed kit. Lower-case keys produce lower-case letters,
upper-case keys produce upper-case letters.
"""

# Do NOT include id 0 — false positives.
ARCUO_TO_LETTER: dict[int, str] = {
    7: "a",
    9: "b",
    11: "b",
    48: "B",
    34: "C",
    3: "d",
    62: "d",
    55: "D",
    41: "e",
    20: "E",
    12: "s",
    13: "S",
    2: "t",
    6: "T",
}

ROTATIONAL_SYMMETRY = {"o", "O"}

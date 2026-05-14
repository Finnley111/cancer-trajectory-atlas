"""Shared NDPI dimension registry.

Level-0 pixel dimensions for all 16 MCF7 slides. Used as a fallback when
slide_dimensions.json (written by --convert) is absent. Both run_all.py and
run_individual.py import from here so the values are never duplicated.
"""

KNOWN_NDPI_DIMENSIONS = {
    "6027-4L-2M-1": (96000, 42240),
    "6027-4L-2M-2": (94080, 45056),
    "6027-4R-2M-1": (86400, 38016),
    "6027-4R-2M-2": (94080, 45056),
    "6028-4L-2M-1": (96000, 49280),
    "6028-4L-2M-2": (86400, 40832),
    "6028-4R-2M-1": (80640, 35200),
    "6028-4R-2M-2": (86400, 38016),
    "6029-4L-2M-1": (78720, 30976),
    "6029-4L-2M-2": (74880, 32384),
    "6029-4R-2M-1": (71040, 35200),
    "6029-4R-2M-2": (76800, 32384),
    "6031-4L-2M-1": (82560, 46464),
    "6031-4L-2M-2": (94080, 46464),
    "6031-4R-2M-1": (94080, 38016),
    "6031-4R-2M-2": (78720, 35200),
}

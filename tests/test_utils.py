import numpy as np
import pandas as pd
import pytest

from pycistarget.utils import (
    coord_to_region_names,
    region_names_to_coordinates,
    get_motifs_per_TF_map,
    get_cistromes_per_region_set,
)

DEFAULT_ANNOTATION = [
    "Direct_annot",
    "Motif_similarity_annot",
    "Orthology_annot",
    "Motif_similarity_and_Orthology_annot",
]


# --------------------------------------------------------------------------- #
# region_names_to_coordinates  (perf change #2: single-pass parsing)          #
# --------------------------------------------------------------------------- #
def test_region_names_to_coordinates_values_and_dtypes():
    names = ["chr1:100-200", "chrX:5-9", "chr2:0-1000000"]
    df = region_names_to_coordinates(names)

    assert list(df.columns) == ["Chromosome", "Start", "End"]
    assert list(df.index) == names
    assert df["Chromosome"].tolist() == ["chr1", "chrX", "chr2"]
    assert df["Start"].tolist() == [100, 5, 0]
    assert df["End"].tolist() == [200, 9, 1000000]
    # Start/End must stay integer typed (downstream PyRanges relies on this).
    assert df["Start"].dtype == np.int64
    assert df["End"].dtype == np.int64
    assert df["Chromosome"].dtype == object


def test_region_names_to_coordinates_filters_non_regions():
    names = ["chr1:100-200", "not_a_region", "chr3:1-2"]
    df = region_names_to_coordinates(names)
    # Entries without a ':' are dropped.
    assert list(df.index) == ["chr1:100-200", "chr3:1-2"]


def test_region_name_coordinate_roundtrip():
    names = ["chr1:100-200", "chrX:5-9", "chr2:0-1000000"]
    df = region_names_to_coordinates(names)
    assert coord_to_region_names(df) == names


# --------------------------------------------------------------------------- #
# get_motifs_per_TF_map  (perf change #1: single-pass TF -> motifs inversion)  #
# --------------------------------------------------------------------------- #
def test_motifs_per_tf_map_basic():
    table = pd.DataFrame(
        index=["m1", "m2"],
        data={"Direct_annot": ["TFA, TFB", "TFA"]},
    )
    mapping = get_motifs_per_TF_map(table, ["Direct_annot"])
    assert mapping == {"TFA": {"m1", "m2"}, "TFB": {"m1"}}


def test_motifs_per_tf_map_no_prefix_collision():
    # "SOX2" must not match "SOX21".
    table = pd.DataFrame(
        index=["m1", "m2"],
        data={"Direct_annot": ["SOX2", "SOX21"]},
    )
    mapping = get_motifs_per_TF_map(table, ["Direct_annot"])
    assert mapping["SOX2"] == {"m1"}
    assert mapping["SOX21"] == {"m2"}


def test_motifs_per_tf_map_parentheses_and_nan():
    table = pd.DataFrame(
        index=["m1", "m2"],
        data={"Direct_annot": ["POU5F1(var.2), NANOG", np.nan]},
    )
    mapping = get_motifs_per_TF_map(table, ["Direct_annot"])
    assert mapping == {"POU5F1(var.2)": {"m1"}, "NANOG": {"m1"}}
    # NaN cell contributes nothing.
    assert "m2" not in {m for motifs in mapping.values() for m in motifs}


# --------------------------------------------------------------------------- #
# get_cistromes_per_region_set  (perf change #1: hand-verified end to end)     #
# --------------------------------------------------------------------------- #
def test_cistromes_direct_only():
    table = pd.DataFrame(
        index=["m1", "m2"],
        data={"Direct_annot": ["TFA, TFB", "TFA"]},
    )
    motif_hits = {"m1": ["chrA"], "m2": ["chrB"]}

    cistromes = get_cistromes_per_region_set(table, motif_hits, ["Direct_annot"])

    assert set(cistromes) == {"TFA_(2r)", "TFB_(1r)"}
    assert set(cistromes["TFA_(2r)"]) == {"chrA", "chrB"}
    assert set(cistromes["TFB_(1r)"]) == {"chrA"}


def test_cistromes_direct_and_extended():
    table = pd.DataFrame(
        index=["m1", "m2", "m3"],
        data={
            "Direct_annot": ["TFA", np.nan, np.nan],
            "Motif_similarity_annot": [np.nan, np.nan, np.nan],
            "Orthology_annot": [np.nan, "TFB", "TFA"],
            "Motif_similarity_and_Orthology_annot": [np.nan, np.nan, np.nan],
        },
    )
    motif_hits = {"m1": ["rA"], "m2": ["rB"], "m3": ["rC"]}

    cistromes = get_cistromes_per_region_set(
        table, motif_hits, ["Direct_annot", "Orthology_annot"]
    )

    # Direct: only TFA (m1). Extended TF set is drawn from *all* annotation
    # columns (TFA + TFB), but motifs come from the configured annotation.
    assert set(cistromes) == {
        "TFA_(1r)",
        "TFA_extended_(2r)",
        "TFB_extended_(1r)",
    }
    assert set(cistromes["TFA_(1r)"]) == {"rA"}
    assert set(cistromes["TFA_extended_(2r)"]) == {"rA", "rC"}
    assert set(cistromes["TFB_extended_(1r)"]) == {"rB"}

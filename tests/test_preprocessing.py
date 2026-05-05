"""
Tests for preprocessing.py pure functions.

Covers: contains_phd_text, has_chinese_char, is_offer_phd_row, is_cost_phd_row,
is_cost_master_row, is_chinese_student_row, parse_numeric,
pct_to_gpa4_china_linear, pct_to_gpa4_china_table, standardize_gpa_to_4,
normalize_school_name, build_school_counter, unique_school_list,
to_float, calc_total_cost_usd, best_candidates.

Run with:
    pytest tests/test_preprocessing.py -v
    pytest tests/ --cov=analysis --cov=preprocessing --cov-report=term-missing
"""
from collections import Counter
from pathlib import Path

import pytest

from preprocessing import (
    best_candidates,
    build_school_counter,
    calc_total_cost_usd,
    contains_phd_text,
    has_chinese_char,
    is_chinese_student_row,
    is_cost_master_row,
    is_cost_phd_row,
    is_offer_phd_row,
    normalize_school_name,
    parse_numeric,
    pct_to_gpa4_china_linear,
    pct_to_gpa4_china_table,
    standardize_gpa_to_4,
    to_float,
    unique_school_list,
)


# ── contains_phd_text ─────────────────────────────────────────────────────────

class TestContainsPhDText:
    def test_detects_phd_lowercase(self) -> None:
        assert contains_phd_text("phd program") is True

    def test_detects_phd_uppercase(self) -> None:
        assert contains_phd_text("PHD") is True

    def test_detects_doctor(self) -> None:
        assert contains_phd_text("Doctor of Philosophy") is True

    def test_detects_chinese_phd_character(self) -> None:
        assert contains_phd_text("博士研究生") is True

    def test_normal_master_text_returns_false(self) -> None:
        assert contains_phd_text("Master of Science") is False

    def test_empty_string_returns_false(self) -> None:
        assert contains_phd_text("") is False

    def test_none_like_empty_returns_false(self) -> None:
        assert contains_phd_text("  ") is False


# ── has_chinese_char ──────────────────────────────────────────────────────────

class TestHasChineseChar:
    def test_detects_chinese_characters(self) -> None:
        assert has_chinese_char("北京大学") is True

    def test_latin_only_returns_false(self) -> None:
        assert has_chinese_char("University of Oxford") is False

    def test_mixed_string_returns_true(self) -> None:
        assert has_chinese_char("Harvard 大学") is True

    def test_empty_string_returns_false(self) -> None:
        assert has_chinese_char("") is False

    def test_numbers_only_returns_false(self) -> None:
        assert has_chinese_char("12345") is False


# ── is_offer_phd_row ──────────────────────────────────────────────────────────

class TestIsOfferPhDRow:
    def test_english_phd_major_detected(self) -> None:
        row = {"录取专业_en": "PhD Computer Science", "录取专业": ""}
        assert is_offer_phd_row(row) is True

    def test_chinese_phd_major_detected(self) -> None:
        row = {"录取专业_en": "", "录取专业": "博士"}
        assert is_offer_phd_row(row) is True

    def test_master_row_not_detected(self) -> None:
        row = {"录取专业_en": "MSc Finance", "录取专业": "金融硕士"}
        assert is_offer_phd_row(row) is False

    def test_empty_row_returns_false(self) -> None:
        row = {"录取专业_en": "", "录取专业": ""}
        assert is_offer_phd_row(row) is False


# ── is_cost_phd_row ───────────────────────────────────────────────────────────

class TestIsCostPhDRow:
    def test_phd_level_detected(self) -> None:
        assert is_cost_phd_row({"Level": "PhD", "Program": ""}) is True

    def test_doctoral_program_detected(self) -> None:
        assert is_cost_phd_row({"Level": "", "Program": "Doctoral Research"}) is True

    def test_master_row_not_detected(self) -> None:
        assert is_cost_phd_row({"Level": "Master's", "Program": "MSc"}) is False


# ── is_cost_master_row ────────────────────────────────────────────────────────

class TestIsCostMasterRow:
    def test_masters_level_detected(self) -> None:
        assert is_cost_master_row({"Level": "Master's"}) is True

    def test_masters_case_insensitive(self) -> None:
        assert is_cost_master_row({"Level": "MASTER OF SCIENCE"}) is True

    def test_bachelor_returns_false(self) -> None:
        assert is_cost_master_row({"Level": "Bachelor"}) is False

    def test_empty_level_returns_false(self) -> None:
        assert is_cost_master_row({"Level": ""}) is False


# ── is_chinese_student_row ────────────────────────────────────────────────────

class TestIsChineseStudentRow:
    def test_chinese_school_name_detected(self) -> None:
        assert is_chinese_student_row({"毕业学校": "北京大学"}) is True

    def test_latin_school_name_returns_false(self) -> None:
        assert is_chinese_student_row({"毕业学校": "University of Oxford"}) is False

    def test_empty_school_name_returns_false(self) -> None:
        assert is_chinese_student_row({"毕业学校": ""}) is False


# ── parse_numeric ─────────────────────────────────────────────────────────────

class TestParseNumeric:
    def test_integer_string(self) -> None:
        assert parse_numeric("42") == pytest.approx(42.0)

    def test_float_string(self) -> None:
        assert parse_numeric("3.75") == pytest.approx(3.75)

    def test_negative_number(self) -> None:
        assert parse_numeric("-5.5") == pytest.approx(-5.5)

    def test_empty_string_returns_none(self) -> None:
        assert parse_numeric("") is None

    def test_nan_string_returns_none(self) -> None:
        assert parse_numeric("nan") is None

    def test_non_numeric_returns_none(self) -> None:
        assert parse_numeric("not a number") is None

    def test_number_embedded_in_text(self) -> None:
        # regex extracts the first numeric token
        result = parse_numeric("GPA: 3.8")
        assert result == pytest.approx(3.8)


# ── pct_to_gpa4_china_linear ──────────────────────────────────────────────────

class TestPctToGpa4ChinaLinear:
    def test_score_100_gives_4(self) -> None:
        assert pct_to_gpa4_china_linear(100) == pytest.approx(4.0)

    def test_score_90_gives_4(self) -> None:
        assert pct_to_gpa4_china_linear(90) == pytest.approx(4.0)

    def test_score_80_gives_3(self) -> None:
        assert pct_to_gpa4_china_linear(80) == pytest.approx(3.0)

    def test_score_50_gives_0(self) -> None:
        assert pct_to_gpa4_china_linear(50) == pytest.approx(0.0)

    def test_score_below_50_clipped_to_0(self) -> None:
        assert pct_to_gpa4_china_linear(30) == pytest.approx(0.0)

    def test_result_never_exceeds_4(self) -> None:
        assert pct_to_gpa4_china_linear(120) == pytest.approx(4.0)


# ── pct_to_gpa4_china_table ───────────────────────────────────────────────────

class TestPctToGpa4ChinaTable:
    def test_90_and_above_gives_4(self) -> None:
        assert pct_to_gpa4_china_table(95) == pytest.approx(4.0)
        assert pct_to_gpa4_china_table(90) == pytest.approx(4.0)

    def test_85_to_89_gives_3_7(self) -> None:
        assert pct_to_gpa4_china_table(87) == pytest.approx(3.7)

    def test_78_to_81_gives_3(self) -> None:
        assert pct_to_gpa4_china_table(79) == pytest.approx(3.0)

    def test_60_to_63_gives_1(self) -> None:
        assert pct_to_gpa4_china_table(61) == pytest.approx(1.0)

    def test_below_60_gives_0(self) -> None:
        assert pct_to_gpa4_china_table(55) == pytest.approx(0.0)
        assert pct_to_gpa4_china_table(0) == pytest.approx(0.0)

    def test_boundary_82_gives_3_3(self) -> None:
        assert pct_to_gpa4_china_table(82) == pytest.approx(3.3)


# ── standardize_gpa_to_4 ─────────────────────────────────────────────────────

class TestStandardizeGpaTo4:
    def test_already_on_4_scale(self) -> None:
        row = {"GPA": "3.5", "毕业学校": "University of Oxford"}
        raw, gpa4, status = standardize_gpa_to_4(row, "china_linear")
        assert status == "already_4_scale"
        assert float(gpa4) == pytest.approx(3.5)

    def test_chinese_pct_converted_linear(self) -> None:
        row = {"GPA": "80", "毕业学校": "北京大学"}
        raw, gpa4, status = standardize_gpa_to_4(row, "china_linear")
        assert status == "converted_from_100"
        assert float(gpa4) == pytest.approx(3.0)

    def test_chinese_pct_converted_table(self) -> None:
        row = {"GPA": "90", "毕业学校": "清华大学"}
        _, gpa4, status = standardize_gpa_to_4(row, "china_table")
        assert status == "converted_from_100"
        assert float(gpa4) == pytest.approx(4.0)

    def test_missing_gpa_returns_missing_status(self) -> None:
        row = {"GPA": "", "毕业学校": "北京大学"}
        _, _, status = standardize_gpa_to_4(row, "china_linear")
        assert status == "missing_or_invalid"

    def test_non_chinese_pct_not_converted(self) -> None:
        row = {"GPA": "75", "毕业学校": "University of Oxford"}
        _, _, status = standardize_gpa_to_4(row, "china_linear")
        assert status == "percent_non_china_not_converted"


# ── normalize_school_name ─────────────────────────────────────────────────────

class TestNormalizeSchoolName:
    def test_lowercases_name(self) -> None:
        assert normalize_school_name("Oxford") == "oxford"

    def test_removes_leading_the(self) -> None:
        assert normalize_school_name("The University of Oxford") == "university of oxford"

    def test_replaces_ampersand_with_and(self) -> None:
        assert normalize_school_name("Arts & Sciences") == "arts and sciences"

    def test_collapses_extra_whitespace(self) -> None:
        assert normalize_school_name("  MIT  ") == "mit"

    def test_removes_hyphens(self) -> None:
        assert normalize_school_name("Winston-Salem State") == "winston salem state"

    def test_empty_string_returns_empty(self) -> None:
        assert normalize_school_name("") == ""


# ── to_float ──────────────────────────────────────────────────────────────────

class TestToFloat:
    def test_valid_integer_string(self) -> None:
        assert to_float("1000") == pytest.approx(1000.0)

    def test_valid_float_string(self) -> None:
        assert to_float("3.14") == pytest.approx(3.14)

    def test_empty_string_returns_zero(self) -> None:
        assert to_float("") == pytest.approx(0.0)

    def test_non_numeric_returns_zero(self) -> None:
        assert to_float("N/A") == pytest.approx(0.0)

    def test_whitespace_stripped(self) -> None:
        assert to_float("  500  ") == pytest.approx(500.0)


# ── calc_total_cost_usd ───────────────────────────────────────────────────────

class TestCalcTotalCostUsd:
    def test_basic_calculation(self) -> None:
        row = {
            "Duration_Years": "1",
            "Tuition_USD": "20000",
            "Rent_USD": "1000",
            "Visa_Fee_USD": "350",
            "Insurance_USD": "600",
        }
        # 20000*1 + 1000*12*1 + 350 + 600 = 32950
        assert calc_total_cost_usd(row) == pytest.approx(32_950.0)

    def test_two_year_duration_doubles_tuition_and_rent(self) -> None:
        row = {
            "Duration_Years": "2",
            "Tuition_USD": "10000",
            "Rent_USD": "500",
            "Visa_Fee_USD": "0",
            "Insurance_USD": "0",
        }
        # 10000*2 + 500*12*2 = 20000 + 12000 = 32000
        assert calc_total_cost_usd(row) == pytest.approx(32_000.0)

    def test_missing_fields_default_to_zero(self) -> None:
        row = {
            "Duration_Years": "1",
            "Tuition_USD": "50000",
            "Rent_USD": "",
            "Visa_Fee_USD": "",
            "Insurance_USD": "",
        }
        assert calc_total_cost_usd(row) == pytest.approx(50_000.0)


# ── build_school_counter ──────────────────────────────────────────────────────

class TestBuildSchoolCounter:
    def test_counts_correctly(self) -> None:
        rows = [
            {"school": "MIT"},
            {"school": "MIT"},
            {"school": "Oxford"},
        ]
        missing, counter = build_school_counter(rows, "school")
        assert counter["MIT"] == 2
        assert counter["Oxford"] == 1

    def test_empty_school_counted_as_missing(self) -> None:
        rows = [{"school": ""}, {"school": "MIT"}]
        missing, counter = build_school_counter(rows, "school")
        assert missing == 1
        assert "MIT" in counter

    def test_returns_counter_type(self) -> None:
        rows = [{"school": "A"}, {"school": "B"}]
        _, counter = build_school_counter(rows, "school")
        assert isinstance(counter, Counter)


# ── unique_school_list ────────────────────────────────────────────────────────

class TestUniqueSchoolList:
    def test_deduplicates_preserving_order(self) -> None:
        rows = [
            {"school": "MIT"},
            {"school": "Oxford"},
            {"school": "MIT"},
        ]
        result = unique_school_list(rows, "school")
        assert result == ["MIT", "Oxford"]

    def test_empty_values_excluded(self) -> None:
        rows = [{"school": ""}, {"school": "MIT"}]
        result = unique_school_list(rows, "school")
        assert "" not in result
        assert "MIT" in result

    def test_empty_input_returns_empty_list(self) -> None:
        assert unique_school_list([], "school") == []


# ── best_candidates ───────────────────────────────────────────────────────────

class TestBestCandidates:
    def test_exact_match_returns_score_1(self) -> None:
        candidates = best_candidates("MIT", ["MIT", "Oxford"], top_n=1, cutoff=0.5)
        assert len(candidates) == 1
        assert candidates[0][0] == "MIT"
        assert candidates[0][1] == pytest.approx(1.0)

    def test_no_match_above_cutoff_returns_empty(self) -> None:
        result = best_candidates("MIT", ["Oxford", "Cambridge"], top_n=3, cutoff=0.9)
        assert result == []

    def test_results_sorted_by_score_descending(self) -> None:
        pool = ["University of Manchester", "University of Michigan", "Stanford"]
        results = best_candidates(
            "University of Michigan", pool, top_n=3, cutoff=0.5
        )
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limits_results(self) -> None:
        pool = ["University of Manchester", "University of Michigan",
                "University of Melbourne", "University of Miami"]
        results = best_candidates("University of M", pool, top_n=2, cutoff=0.5)
        assert len(results) <= 2

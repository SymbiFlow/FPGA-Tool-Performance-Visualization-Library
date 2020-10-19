""" Tests for Processors """

import pandas as pd
from pandas.testing import (
    assert_frame_equal, assert_series_equal, assert_index_equal
)

from ftpvl.processors import *

from ftpvl.evaluation import Evaluation


class TestProcessor:
    """
    Testing by partition.

    MinusOne()
    StandardizeTypes()
    CleanDuplicates()
    AddNormalizedColumn()
    ExpandToolchain()
    Reindex()
    SortIndex()
    NormalizeAround()
    Normalize()
    FilterByIndex()
    RemoveByIndex()
    Aggregate()
    GeomeanAggregate()
    CompareToFirst()
    """

    def test_minusone(self):
        """ Test whether all values are correctly changed """

        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        eval1 = Evaluation(df, eval_id=10)

        result = eval1.get_df()["a"]
        expected = pd.Series([1, 2, 3, 4, 5], name="a")

        assert_series_equal(result, expected)

        pipeline = [MinusOne()]
        result_processed = eval1.process(pipeline)
        result_df = result_processed.get_df()["a"]
        expected_df = pd.Series([0, 1, 2, 3, 4], name="a")

        assert_series_equal(result_df, expected_df)

        assert result_processed.get_eval_id() == 10


    def test_standardizetypes(self):
        """ Test whether types are standardized """
        types = {"a": float}
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        eval1 = Evaluation(df, eval_id=10)

        assert eval1.get_df().dtypes["a"] == int

        pipeline = [StandardizeTypes(types)]

        result = eval1.process(pipeline)

        assert result.get_df().dtypes["a"] == float

        assert result.get_eval_id() == 10

    def test_cleanduplicates_no_duplicates(self):
        """ Test for evaluation that has no duplicates in specified column """
        df = pd.DataFrame(
            [
                {"a": 1, "b": 1, "c": 5},
                {"a": 1, "b": 2, "c": 4},
                {"a": 3, "b": 3, "c": 3},
                {"a": 4, "b": 4, "c": 2},
                {"a": 5, "b": 5, "c": 1},
            ]
        )
        eval1 = Evaluation(df, eval_id=10)

        # test no duplicates
        pipeline = [CleanDuplicates(["b"])]
        result = eval1.process(pipeline)
        assert_frame_equal(result.get_df(), df, check_like=True)

        assert result.get_eval_id() == 10

    def test_cleanduplicates_one_col(self):
        """ Test for evaluation that has duplicate in one column """
        df = pd.DataFrame(
            [
                {"a": 1, "b": 1, "c": 5},
                {"a": 1, "b": 2, "c": 4},
                {"a": 3, "b": 3, "c": 3},
                {"a": 4, "b": 4, "c": 2},
                {"a": 5, "b": 5, "c": 1},
            ]
        )
        eval1 = Evaluation(df)

        pipeline = [CleanDuplicates(["a"])]
        result = eval1.process(pipeline).get_df()
        expected = df.drop(1)

        assert_frame_equal(result, expected, check_like=True)

    def test_cleanduplicates_multi_col(self):
        """
        Test for evaluation that doesn't have duplicates when comparing
        more than one column
        """
        df = pd.DataFrame(
            [
                {"a": 1, "b": 1, "c": 5},
                {"a": 1, "b": 2, "c": 4},
                {"a": 3, "b": 3, "c": 3},
                {"a": 4, "b": 4, "c": 2},
                {"a": 5, "b": 5, "c": 1},
            ]
        )
        eval1 = Evaluation(df)
        pipeline = [CleanDuplicates(["a", "b"])]
        result2 = eval1.process(pipeline).get_df()
        assert_frame_equal(result2, df, check_like=True)

    def test_cleanduplicates_sorting(self):
        """
        Test by sorting before removing duplicate.
        """
        df = pd.DataFrame(
            [
                {"a": 1, "b": 1, "c": 5},
                {"a": 1, "b": 2, "c": 4},
                {"a": 3, "b": 3, "c": 3},
                {"a": 4, "b": 4, "c": 2},
                {"a": 5, "b": 5, "c": 1},
            ]
        )
        eval1 = Evaluation(df)

        pipeline = [CleanDuplicates(["a"], ["c"])]
        result = eval1.process(pipeline).get_df()  # will remove idx 1
        expected = df.drop(1)
        assert_frame_equal(result, expected, check_like=True)

        pipeline = [CleanDuplicates(["a"], ["c"], reverse_sort=True)]
        result = eval1.process(pipeline).get_df()  # will remove idx 0
        expected = df.drop(0).sort_index(level=0, ascending=False)
        assert_frame_equal(result, expected, check_like=True)

    def test_addnormalizedcolumn(self):
        """ Test whether normalized column is added """
        df = pd.DataFrame(
            [
                {"group": "a", "value": 10},
                {"group": "a", "value": 5},
                {"group": "a", "value": 3},
                {"group": "b", "value": 100},
                {"group": "b", "value": 31},
            ]
        )
        eval1 = Evaluation(df, eval_id=10)

        pipeline = [AddNormalizedColumn("group", "value", "normalized")]
        result = eval1.process(pipeline)
        expected = pd.DataFrame(
            [
                {"group": "a", "value": 10, "normalized": 1.0},
                {"group": "a", "value": 5, "normalized": 0.5},
                {"group": "a", "value": 3, "normalized": 0.3},
                {"group": "b", "value": 100, "normalized": 1.0},
                {"group": "b", "value": 31, "normalized": 0.31},
            ]
        )

        assert_frame_equal(result.get_df(), expected)

        assert result.get_eval_id() == 10

    def test_addnormalizedcolumn_direction(self):
        """ Test whether normalized column direction parameter works """
        df = pd.DataFrame(
            [
                {"group": "a", "value": 10},
                {"group": "a", "value": 5},
                {"group": "a", "value": 3},
                {"group": "b", "value": 100},
                {"group": "b", "value": 31},
            ]
        )
        eval1 = Evaluation(df, eval_id=10)

        # direction is -1 => normalize around minimum
        pipeline = [AddNormalizedColumn("group", "value", "normalized", Direction.MINIMIZE)]
        result = eval1.process(pipeline)
        expected = pd.DataFrame(
            [
                {"group": "a", "value": 10, "normalized": 10/3},
                {"group": "a", "value": 5, "normalized": 5/3},
                {"group": "a", "value": 3, "normalized": 1.0},
                {"group": "b", "value": 100, "normalized": 100/31},
                {"group": "b", "value": 31, "normalized": 1.0},
            ]
        )

        assert_frame_equal(result.get_df(), expected)

        assert result.get_eval_id() == 10

    def test_expandcolumn(self):
        """ Test whether the column is expanded """
        df = pd.DataFrame(
            [
                {"group": "a", "value": 10},
                {"group": "a", "value": 5},
                {"group": "a", "value": 3},
                {"group": "b", "value": 100},
                {"group": "b", "value": 31},
            ]
        )
        eval1 = Evaluation(df, eval_id=10)

        mapping = {
            "a": ("a", "x"),
            "b": ("b", "y"),
        }

        pipeline = [ExpandColumn("group", ["group1", "group2"], mapping)]
        result = eval1.process(pipeline)
        expected = pd.DataFrame(
            [
                {"group": "a", "group1": "a", "group2": "x", "value": 10},
                {"group": "a", "group1": "a", "group2": "x", "value": 5},
                {"group": "a", "group1": "a", "group2": "x", "value": 3},
                {"group": "b", "group1": "b", "group2": "y", "value": 100},
                {"group": "b", "group1": "b", "group2": "y", "value": 31},
            ]
        )

        assert_frame_equal(result.get_df(), expected, check_like=True)
        assert result.get_eval_id() == 10

    def test_reindex(self):
        """ Test whether the dataframe was reindexed """
        df = pd.DataFrame(
            [
                {"group": "a", "key": "a", "value": 10},
                {"group": "a", "key": "b", "value": 5},
                {"group": "a", "key": "c", "value": 3},
                {"group": "b", "key": "d", "value": 100},
                {"group": "b", "key": "e", "value": 31},
            ]
        )
        eval1 = Evaluation(df, eval_id=10)

        pipeline = [Reindex(["value"])]
        result = eval1.process(pipeline)
        expected_index = pd.Index([10, 5, 3, 100, 31], name="value")
        assert_index_equal(result.get_df().index, expected_index)
        assert result.get_eval_id() == 10

        pipeline = [Reindex(["group", "key"])]
        result = eval1.process(pipeline)
        arrays = [["a", "a", "a", "b", "b"], ["a", "b", "c", "d", "e"]]
        expected_index = pd.MultiIndex.from_arrays(arrays, names=("group", "key"))
        assert_index_equal(result.get_df().index, expected_index)
        assert result.get_eval_id() == 10

    def test_sortindex(self):
        """ Test whether the dataframe is sorted by index """
        df = pd.DataFrame(
            data = [
                {"group": "a", "value": 10},
                {"group": "a", "value": 5},
                {"group": "a", "value": 3},
                {"group": "b", "value": 100},
                {"group": "b", "value": 31},
            ],
            index = pd.Index([
                5,
                4,
                3,
                2,
                1,
            ], name="idx")
        )

        eval1 = Evaluation(df, eval_id=10)

        pipeline = [SortIndex(["idx"])]
        result = eval1.process(pipeline)
        expected = pd.DataFrame(
            data = [
                {"group": "b", "value": 31},
                {"group": "b", "value": 100},
                {"group": "a", "value": 3},
                {"group": "a", "value": 5},
                {"group": "a", "value": 10},
            ],
            index = pd.Index([
                1,
                2,
                3,
                4,
                5,
            ], name="idx")
        )
        assert_frame_equal(result.get_df(), expected)
        assert result.get_eval_id() == 10

    def test_normalizearound(self):
        """
        Test whether all values are normalized around a certain (set of) rows
        """
        arrays = [
            ["blinky", "blinky", "blinky", "ibex", "ibex"],
            ["yosys", "yosys", "vivado", "yosys", "vivado"],
        ]
        index = pd.MultiIndex.from_arrays(arrays, names=("project", "synthesis_tool"))
        df = pd.DataFrame(
            data=[
                {"group": "b", "value": 0},
                {"group": "b", "value": 50},
                {"group": "b", "value": 100},
                {"group": "a", "value": 0},
                {"group": "a", "value": 10},
            ],
            index=index
        )
        eval1 = Evaluation(df, eval_id=10)

        normalize_direction = {
            "value": Direction.MINIMIZE
        }
        pipeline = [NormalizeAround(
            normalize_direction,
            group_by="project",
            idx_name="synthesis_tool",
            idx_value="vivado"
        )]

        result = eval1.process(pipeline)
        expected = pd.DataFrame(
            data=[
                {"group": "b", "value": 0},
                {"group": "b", "value": 0.25},
                {"group": "b", "value": 0.5},
                {"group": "a", "value": 0},
                {"group": "a", "value": 0.5},
            ],
            index=index
        )
        assert_frame_equal(result.get_df(), expected)
        assert result.get_eval_id() == 10

    def test_normalizearound_negated(self):
        """
        Test whether all values are normalized in the correct direction based
        on the negation.
        """
        arrays = [
            ["blinky", "blinky", "blinky", "ibex", "ibex"],
            ["yosys", "yosys", "vivado", "yosys", "vivado"],
        ]
        index = pd.MultiIndex.from_arrays(arrays, names=("project", "synthesis_tool"))
        df = pd.DataFrame(
            data=[
                {"group": "b", "value": 0},
                {"group": "b", "value": 50},
                {"group": "b", "value": 100},
                {"group": "a", "value": 0},
                {"group": "a", "value": 10},
            ],
            index=index
        )
        eval1 = Evaluation(df, eval_id=10)

        normalize_direction = {
            "value": Direction.MAXIMIZE
        }
        pipeline = [NormalizeAround(
            normalize_direction,
            group_by="project",
            idx_name="synthesis_tool",
            idx_value="vivado"
        )]

        result = eval1.process(pipeline)
        expected = pd.DataFrame(
            data=[
                {"group": "b", "value": 1},
                {"group": "b", "value": 0.75},
                {"group": "b", "value": 0.5},
                {"group": "a", "value": 1},
                {"group": "a", "value": 0.5},
            ],
            index=index
        )
        assert_frame_equal(result.get_df(), expected)
        assert result.get_eval_id() == 10

    def test_normalize(self):
        """
        Test whether all values are normalized
        """
        df = pd.DataFrame(
            data=[
                {"group": "b", "value": -50},
                {"group": "b", "value": 50},
                {"group": "b", "value": 100},
                {"group": "a", "value": 0},
                {"group": "a", "value": 10},
            ]
        )
        eval1 = Evaluation(df, eval_id=10)

        normalize_direction = {
            "value": Direction.MINIMIZE
        }
        pipeline = [Normalize(normalize_direction)]

        result = eval1.process(pipeline)
        expected = pd.DataFrame(
            data=[
                {"group": "b", "value": 0.25},
                {"group": "b", "value": 0.75},
                {"group": "b", "value": 1.0},
                {"group": "a", "value": 0.5},
                {"group": "a", "value": 0.55},
            ],
        )

        assert_frame_equal(result.get_df(), expected)
        assert result.get_eval_id() == 10

    def test_normalize_negated(self):
        """
        Test whether all values are normalized
        """
        df = pd.DataFrame(
            data=[
                {"group": "b", "value": -50},
                {"group": "b", "value": 50},
                {"group": "b", "value": 100},
                {"group": "a", "value": 0},
                {"group": "a", "value": 10},
            ]
        )
        eval1 = Evaluation(df)

        normalize_direction = {
            "value": -1
        }
        pipeline = [Normalize(normalize_direction)]

        result = eval1.process(pipeline).get_df()
        expected = pd.DataFrame(
            data=[
                {"group": "b", "value": 0.75},
                {"group": "b", "value": 0.25},
                {"group": "b", "value": 0},
                {"group": "a", "value": 0.5},
                {"group": "a", "value": 0.45},
            ],
        )

        assert_frame_equal(result, expected)

    def test_relativediff(self):
        """
        Test if difference is correct
        """
        a = pd.DataFrame(
            data=[
                {"a": 2, "b": 2},
                {"a": 5, "b": 10},
            ]
        )
        b = pd.DataFrame(
            data=[
                {"a": 4, "b": 1},
                {"a": 20, "b": 1},
            ]
        )

        a_eval = Evaluation(a)
        b_eval = Evaluation(b)

        diff = b_eval.process([RelativeDiff(a_eval)])
        result = diff.get_df()

        expected = pd.DataFrame(
            data=[
                {"a": 1.0, "b": -0.5},
                {"a": 3.0, "b": -0.9},
            ]
        )

        assert_frame_equal(expected, result)

    def test_filterbyindex_multindex(self):
        """ tests if filtering by index works for multi-index dataframe """
        # test dataframe
        # {"group": "a", "key": "a", "value": 10},
        # {"group": "a", "key": "b", "value": 5},
        # {"group": "a", "key": "c", "value": 3},
        # {"group": "b", "key": "d", "value": 100},
        # {"group": "b", "key": "e", "value": 31}

        idx_arrays = [["a", "a", "a", "b", "b"], ["a", "b", "c", "d", "e"]]
        index = pd.MultiIndex.from_arrays(idx_arrays, names=("group", "key"))
        df = pd.DataFrame({"value": [10, 5, 3, 100, 31]}, index=index)
        eval1 = Evaluation(df, eval_id=10)

        # filter by first index
        pipeline = [FilterByIndex("group", "a")]
        result = eval1.process(pipeline)

        expected_index = pd.Index(["a", "b", "c"], name="key")
        expected_df = pd.DataFrame({"value": [10, 5, 3]}, index=expected_index)

        assert_frame_equal(result.get_df(), expected_df)
        assert result.get_eval_id() == 10

    def test_filterbyindex_singleindex(self):
        """ tests if filtering by index works for single-index dataframe """
        # test dataframe
        # {"group": "a", "key": "a", "value": 10},
        # {"group": "a", "key": "b", "value": 5},
        # {"group": "a", "key": "c", "value": 3},
        # {"group": "b", "key": "d", "value": 100},
        # {"group": "b", "key": "e", "value": 31}

        idx_array = ["a", "a", "a", "b", "b"]
        index = pd.Index(idx_array, name="key")
        df = pd.DataFrame({"value": [10, 5, 3, 100, 31]}, index=index)
        eval1 = Evaluation(df, eval_id=10)

        # filter by first index
        pipeline = [FilterByIndex("key", "a")]
        result = eval1.process(pipeline)
        expected_index = pd.Index(["a", "a", "a"], name="key")
        expected_df = pd.DataFrame({"value": [10, 5, 3]}, index=expected_index)

        assert_frame_equal(result.get_df(), expected_df)
        assert result.get_eval_id() == 10
        """ tests if filtering by index works for single-index dataframe """

    def test_removebyindex_multindex(self):
        """ tests if removing by index works for multi-index dataframe """
        # test dataframe
        # {"group": "a", "key": "a", "value": 10},
        # {"group": "a", "key": "b", "value": 5},
        # {"group": "a", "key": "c", "value": 3},
        # {"group": "b", "key": "d", "value": 100},
        # {"group": "b", "key": "e", "value": 31}

        idx_arrays = [["a", "a", "a", "b", "b"], ["a", "b", "c", "d", "e"]]
        index = pd.MultiIndex.from_arrays(idx_arrays, names=("group", "key"))
        df = pd.DataFrame({"value": [10, 5, 3, 100, 31]}, index=index)
        eval1 = Evaluation(df, eval_id=10)

        expected_idx_arrays = [["b", "b"], ["d", "e"]]
        expected_index = pd.MultiIndex.from_arrays(expected_idx_arrays, names=("group", "key"))
        expected_df = pd.DataFrame({"value": [100, 31]}, index=expected_index)

        # filter by first index
        pipeline = [RemoveByIndex("group", ["a"])]
        result_group = eval1.process(pipeline)

        # filter by first index
        pipeline = [RemoveByIndex("key", ["a", "b", "c"])]
        result_key = eval1.process(pipeline)

        assert_frame_equal(result_group.get_df(), expected_df)
        assert_frame_equal(result_key.get_df(), expected_df)

        assert result_group.get_eval_id() == 10
        assert result_key.get_eval_id() == 10

    def test_removebyindex_singleindex(self):
        """ tests if removing by index works for single-index dataframe """
        # test dataframe
        # {"key": "a", "value": 10},
        # {"key": "a", "value": 5},
        # {"key": "a", "value": 3},
        # {"key": "b", "value": 100},
        # {"key": "b", "value": 31}

        idx_array = ["a", "a", "a", "b", "b"]
        index = pd.Index(idx_array, name="key")
        df = pd.DataFrame({"value": [10, 5, 3, 100, 31]}, index=index)
        eval1 = Evaluation(df, eval_id=10)

        expected_idx_array = ["b", "b"]
        expected_index = pd.Index(expected_idx_array, name="key")
        expected_df = pd.DataFrame({"value": [100, 31]}, index=expected_index)

        # filter by first index
        pipeline = [RemoveByIndex("key", ["a"])]
        result = eval1.process(pipeline)

        assert_frame_equal(result.get_df(), expected_df)
        assert result.get_eval_id() == 10

    def test_aggregate(self):
        """ Test aggregate processor with custom aggregator functions """
        df = pd.DataFrame(
            [
                {"a": 1, "b": 1, "c": 5},
                {"a": 1, "b": 2, "c": 4},
                {"a": 3, "b": 3, "c": 3},
                {"a": 4, "b": 4, "c": 2},
                {"a": 5, "b": 5, "c": 1},
            ]
        )
        eval1 = Evaluation(df, eval_id=20)

        pipeline = [Aggregate(lambda x: x.sum())]
        result = eval1.process(pipeline)

        expected_df = pd.DataFrame(
            [
                {"a": 14, "b": 15, "c": 15}
            ]
        )
        assert_frame_equal(result.get_df(), expected_df)
        assert eval1.get_eval_id() == 20

        pipeline2 = [Aggregate(lambda x: x.product())]
        result2 = eval1.process(pipeline2)

        expected_df2 = pd.DataFrame(
            [
                {"a": 60, "b": 120, "c": 120}
            ]
        )
        assert_frame_equal(result2.get_df(), expected_df2)
        assert result2.get_eval_id() == 20

    def test_aggregate_exclude_nonnumeric(self):
        """ Check if aggregate processor excludes fields that are non-numeric """
        df = pd.DataFrame(
            [
                {"a": 1, "b": 1, "c": "a"},
                {"a": 1, "b": 2, "c": "b"},
                {"a": 3, "b": 3, "c": "c"},
                {"a": 4, "b": 4, "c": "d"},
                {"a": 5, "b": 5, "c": "e"},
            ]
        )
        eval1 = Evaluation(df, eval_id=20)

        pipeline = [Aggregate(lambda x: x.sum())]
        result = eval1.process(pipeline)

        expected_df = pd.DataFrame(
            [
                {"a": 14, "b": 15}
            ]
        )
        assert_frame_equal(result.get_df(), expected_df)
        assert eval1.get_eval_id() == 20

    def test_geomean_aggregate(self):
        """ Test built-in geomean aggregator """
        df = pd.DataFrame(
            [
                {"a": 1, "b": 1, "c": 5},
                {"a": 1, "b": 2, "c": 4},
                {"a": 3, "b": 3, "c": 3},
                {"a": 4, "b": 4, "c": 2},
                {"a": 5, "b": 5, "c": 1},
            ]
        )
        eval1 = Evaluation(df, eval_id=20)

        pipeline = [GeomeanAggregate()]
        eval1 = eval1.process(pipeline)

        expected_a = (1 * 1 * 3 * 4 * 5) ** (1/5)
        expected_b = expected_c = (1 * 2 * 3 * 4 * 5) ** (1/5)
        expected_df = pd.DataFrame(
            [
                {"a": expected_a, "b": expected_b, "c": expected_c}
            ]
        )
        assert_frame_equal(eval1.get_df(), expected_df)
        assert eval1.get_eval_id() == 20

    def test_comparetofirst(self):
        """ Test if CompareToFirst works with default params """
        df = pd.DataFrame(
            [
                {"a": 1, "b": 5},
                {"a": 2, "b": 4},
                {"a": 3, "b": 3},
                {"a": 4, "b": 2},
                {"a": 5, "b": 1},
            ]
        )
        eval1 = Evaluation(df, eval_id=20)

        direction = {
            "a": Direction.MAXIMIZE,
            "b": Direction.MAXIMIZE
        }

        pipeline = [CompareToFirst(direction)]
        eval1 = eval1.process(pipeline)

        expected_df = pd.DataFrame(
            [
                {"a": 1, "a.relative": 1.0/1, "b": 5, "b.relative": 5.0/5},
                {"a": 2, "a.relative": 2.0/1, "b": 4, "b.relative": 4.0/5},
                {"a": 3, "a.relative": 3.0/1, "b": 3, "b.relative": 3.0/5},
                {"a": 4, "a.relative": 4.0/1, "b": 2, "b.relative": 2.0/5},
                {"a": 5, "a.relative": 5.0/1, "b": 1, "b.relative": 1.0/5},
            ]
        )
        assert_frame_equal(eval1.get_df(), expected_df)
        assert eval1.get_eval_id() == 20

    def test_comparetofirst_dir_subset(self):
        """ Test if CompareToFirst works with different direction and subset"""
        df = pd.DataFrame(
            [
                {"a": 1, "b": 5},
                {"a": 2, "b": 4},
                {"a": 3, "b": 3},
                {"a": 4, "b": 2},
                {"a": 5, "b": 1},
            ]
        )
        eval1 = Evaluation(df, eval_id=20)

        direction = {
            "a": Direction.MINIMIZE
        }

        pipeline = [CompareToFirst(direction)]
        eval1 = eval1.process(pipeline)

        expected_df = pd.DataFrame(
            [
                {"a": 1, "a.relative": 1.0/1},
                {"a": 2, "a.relative": 1.0/2},
                {"a": 3, "a.relative": 1.0/3},
                {"a": 4, "a.relative": 1.0/4},
                {"a": 5, "a.relative": 1.0/5},
            ]
        )
        assert_frame_equal(eval1.get_df(), expected_df)
        assert eval1.get_eval_id() == 20

    def test_comparetofirst_suffix(self):
        """ Test if CompareToFirst works with different suffix """
        df = pd.DataFrame(
            [
                {"a": 1, "b": 5},
                {"a": 2, "b": 4},
                {"a": 3, "b": 3},
                {"a": 4, "b": 2},
                {"a": 5, "b": 1},
            ]
        )
        eval1 = Evaluation(df, eval_id=20)

        direction = {
            "a": Direction.MAXIMIZE,
            "b": Direction.MAXIMIZE
        }

        pipeline = [CompareToFirst(direction, suffix=".diff")]
        eval1 = eval1.process(pipeline)

        expected_df = pd.DataFrame(
            [
                {"a": 1, "a.diff": 1.0/1, "b": 5, "b.diff": 5.0/5},
                {"a": 2, "a.diff": 2.0/1, "b": 4, "b.diff": 4.0/5},
                {"a": 3, "a.diff": 3.0/1, "b": 3, "b.diff": 3.0/5},
                {"a": 4, "a.diff": 4.0/1, "b": 2, "b.diff": 2.0/5},
                {"a": 5, "a.diff": 5.0/1, "b": 1, "b.diff": 1.0/5},
            ]
        )
        assert_frame_equal(eval1.get_df(), expected_df)
        assert eval1.get_eval_id() == 20

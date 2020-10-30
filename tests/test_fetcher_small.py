# pylint: disable=invalid-name, line-too-long

""" Tests for Evaluation class """
import unittest

import pandas as pd
import requests_mock

from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal

from ftpvl.fetchers import HydraFetcher, JSONFetcher, GoogleStorageFetcher

class TestGoogleStorageFetcherSmall(unittest.TestCase):
    """
    Testing by partition.

    GoogleStorageFetcher:
        __init__(eval_num, mapping_dict, hydra_clock_names)
        get_evaluation()
            different eval_num
                relative, absolute
            different mapping
                exclusion, renaming
            pagination
            with/without board and date information
    """

    def test_googlestoragefetcher_init(self):
        """
        Calling init should save the arguments as an instance variable.
        """
        fetcher = GoogleStorageFetcher(
            project="fpga-tool-perf",
            jobset="presubmit",
            eval_num=0,
            mapping={"a": "c", "b": "d"},
            hydra_clock_names=["clk", "clk1"]
        )

        self.assertEqual(fetcher.project, "fpga-tool-perf")
        self.assertEqual(fetcher.jobset, "presubmit")
        self.assertEqual(fetcher.eval_num, 0)
        self.assertEqual(fetcher.mapping, {"a": "c", "b": "d"})
        self.assertEqual(fetcher.hydra_clock_names, ["clk", "clk1"])

    def test_googlestoragefetcher_get_evaluation_eval_num(self):
        """
        get_evaluation() should return an Evaluation corresponding to the small
        dataset and the specified eval_num.

        Tests eval_num functionality along with pagination
        """
        project = "fpga-tool-perf"
        jobset = "presubmit"
        eval_num = 100
        next_page_token = ""
        with requests_mock.Mocker() as m:
            evals_fn = 'tests/sample_data/evals_googlestorage_small.json'
            evals_fn2 = 'tests/sample_data/evals_googlestorage_small2.json'
            evals_url = f'https://www.googleapis.com/storage/v1/b/{project}/o?delimiter=meta.json&prefix=artifacts/prod/foss-fpga-tools/fpga-tool-perf/{jobset}/{eval_num}'
            base_download_path = f"https://storage.googleapis.com/{project}/"
            build_num = 0
            import json
            # setup evals request mock
            with open(evals_fn, "r") as f:
                json_data = f.read()
                json_dict = json.loads(json_data)
                next_page_token = "&pageToken=" + json_dict["nextPageToken"]
                m.get(evals_url, text=json_data)
                for prefix in json_dict['prefixes']:
                    payload = {
                        "build_num": build_num,
                        "date": "2020-07-17T22:12:40",
                        "board": "arty",
                        "toolchain": { "vivado": { "pr_tool": "vivado", "synthesis_tool": "vivado" } }
                    }
                    meta_url = f'{base_download_path}{prefix}'
                    m.get(meta_url, json=payload)
                    build_num += 1


            # setup evals next page request mock
            with open(evals_fn2, "r") as f:
                json_data = f.read()
                json_dict = json.loads(json_data)
                m.get(evals_url + next_page_token, text=json_data)
                for prefix in json_dict['prefixes']:
                    payload = {
                        "build_num": build_num,
                        "date": "2020-07-17T22:12:40",
                        "board": "arty",
                        "toolchain": { "vivado": { "pr_tool": "vivado", "synthesis_tool": "vivado" } }
                    }
                    meta_url = f'{base_download_path}{prefix}'
                    m.get(meta_url, json=payload)
                    build_num += 1

            # if mapping is not defined, should not remap, but it should split toolchain into pr/synthesis tool
            gsf = GoogleStorageFetcher(
                project="fpga-tool-perf",
                jobset="presubmit",
                eval_num=eval_num)
            result = gsf.get_evaluation().get_df()

            import sys
            print("RESULT", result, file=open("/tmp/err", "w"))

            try:
                expected = pd.DataFrame({
                    "build_num": [x for x in range(0, 22)],
                    "date": ["2020-07-17T22:12:40" for _ in range(22)],
                    "board": ["arty" for _ in range(22)],
                    "toolchain": ["vivado" for _ in range(22)],
                    "pr_tool": ["vivado" for _ in range(22)],
                    "synthesis_tool": ["vivado" for _ in range(22)],
                    "clock_name": ["" for _ in range(22)],
                    "actual": [0.0 for _ in range(22)],
                    "requested": [0.0 for _ in range(22)],
                    "met": [False for _ in range(22)],
                    "hold_violation": [0.0 for _ in range(22)],
                    "setup_violation": [0.0 for _ in range(22)],
                    })
            except Exception as e:
                print(e, file=open("/tmp/err", "a"))
            assert_frame_equal(result, expected)

            eval_id = gsf.get_evaluation().get_eval_id()
            assert eval_id == eval_num

class TestHydraFetcherSmall(unittest.TestCase):
    """
    Testing by partition.

    HydraFetcher:
        __init__(eval_num, mapping_dict, hydra_clock_names)
        get_evaluation()
            different eval_num
                relative, absolute
            different mapping
                exclusion, renaming
            pagination
            with/without board and date information
    """

    def test_hydrafetcher_init(self):
        """
        Calling init should save the arguments as an instance variable.
        """
        fetcher = HydraFetcher(
            project="dusty",
            jobset="fpga-tool-perf",
            eval_num=0,
            mapping={"a": "c", "b": "d"},
            hydra_clock_names=["clk", "clk1"]
        )

        self.assertEqual(fetcher.eval_num, 0)
        self.assertEqual(fetcher.mapping, {"a": "c", "b": "d"})
        self.assertEqual(fetcher.hydra_clock_names, ["clk", "clk1"])

    def test_hydrafetcher_get_evaluation_eval_num(self):
        """
        get_evaluation() should return an Evaluation corresponding to the small
        dataset and the specified eval_num.

        Tests different eval_num functionality along with pagination
        """
        with requests_mock.Mocker() as m:
            evals_fn = 'tests/sample_data/evals.small.json'
            evals_fn2 = 'tests/sample_data/evals.small.2.json'
            evals_url = 'https://hydra.vtr.tools/jobset/dusty/fpga-tool-perf/evals'

            build_fn = 'tests/sample_data/build.small.json'

            # setup /evals request mock
            with open(evals_fn, "r") as f:
                json_data = f.read()
                m.get(evals_url, text=json_data)

            # setup /evals?page=2 request mock
            with open(evals_fn2, "r") as f:
                json_data = f.read()
                m.get(evals_url + "?page=2", text=json_data)

            # setup /build and /meta.json request mock
            with open(build_fn, "r") as f:
                json_data = f.read()

                for build_num in range(24):
                    # /build/:buildid
                    build_url = f'https://hydra.vtr.tools/build/{build_num}'
                    m.get(build_url, text=json_data)

                    # /meta.json
                    meta_url = f'https://hydra.vtr.tools/build/{build_num}/download/5/meta.json'
                    payload = {
                        "build_num": build_num,
                        "date": "2020-07-17T22:12:40",
                        "board": "arty"
                    }
                    m.get(meta_url, json=payload) # setup /meta.json request mock

            # run tests on different eval_num
            for eval_num in range(0, 6):
                with self.subTest(eval_num=eval_num):
                    # if mapping is not defined, should not remap
                    hf = HydraFetcher(
                        project="dusty",
                        jobset="fpga-tool-perf",
                        eval_num=eval_num)
                    result = hf.get_evaluation().get_df()

                    expected = pd.DataFrame({
                        "build_num": [x for x in range(eval_num * 4, eval_num * 4 + 4)],
                        "date": ["2020-07-17T22:12:40" for _ in range(4)],
                        "board": ["arty" for _ in range(4)]
                        })
                    assert_frame_equal(result, expected)

                    eval_id = hf.get_evaluation().get_eval_id()
                    assert eval_id == eval_num + 1

    def test_hydrafetcher_get_evaluation_icebreaker_old(self):
        """
        get_evaluation() should return an Evaluation corresponding to the small
        dataset and the specified eval_num.

        Tests legacy icebreaker support, where icebreaker boards report
        max frequency in MHz instead of Hz.
        """
        with requests_mock.Mocker() as m:
            evals_fn = 'tests/sample_data/evals.small.json'
            evals_url = 'https://hydra.vtr.tools/jobset/dusty/fpga-tool-perf/evals'

            build_fn = 'tests/sample_data/build.small.json'

            # setup /evals request mock
            with open(evals_fn, "r") as f:
                json_data = f.read()
                m.get(evals_url, text=json_data)

            # setup /build and /meta.json request mock
            with open(build_fn, "r") as f:
                json_data = f.read()

                for build_num in range(12):
                    # /build/:buildid
                    build_url = f'https://hydra.vtr.tools/build/{build_num}'
                    m.get(build_url, text=json_data)

                    # /meta.json
                    meta_url = f'https://hydra.vtr.tools/build/{build_num}/download/5/meta.json'
                    payload = {
                        "build_num": build_num,
                        "date": "2020-07-30T22:12:40",
                        "board": "icebreaker",
                        "max_freq": 81.05
                    }
                    m.get(meta_url, json=payload) # setup /meta.json request mock

            # run tests on different eval_num
            for eval_num in range(0, 3):
                with self.subTest(eval_num=eval_num):
                    # if mapping is not defined, should not remap
                    hf = HydraFetcher(
                        project="dusty",
                        jobset="fpga-tool-perf",
                        eval_num=eval_num,
                        mapping={"build_num": "build_num"})
                    result = hf.get_evaluation().get_df()

                    expected = pd.DataFrame({
                        "build_num": [x for x in range(eval_num * 4, eval_num * 4 + 4)],
                        "freq": [81.05 for _ in range(4)]
                        })
                    print(result.columns)
                    assert_frame_equal(result, expected)

                    eval_id = hf.get_evaluation().get_eval_id()
                    assert eval_id == eval_num + 1

    def test_hydrafetcher_get_evaluation_icebreaker_new(self):
        """
        get_evaluation() should return an Evaluation corresponding to the small
        dataset and the specified eval_num.

        Tests legacy icebreaker support does not affect an eval where the date
        is not before 7-31-2020.
        """
        with requests_mock.Mocker() as m:
            evals_fn = 'tests/sample_data/evals.small.json'
            evals_url = 'https://hydra.vtr.tools/jobset/dusty/fpga-tool-perf/evals'

            build_fn = 'tests/sample_data/build.small.json'

            # setup /evals request mock
            with open(evals_fn, "r") as f:
                json_data = f.read()
                m.get(evals_url, text=json_data)

            # setup /build and /meta.json request mock
            with open(build_fn, "r") as f:
                json_data = f.read()

                # test if date is not before 7-31-2020
                for build_num in range(4):
                    # /build/:buildid
                    build_url = f'https://hydra.vtr.tools/build/{build_num}'
                    m.get(build_url, text=json_data)

                    # /meta.json
                    meta_url = f'https://hydra.vtr.tools/build/{build_num}/download/5/meta.json'
                    payload = {
                        "build_num": build_num,
                        "date": "2020-07-31T22:12:40",
                        "board": "icebreaker",
                        "max_freq": 81050000
                    }
                    m.get(meta_url, json=payload) # setup /meta.json request mock

                # test if board is not icebreaker
                for build_num in range(4, 8):
                    # /build/:buildid
                    build_url = f'https://hydra.vtr.tools/build/{build_num}'
                    m.get(build_url, text=json_data)

                    # /meta.json
                    meta_url = f'https://hydra.vtr.tools/build/{build_num}/download/5/meta.json'
                    payload = {
                        "build_num": build_num,
                        "date": "2020-07-30T22:12:40",
                        "board": "arty",
                        "max_freq": 81050000
                    }
                    m.get(meta_url, json=payload) # setup /meta.json request mock

                # test if date and board are not included
                for build_num in range(8, 12):
                    # /build/:buildid
                    build_url = f'https://hydra.vtr.tools/build/{build_num}'
                    m.get(build_url, text=json_data)

                    # /meta.json
                    meta_url = f'https://hydra.vtr.tools/build/{build_num}/download/5/meta.json'
                    payload = {
                        "build_num": build_num,
                        # date and build intentionally removed
                        "max_freq": 81050000
                    }
                    m.get(meta_url, json=payload) # setup /meta.json request mock

            # run tests on different eval_num
            for eval_num in range(0, 3):
                with self.subTest(eval_num=eval_num):
                    # if mapping is not defined, should not remap
                    hf = HydraFetcher(
                        project="dusty",
                        jobset="fpga-tool-perf",
                        eval_num=eval_num,
                        mapping={"build_num": "build_num"})
                    result = hf.get_evaluation().get_df()

                    # all results should have frequency converted from hz to mhz (no legacy icebreaker)
                    expected = pd.DataFrame({
                        "build_num": [x for x in range(eval_num * 4, eval_num * 4 + 4)],
                        "freq": [81.05 for _ in range(4)]
                        })
                    assert_frame_equal(result, expected)

                    eval_id = hf.get_evaluation().get_eval_id()
                    assert eval_id == eval_num + 1

    def test_hydrafetcher_get_evaluation_eval_num_absolute(self):
        """
        get_evaluation() should return an Evaluation corresponding to the small
        dataset and the specified *absolute* eval_num.

        Tests whether the fetcher is able to get an exact evaluation by absolute
        eval num and pagination.
        """
        with requests_mock.Mocker() as m:
            evals_fn = 'tests/sample_data/evals.small.json'
            evals_fn2 = 'tests/sample_data/evals.small.2.json'
            evals_url = 'https://hydra.vtr.tools/jobset/dusty/fpga-tool-perf/evals'

            build_fn = 'tests/sample_data/build.small.json'

            # setup /evals request mock
            with open(evals_fn, "r") as f:
                json_data = f.read()
                m.get(evals_url, text=json_data)

            # setup /evals?page=2 request mock
            with open(evals_fn2, "r") as f:
                json_data = f.read()
                m.get(evals_url + "?page=2", text=json_data)

            # setup /build and /meta.json request mock
            with open(build_fn, "r") as f:
                json_data = f.read()

                for build_num in range(24):
                    # /build/:buildid
                    build_url = f'https://hydra.vtr.tools/build/{build_num}'
                    m.get(build_url, text=json_data)

                    # /meta.json
                    meta_url = f'https://hydra.vtr.tools/build/{build_num}/download/5/meta.json'
                    payload = {"build_num": build_num}
                    m.get(meta_url, json=payload) # setup /meta.json request mock

            # run tests on different eval_num
            for eval_num in range(1, 7):
                with self.subTest(eval_num=eval_num):
                    # if mapping is not defined, should not remap
                    hf = HydraFetcher(
                        project="dusty",
                        jobset="fpga-tool-perf",
                        eval_num=eval_num,
                        absolute_eval_num=True)
                    result = hf.get_evaluation().get_df()

                    col = [x for x in range(eval_num * 4 - 4, eval_num * 4)]
                    expected = pd.DataFrame({"build_num": col})
                    assert_frame_equal(result, expected)

                    eval_id = hf.get_evaluation().get_eval_id()
                    assert eval_id == eval_num

    def test_hydrafetcher_get_evaluation_mapping(self):
        """
        get_evaluation() should return an Evaluation corresponding to the small
        dataset and mapping.

        Tests whether exclusion and renaming works when remapping.
        """
        with requests_mock.Mocker() as m:
            evals_fn = 'tests/sample_data/evals.small.json'
            evals_url = 'https://hydra.vtr.tools/jobset/dusty/fpga-tool-perf/evals'

            build_fn = 'tests/sample_data/build.small.json'

            # setup /evals request mock
            with open(evals_fn, "r") as f:
                json_data = f.read()
                m.get(evals_url, text=json_data)

            # setup /build and /meta.json request mock
            with open(build_fn, "r") as f:
                json_data = f.read()

                for build_num in range(12):
                    # /build/:buildid
                    build_url = f'https://hydra.vtr.tools/build/{build_num}'
                    m.get(build_url, text=json_data)

                    # /meta.json
                    meta_url = f'https://hydra.vtr.tools/build/{build_num}/download/5/meta.json'
                    payload = {
                        "build_num": build_num,
                        "date": "2020-07-17T22:12:40",
                        "board": "arty"
                    }
                    m.get(meta_url, json=payload) # setup /meta.json request mock

            # test exclusion
            hf1 = HydraFetcher(
                project="dusty",
                jobset="fpga-tool-perf",
                eval_num=0,
                mapping={"build_num": "build_num"})
            result1 = hf1.get_evaluation().get_df()

            col = [x for x in range(4)]
            expected1 = pd.DataFrame({"build_num": col})

            assert_frame_equal(result1, expected1)

            # test renaming
            hf2 = HydraFetcher(
                project="dusty",
                jobset="fpga-tool-perf",
                eval_num=0,
                mapping={"build_num": "renamed_num"})
            result2 = hf2.get_evaluation().get_df()

            col = [x for x in range(4)]
            expected2 = pd.DataFrame({"renamed_num": col})

            assert_frame_equal(result2, expected2)

    def test_hydrafetcher_hydra_clock_names(self):
        """
        get_evaluation() should return an Evaluation corresponding to the small
        dataset and hydra_clock_names attribute.

        Tests whether the ordering of hydra_clock_names is correctly
        implemented.
        """
        with requests_mock.Mocker() as m:
            evals_fn = 'tests/sample_data/evals.small.json'
            evals_url = 'https://hydra.vtr.tools/jobset/dusty/fpga-tool-perf/evals'

            build_fn = 'tests/sample_data/build.small.json'

            # setup /evals request mock
            with open(evals_fn, "r") as f:
                json_data = f.read()
                m.get(evals_url, text=json_data)

            # setup /build and /meta.json request mock
            with open(build_fn, "r") as f:
                json_data = f.read()

                for build_num in range(4):
                    # /build/:buildid0
                    build_url = f'https://hydra.vtr.tools/build/{build_num}'
                    m.get(build_url, text=json_data)

                    # /meta.json
                    url = f'https://hydra.vtr.tools/build/{build_num}/download/5/meta.json'
                    payload = {
                        "max_freq": {
                            "clk": {
                                "actual": 1000000
                            },
                            "clk_i": {
                                "actual": 2000000
                            },
                            "sys_clk": {
                                "actual": 3000000
                            }
                        },
                        "date": "2020-07-17T22:12:40",
                        "board": "arty"
                    }
                    m.get(url, json=payload) # setup /meta.json request mock

            # test which clock is chosen if there are multiple clocks

            # format test cases as tuples:
            # ({hydra_clock_names}, {expected_clock})
            test_cases = [
                (["clk", "clk_i", "sys_clk"], 1.0),
                (["clk_i", "sys_clk", "clk"], 2.0),
                (["sys_clk", "clk", "clk_i"], 3.0),
            ]

            for hydra_clock_name, expected_clock in test_cases:
                with self.subTest(hydra_clock_name=hydra_clock_name):
                    hf = HydraFetcher(
                        project="dusty",
                        jobset="fpga-tool-perf",
                        eval_num=0,
                        hydra_clock_names=hydra_clock_name)
                    result = hf.get_evaluation().get_df()

                    expected_col = [expected_clock for _ in range(4)]
                    expected_series = pd.Series(expected_col, name="freq")
                    assert_series_equal(result["freq"], expected_series)

            # test that the shortest clock name is chosen if no matches
            hf = HydraFetcher(
                project="dusty",
                jobset="fpga-tool-perf",
                eval_num=0,
                hydra_clock_names=[])
            result = hf.get_evaluation().get_df()

            expected_col = [1.0 for _ in range(4)]
            expected_series = pd.Series(expected_col, name="freq")
            assert_series_equal(result["freq"], expected_series)


class TestJSONFetcherSmall(unittest.TestCase):
    """
    Testing by partition.

    JSONFetcher:
        __init__(path, mapping_dict)
        get_evaluation()
    """

    def test_jsonfetcher_init(self):
        """
        Calling init should save the arguments as an instance variable.
        """
        fetcher = JSONFetcher(
            path="tests/sample_data/dataframe_small.json",
            mapping={"a": "c", "b": "d"}
        )

        self.assertEqual(fetcher.path, "tests/sample_data/dataframe_small.json")
        self.assertEqual(fetcher.mapping, {"a": "c", "b": "d"})

    def test_jsonfetcher_get_evaluation(self):
        """
        get_evaluation() should return an Evaluation corresponding to the small
        dataset.
        """
        fetcher = JSONFetcher('tests/sample_data/dataframe_small.json')
        result = fetcher.get_evaluation().get_df()

        expected_projects = pd.Series(
            ["ibex", "oneblink", "litex-linux", "oneblink", "litex-linux"],
            name="project"
        )
        assert_series_equal(result["project"], expected_projects)


    def test_jsonfetcher_get_evaluation_mapping(self):
        """
        get_evaluation() should return an Evaluation corresponding to the small
        dataset and mapping.

        Tests whether exclusion and renaming works when remapping.
        """
        fetcher = JSONFetcher(
            'tests/sample_data/dataframe_small.json',
            mapping={"project": "proj"}
        )
        result = fetcher.get_evaluation().get_df()

        # test renaming
        expected_projects = pd.Series(
            ["ibex", "oneblink", "litex-linux", "oneblink", "litex-linux"],
            name="proj"
        )
        assert_series_equal(result["proj"], expected_projects)

        # test exclusion
        expected_columns = pd.Index(["proj"]) # should not have other columns
        assert_index_equal(result.columns, expected_columns)

""" Fetchers are responsible for ingesting and standardizing data for future processing. """
from datetime import datetime
import json
from typing import Any, Dict, List

import pandas as pd
import requests

import ftpvl.helpers as Helpers
from ftpvl.evaluation import Evaluation


class Fetcher:
    """
    This is a superclass for all fetchers.

    Fetchers allow the user to retrieve test data from a data source and output
    as an Evaluation for use by other tools in the library.
    """

    def __init__(self):
        self._abs_eval_id = None

    def _download(self) -> Any:
        """
        Retrieves evaluation data over the internet and returns the data.
        """
        raise NotImplementedError

    def _preprocess(self, data: Any) -> pd.DataFrame:
        """
        Process the data and return the resulting dataframe.
        """
        raise NotImplementedError

    def get_evaluation(self) -> Evaluation:
        """
        Returns an Evaluation that represents the fetched data.
        """
        data = self._download()
        preprocessed_df = self._preprocess(data)
        return Evaluation(preprocessed_df, eval_id=self._abs_eval_id)

class GoogleStorageFetcher(Fetcher):
    """
    Represents a download and preprocessor of test results from
    'google cloud storage'.

    Parameters
    ----------
    eval_num : int
        An integer that specifies the evaluation to download.
    project : str, optional
        The project name to use when fetch from Google Cloud Storage. By default fpga-tool-perf
    jobset : str, optional
        The jobset name to use when fetching from Google Cloud Storage. By default continuous.
    """
    def __init__(
        self,
        eval_num: int,
        project: str = "fpga-tool-perf",
        jobset: str = "continuous",
        mapping: dict = None,
        hydra_clock_names: list = None
    ) -> None:
        self._abs_eval_id = eval_num
        self.eval_num = eval_num
        self.project = project
        self.jobset = jobset
        self.mapping = mapping
        self.hydra_clock_names = hydra_clock_names

    def _get_builds(self, eval_num: int, params: str = "") -> List[str]:
        """
        Function that returns a list of paths to download given an eval_num.

        Parameters
        ----------
        eval_num : int
            An integer that specifies the evaluation to download. Functionality
            differs depending on whether `absolute_eval_num` is True
        params : str
            A string of query parameters used when fetching the evaluations.
            Most commonly used for pagination. By default, "".

        Returns
        -------
        List[int]
            A list of paths to download that correspond with the eval_num

        Raises
        ------
        ConnectionError
            Raised if the HTTP request to get the evaluations fails.
        """
        base_api_path = f"https://www.googleapis.com/storage/v1/b/{self.project}/o"
        prefix_param = f"artifacts/prod/foss-fpga-tools/{self.project}/{self.jobset}/"
        base_download_path = f"https://storage.googleapis.com/{self.project}/"
        next_page_token = ""
        meta_urls = []

        while True:
            req_url = f"{base_api_path}?delimiter=meta.json&prefix={prefix_param}{eval_num}{next_page_token}"
            print(f"Trying to download meta.json files from: {req_url}...")
            resp = requests.get(
                req_url,
                headers={"Content-Type": "application/json"},
            )

            if resp.status_code != 200:
                raise ConnectionError(f"Unable to get evals from server. Status code: {resp.status_code}")

            evals_json = resp.json()

            if "prefixes" not in evals_json:
                raise KeyError(f"Build is empty! No artifacts were generated.")

            for prefix in evals_json["prefixes"]:
                meta_urls += [base_download_path + prefix]
            try:
                next_page_token = "&pageToken=" + evals_json["nextPageToken"]
            except KeyError:
                break

        return meta_urls

    def _download(self) -> List[Dict]:
        """
        Fetches data from Google Cloud Storage, returning a list of decoded meta.json dicts
        corresponding to the builds of the eval_num evaluation.

        Returns
        -------
        List[Dict]
            A list of dictionaries that represent the decoded meta.json files
            of each test result.

        Raises
        ------
        ValueError
            Raised when fetcher wont find any successful builds from eval_num

        """
        # get build numbers from eval_num
        meta_urls = self._get_builds(self.eval_num)

        # fetch build info and download 'meta.json'
        data = []
        for meta_url in meta_urls:
            # get build info
            resp = requests.get(
                    meta_url,
                    headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                print(
                    "Warning:",
                    f"Unable to get build {meta_url} file. Status code: {resp.status_code}",
                )
                continue
            try:
                data.append(resp.json())
            except json.decoder.JSONDecodeError:
                print("Warning:", f"Unable to decode build {meta_url}")

        if len(data) == 0:
            raise ValueError(f"Unable to get any successful builds from eval_num {self.eval_num}.")

        return data

    def _process_toolchain(self, data: List[Dict]) -> List[Dict]:
        for index in range(len(data)):
            for key, value in data[index]['toolchain'].items():
                toolchain = key
                pr_tool = value['pr_tool']
                synthesis_tool = value['synthesis_tool']
            data[index]['toolchain'] = toolchain
            data[index]['pr_tool'] = pr_tool
            data[index]['synthesis_tool'] = synthesis_tool
        return data

    def _preprocess(self, data: List[Dict]) -> pd.DataFrame:
        """
        Using data from _download(), processes and standardizes the data and
        returns a Pandas DataFrame.
        """

        self._process_toolchain(data)
        flattened_data = [Helpers.flatten(x) for x in data]

        processed_data = []
        def process_data(row):
            processed_row = {}
            if self.mapping is None:
                processed_row = row
            else:
                for in_col_name, out_col_name in self.mapping.items():
                    processed_row[out_col_name] = row[in_col_name]

            processed_row.update(Helpers.get_versions(row))
            processed_data.append(processed_row)

        for row in flattened_data:
            if row['toolchain'] == "nextpnr-ice40":
                continue

            clocks = {}
            for col in row.keys():
                parts = col.split('.')
                is_max_freq = parts[0] == "max_freq"

                if is_max_freq:
                    clock_name = ".".join(parts[1:-1])
                else:
                    continue

                if clock_name not in clocks.keys():
                    clocks[clock_name] = {}

                clocks[clock_name][parts[-1]] = row[col]

            if clocks:
                for k, v in clocks.items():
                    row["clock_name"] = k

                    for k2, v2 in v.items():
                        row[k2] = v2

                    process_data(row)
            else:
                row["clock_name"] = ""
                row["actual"] = 0.0
                row["requested"] = 0.0
                row["met"] = False
                row["hold_violation"] = 0.0
                row["setup_violation"] = 0.0
                process_data(row)


        return pd.DataFrame(processed_data).dropna(axis=1, how="all")

    def get_evaluation(self) -> Evaluation:
        return super().get_evaluation()


class HydraFetcher(Fetcher):
    """
    Represents a downloader and preprocessor of test results from
    `hydra.vtr.tools`.

    Parameters
    ----------
    project : str
        The project name to use when fetching from Hydra
    jobset : str
        The jobset name to use when fetching from Hydra
    eval_num : int, optional
        An integer that specifies the evaluation to download. Functionality
        differs depending on whether `absolute_eval_num` is True, by default 0
    absolute_eval_num : bool, optional
        Flag that specifies if the eval_num is an absolute identifier instead of
        a relative identifier. If True, the fetcher will find an evaluation
        with the exact ID in `eval_num`. If False, `eval_num` should be a
        non-negative integer with `0` being the latest evaluation and `1` being
        the second latest evaluation, etc. By default False.
    mapping : dict, optional
        A dictionary mapping input column names to output
        column names, if needed for remapping, by default None
    hydra_clock_names : list, optional
        An optional ordered list of strings used in finding
        the actual frequency for each build result, by default None
    """

    def __init__(
        self,
        project: str,
        jobset: str,
        eval_num: int = 0,
        absolute_eval_num: bool = False,
        mapping: dict = None,
        hydra_clock_names: list = None
    ) -> None:
        super().__init__() # inits self._abs_eval_id
        self.project = project
        self.jobset = jobset
        self.eval_num = eval_num
        self.absolute_eval_num = absolute_eval_num
        self.mapping = mapping
        self.hydra_clock_names = hydra_clock_names

    def _get_builds(self, eval_num: int, params: str = "") -> List[int]:
        """
        Recursive function that returns a list of build numbers given an eval_num
        and whether it is an absolute eval num.

        Parameters
        ----------
        eval_num : int
            An integer that specifies the evaluation to download. Functionality
            differs depending on whether `absolute_eval_num` is True
        params : str
            A string of query parameters used when fetching the evaluations.
            Most commonly used for pagination. By default, "".

        Returns
        -------
        List[int]
            A list of builds that correspond with the eval_num

        Raises
        ------
        ConnectionError
            Raised if the HTTP request to get the evaluations fails.
        IndexError
            Raised if the relative eval_num is not valid.
        ValueError
            Raised if the eval_num has no associated builds.
        """
        resp = requests.get(
            f"https://hydra.vtr.tools/jobset/{self.project}/{self.jobset}/evals{params}",
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            raise ConnectionError("Unable to get evals from server.")
        evals_json = resp.json()

        if self.absolute_eval_num:
            self._abs_eval_id = eval_num # set absolute id for eval
            for eval_data in evals_json["evals"]:
                if eval_data["id"] == eval_num:
                    return eval_data["builds"]
            if "next" in evals_json:
                return self._get_builds(
                    eval_num,
                    evals_json["next"] # query param for next page
                )
            else: # if couldn't find ID and there is no next page
                raise ValueError(f"Unable to find absolute eval_num {eval_num}")

        else: # if not absolute eval number
            if eval_num >= len(evals_json["evals"]):
                # check if there is a next page
                if "next" in evals_json:
                    return self._get_builds(
                        eval_num - len(evals_json["evals"]),
                        evals_json["next"] # query param for next page
                    )
                else:
                    raise IndexError(f"Unable to find relative eval_num {self.eval_num}")
            self._abs_eval_id = evals_json["evals"][eval_num]["id"] # set absolute id for eval
            return evals_json["evals"][eval_num]["builds"]

    def _download(self) -> List[Dict]:
        """
        Fetches data from Hydra, returning a list of decoded meta.json dicts
        corresponding to the builds of the eval_num evaluation.

        Returns
        -------
        List[Dict]
            A list of dictionaries that represent the decoded meta.json files
            of each test result.

        Raises
        ------
        ConnectionError
            Raised if `hydra.vtr.tools` returns a non-200 status code when
            fetching evals.

        IndexError
            Raised if the specified eval_num is invalid due to it being too
            large.

        ValueError
            Raised if all builds in a given eval failed.
        """
        # get build numbers from eval_num
        build_nums = self._get_builds(self.eval_num)

        # fetch build info and download 'meta.json'
        data = []
        for build_num in build_nums:
            # get build info
            resp = requests.get(
                f"https://hydra.vtr.tools/build/{build_num}",
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                raise Exception(f"Unable to get build {build_num}, got status code {resp.status_code}.")

            decoded = None
            try:
                decoded = resp.json()
            except json.decoder.JSONDecodeError as err:
                raise Exception(f"Unable to decode build {build_num} JSON file, {str(err)}")

            # check if build was successful
            if decoded.get("buildstatus") != 0:
                print(f"Warning: Build {build_num} failed with non-zero exit. Skipping...")
                continue

            # check if meta.json exists
            meta_json_id = None
            for product_id, product_desc in decoded.get("buildproducts", {}).items():
                if product_desc.get("name", "") == "meta.json":
                    meta_json_id = product_id

            if meta_json_id is None:
                print(f"Warning: Build {build_num} does not contain meta.json file. Skipping...")
                continue

            # download meta.json
            resp = requests.get(
                f"https://hydra.vtr.tools/build/{build_num}/download/{meta_json_id}/meta.json",
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                print(
                    "Warning:",
                    f"Unable to get build {build_num} meta.json file.",
                )
                continue
            try:
                data.append(resp.json())
            except json.decoder.JSONDecodeError:
                print("Warning:", f"Unable to decode build {build_num}")

        if len(data) == 0:
            raise ValueError(f"Unable to get any successful builds from eval_num {self.eval_num}.")

        return data

    def _check_legacy_icebreaker(self, row):
        """
        Returns True if row is from a test on an Icebreaker board before Jul 31,
        2020.

        This is useful because these legacy tests recorded frequency in MHz
        instead of Hz, while all other boards record in Hz. This flag can be
        used to check if the units need to be changed.

        Parameters
        ----------
        row : dict
            a dictionary that is the result of decoding a meta.json file

        Returns
        -------
        bool
            Returns true if row is icebreaker board before Aug 1, 2020. False
            otherwise.
        """
        date = None
        board = None
        try:
            date = row["date"] # format: 2020-07-17T22:12:41
            board = row["board"]
            timestamp = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
            return timestamp < datetime(2020, 7, 31) and board == "icebreaker"
        except KeyError:
            print("Warning: Unable to find date and board in meta.json, required for supporting legacy Icebreaker.")
            return False # Assume not legacy icebreaker

    def _preprocess(self, data: List[Dict]) -> pd.DataFrame:
        """
        Using data from _download(), processes and standardizes the data and
        returns a Pandas DataFrame.
        """
        flattened_data = [Helpers.flatten(x) for x in data]

        processed_data = []
        for row in flattened_data:
            legacy_icestorm = self._check_legacy_icebreaker(row)
            processed_row = {}
            if self.mapping is None:
                processed_row = row
            else:
                for in_col_name, out_col_name in self.mapping.items():
                    processed_row[out_col_name] = row[in_col_name]

            actual_freq = Helpers.get_actual_freq(row, self.hydra_clock_names)
            if actual_freq:
                if legacy_icestorm:
                    # freq in MHz, no change needed
                    processed_row["freq"] = actual_freq
                else:
                    processed_row["freq"] = actual_freq / 1_000_000 # convert hz to mhz
            processed_row.update(Helpers.get_versions(row))
            processed_data.append(processed_row)

        return pd.DataFrame(processed_data).dropna(axis=1, how="all")

    def get_evaluation(self) -> Evaluation:
        return super().get_evaluation()


class JSONFetcher(Fetcher):
    """
    Represents a loader and preprocessor of test results from a JSON file.

    Parameters
    ----------
    path : str
        A string file path pointing to the dataframe JSON file.

    mapping : dict, optional
        An optional dictionary mapping input column names to output
        column names., by default None
    """

    def __init__(self, path: str, mapping: dict = None) -> None:

        super().__init__()
        self.path = path
        self.mapping = mapping

    def _download(self) -> str:
        """
        Returns the path of the JSON file.
        """
        return self.path

    def _preprocess(self, path: str) -> pd.DataFrame:
        """
        Given the path, load the data in pandas, process the data, and return
        the resulting dataframe.
        """
        loaded_df = pd.read_json(path)
        if self.mapping is None:
            return loaded_df
        else:
            return loaded_df.filter(items=self.mapping.keys()).rename(
                columns=self.mapping
            )

    def get_evaluation(self) -> Evaluation:
        return super().get_evaluation()

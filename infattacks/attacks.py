import numpy as np
import pandas as pd
import itertools as it
from humanize import intword
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from infattacks.data import Data
from abc import ABC, abstractmethod

class Attack(ABC):
    """
    Abstract base class for inference attacks on datasets.

    Parameters:
        - dataset (Data): The dataset to be attacked.
        - qids (list): The quasi-identifiers used for the attack.
        - sensitive (list, optional): List of sensitive attributes, default is None.
    """

    def __init__(self, data: Data, qids: list, sensitive=None) -> None:
        """
        Initialize an instance of the ReidentificationAttacker class.

        Parameters:
            dataset (Data): The dataset to be attacked.
            qids (list): The quasi-identifiers used for the attack.
            sensitive (list, optional): List of sensitive attributes, default is None.

        Attributes:
            - data (Data): The dataset being attacked.
            - qids (list): The quasi-identifiers used for the attack.
            - sensitive (list): List of sensitive attributes.
            - num_qids (int): The number of quasi-identifiers.
            - num_sensitive (int): The number of sensitive attributes.
            - post_comb_qids_reid (None or DataFrame): Posterior vulnerabilities of Re-identification attacks for different number and combinations of quasi-identifiers. It's populated after calling the `attack_comb_qids` method.
            - post_comb_qids_ai (None or DataFrame): Posterior vulnerabilities of Attribute-inference attacks for different number and combinations of quasi-identifiers. It's populated after calling the `attack_comb_qids` method.

        Note:
            - The 'sensitive' parameter is optional. If not provided, it defaults to None.
            - 'post_comb_qids_reid' and 'post_comb_qids_ai' are initially set to None and can be populated
            after calling the 'attack_comb_qids' method.
        """
        self.data = data
        self.qids = qids
        self.sensitive = sensitive
        self.num_qids = len(self.qids)
        self.num_sensitive = len(self.sensitive)
        self.post_comb_qids_reid = None
        self.post_comb_qids_ai = None

    @abstractmethod
    def prior_reid(self) -> float:
        """
        Abstract method for calculating the prior vulnerability of a Re-identification attack.

        Returns:
            float: The prior vulnerability.
        """
        pass

    @abstractmethod
    def post_reid(self):
        """
        Abstract method for calculating the posterior vulnerability of a Re-identification attack.
        """
        pass

    @abstractmethod
    def prior_ai(self) -> dict:
        """
        Abstract method for calculating the prior vulnerability of an Attribute-inference attack.

        Returns:
            dict: A dictionary containing the prior vulnerability for each sensitive attribute.
        """
        pass

    @abstractmethod
    def post_ai(self):
        """
        Abstract method for calculating the posterior vulnerability of an Attribute-inference attack.
        """
        pass

    def _gen_hist(self, partition_vul, counts, hist_bin_size) -> np.ndarray:
        """
        Generate a histogram of posterior vulnerabilities.

        Parameters:
            - partition_vul (list or np.ndarray): List or array where the ith position is the .
            - counts (list or np.ndarray): List or array containing the counts associated with each partition.
            - hist_bin_size (int): Size of each histogram bin.

        Returns:
            np.ndarray: An array representing the histogram counts, where the ith position is the number of counts in the ith bin of the histograms (that varies with hist_bin_size). For instance, if hist_bin_size=5, bin 0: [0, 0.05), bin 2: [0.05, 0.1), ..., bin 19: [0.95, 1].
        """
        partition_vul = np.array(partition_vul)
        counts = np.array(counts)
        num_bins = 100 // hist_bin_size

        histogram_counts = np.zeros(num_bins)
        for i in np.arange(len(partition_vul)):
            # Ex.: If hist_bin_size = 5, bin 0: [0, 0.05), bin 2: [0.05, 0.1), ..., bin 19: [0.95, 1]
            bin_number = min(int(partition_vul[i] / hist_bin_size * 100), num_bins - 1)
            histogram_counts[bin_number] += counts[i]

        return histogram_counts
    
    def post_comb_qids(self, num_min=1, num_max=None) -> None:
        """
        Compute the posterior vulnerabilitiy of Re-identification and Attribute-inference attacks for different number and combinations of quasi-identifiers.

        The results are stored as pandas DataFrames in the class attributes `post_comb_qids_reid` and `post_comb_qids_ai`.

        Parameters: 
            num_min (int, optional): Minimum number of quasi-identifiers (QIDs) to consider.
                Defaults is 1.
            num_max (int, optional): Maximum number of quasi-identifiers (QIDs) to consider.
                If None, it considers all possible combinations up to the total number of QIDs (the power set).
                Defaults is None.

        Returns:
            None: This method updates the object's state by storing the computed results in
            self.post_comb_qids_reid and self.post_comb_qids_ai.
        """
        post_reid = {"qids": [], "post_vul": []}
        post_ai = {"qids": [], "post_vul": []}

        if num_min is None:
            num_min = 1
        if num_max is None:
            num_max = self.num_qids

        # For all possible QIDs combinations
        for num_qids in np.arange(num_min, num_max + 1):
            for qids_sel in it.combinations(self.qids, num_qids):
                qids_sel = list(qids_sel)

                # Re-identification attack
                post_reid["qids"].append(", ".join(qids_sel))
                post_reid["post_vul"].append(self.post_reid(qids_sel))

                # Attribute-inference attack
                post_ai["qids"].append(", ".join(qids_sel))
                post_ai["post_vul"].append(self.post_ai(qids_sel))

        post_reid = pd.DataFrame(post_reid)
        post_ai = pd.DataFrame(post_ai)

        # Transform the result of each sensitive attribute to a column
        for att in self.sensitive:
            post_ai["post_vul_" + att] = post_ai["post_vul"].apply(lambda x: x[att])
        post_ai.drop("post_vul", axis=1, inplace=True)

        # Add number of qids
        post_reid["num_qids"] = post_reid["qids"].apply(lambda x: str(x).count(",") + 1)
        post_ai["num_qids"] = post_ai["qids"].apply(lambda x: str(x).count(",") + 1)

        self.post_comb_qids_reid = post_reid
        self.post_comb_qids_ai = post_ai
    
    def max_post_reid(self) -> pd.DataFrame:
        """
        Given all attacks run in `attack_comb_qids` method, for each number of QIDS, get the combination that produced the highest posterior vulnerability of Re-identification attacks.

        Returns:
            pd.DataFrame: A DataFrame containing the maximum posterior vulnerabilities of Re-identification attacks per number of qids.
        """
        if self.post_comb_qids_reid is None:
            raise ValueError("max_post_reid should be called only after calling attack_comb_qids")

        vul_column = "post_vul"
        max_post = pd.DataFrame(columns=["num_qids", "qids", vul_column])
        for num_att in np.arange(1, self.num_qids + 1):
            filter_results = self.post_comb_qids_reid[
                self.post_comb_qids_reid["num_qids"] == num_att
            ]
            max_vul = filter_results.loc[filter_results[vul_column].idxmax()]
            max_post.loc[len(max_post)] = max_vul

        return max_post

    def max_post_ai(self, sensitive: str) -> pd.DataFrame:
        """
        Given all attacks run in `attack_comb_qids` method, for each number of QIDS, get the combination that produced the highest posterior vulnerability of Attribute-inference attacks.

        Parameters:
            sensitive (str): The sensitive attribute for which to take the maximum posterior vulnerability.

        Returns:
            pd.DataFrame: A DataFrame containing the maximum posterior vulnerabilities of Attribute-inference attacks per number of qids.
        """
        if self.post_comb_qids_ai is None:
            raise ValueError("max_post_ai should be called only after calling attack_comb_qids")
                             
        if sensitive not in self.data.columns:
            raise ValueError(f"Column {sensitive} is not in the dataset")

        vul_column = "post_vul_" + sensitive

        max_post = pd.DataFrame(columns=["num_qids", "qids", vul_column])
        for num_att in np.arange(1, self.num_qids + 1):
            post_results = self.post_comb_qids_ai
            filter_results = post_results[post_results["num_qids"] == num_att]
            max_vul = filter_results.loc[filter_results[vul_column].idxmax()]
            max_post.loc[len(max_post)] = max_vul

        return max_post

    def save_post_reid(self, file_name) -> None:
        """
        Save the results of posterior vulnerabilities generated by `attack_comb_qids` for Re-identification attacks in a CSV file.

        Parameters:
            file_name (str): The name of the CSV file to save the results.

        Note:
            - The CSV file will contain columns such as 'num_qids', 'qids', and 'post_vul' with posterior vulnerability results.
        """
        if self.post_comb_qids_reid is None:
            raise ValueError("save_post_reid should be called only after calling attack_comb_qids")

        self.post_comb_qids_reid.to_csv(
            file_name,
            float_format="%.8f",
            index=False
        )

    def save_post_ai(self, file_name) -> None:
        """
        Save the results of posterior vulnerabilities generated by `attack_comb_qids` for Attribute-inference attacks in a CSV file.

        Parameters:
            file_name (str): The name of the CSV file to save the results.

        Note:
            - The CSV file will contain columns such as 'num_qids', 'qids', and 'post_vul_X' with posterior vulnerability results, where X is a sensitive attribute. If there is more than one sensitive attribute, there will a single column for each one of them.
        """
        if self.post_comb_qids_ai is None:
            raise ValueError("save_post_ai should be called only after calling attack_comb_qids")

        self.post_comb_qids_ai.to_csv(
            file_name,
            float_format="%.8f",
            index=False
        )

    def save_post(self, file_name_reid, file_name_ai) -> None:
        """
        Save both Re-identification and Attribute-inference posterior vulnerability results.

        This function calls 'save_post_reid' and 'save_post_ai' to save the results of posterior vulnerability
        for Re-identification and Attribute-inference attacks, respectively.

        Parameters:
            file_name_reid (str): The name of the CSV file to save Re-identification results.
            file_name_ai (str): The name of the CSV file to save Attribute-inference results.
        """
        self.save_post_reid(file_name_reid)
        self.save_post_ai(file_name_ai)

class Probabilistic(Attack):
    """
    Concrete class for probabilistic attacks on datasets. Inherits from Attack.

    Parameters:
        - dataset (Data): The dataset to be attacked.
        - qids (list): The quasi-identifiers used for the attack.
        - sensitive (list, optional): List of sensitive attributes, default is None.
    """

    def __init__(self, dataset: Data, qids: list, sensitive=None) -> None:
        """
        Initializes an instance of the Probabilistic class.

        Parameters:
            dataset (Data): The dataset to be attacked.
            qids (list): The quasi-identifiers used for the attack.
            sensitive (list, optional): List of sensitive attributes, default is None.
        """
        super().__init__(dataset, qids, sensitive)

    def prior_reid(self) -> float:
        """
        Calculates the prior vulnerability of Probabilistic Re-identification attack.

        Returns:
            float: The prior vulnerability.
        """
        return 1 / self.data.num_rows

    def post_reid(self, qids=None, hist=False, hist_bin_size=5):
        """
        Calculate the expected posterior vulnerability of Probabilistic Re-identification attack for a given dataset. If hist is True, it provides also the histogram of individual posterior vulnerabilities (i.e., the posterior of each person in the dataset).

        Parameters:
            - qids (list, optional): List of quasi-identifiers. If not provided, all columns will be used.
            - hist (bool, optional): Whether to generate the histogram of posterior vulnerabilities. Default is False.
            - hist_bin_size (int, optional): Bin size for the histogram if hist is True. Default is 5.

        Returns:
            float or tuple: If hist is False, returns the expected posterior vulnerability of Probabilistic Re-identification attack.
            If hist is True, returns a tuple containing the expected posterior probability and a histogram of individual posterior vulnerabilities.
        """
        if qids is None:
            qids = self.qids

        partitions = self.data.dataframe.groupby(qids).size().to_numpy()
        expected_post_prob = len(partitions) / self.data.num_rows

        if hist:
            histogram = self._gen_hist(1/np.array(partitions), partitions, hist_bin_size)
            return expected_post_prob, histogram
        
        return expected_post_prob

    def prior_ai(self) -> dict:
        """
        Calculates the prior vulnerability of Probabilistic Attribute-inference attack.

        Returns:
            dict: A dictionary containing the prior vulnerability for each sensitive attribute.
        """
        if self.sensitive is None:
            raise NameError("sensitive is not declared")
        
        results = dict()
        for sens in self.sensitive:
            results[sens] = self.data.dataframe[sens].value_counts().max() / self.data.num_rows
        return results

    def post_ai(self, qids=None, hist=False, hist_bin_size=5):
        """
        Calculate the expected posterior vulnerability of Probabilistic Attribute-inference attack for a given dataset. If hist is True, it provides also the histogram of individual posterior vulnerabilities (i.e., the posterior of each person in the dataset).

        Parameters:
            - qids (list, optional): List of quasi-identifiers. If not provided, all columns will be used.
            - hist (bool, optional): Whether to generate the histogram of posterior vulnerabilities. Default is False.
            - hist_bin_size (int, optional): Bin size for the histogram if hist is True. Default is 5.

        Returns:
            float or tuple: If hist is False, returns the expected posterior vulnerability of Probabilistic Attribute-inference attack.
            If hist is True, returns a tuple containing the expected posterior probability and the histogram of individual posterior vulnerabilities.
        """
        if qids is None:
            qids = self.qids

        if self.sensitive is None:
            raise NameError("sensitive is not declared")
        
        # max = size of largest partition
        # sum = number of people in all partitions (for that combination of qids + [sens])
        expectations = dict()
        histograms = dict()
        for sens in self.sensitive:
            partitions = self.data.dataframe.groupby(
                qids+[sens]).size().droplevel(self.sensitive).to_frame().rename(columns={0: "counts"})
            groupby_qids = partitions.groupby(qids)["counts"].agg(["max", "sum"]).reset_index()
            post_prob = groupby_qids["max"].sum() / self.data.num_rows
            if hist:
                counts = groupby_qids["sum"].to_numpy()
                partition_vul = groupby_qids["max"].to_numpy() / counts
                histograms[sens] = self._gen_hist(partition_vul, counts, hist_bin_size)

            expectations[sens] = post_prob

        if hist:
            return expectations, histograms
        
        return expectations

class Deterministic(Attack):
    """
    Concrete class for deterministic attacks on datasets. Inherits from Attack.

    Parameters:
        - dataset (Data): The dataset to be attacked.
        - qids (list): The quasi-identifiers used for the attack.
        - sensitive (list, optional): List of sensitive attributes, default is None.
    """

    def __init__(self, dataset: Data, qids: list, sensitive=None) -> None:
        """
        Initializes an instance of the Deterministic class.

        Parameters:
            dataset (Data): The dataset to be attacked.
            qids (list): The quasi-identifiers used for the attack.
            sensitive (list, optional): List of sensitive attributes, default is None.
        """
        super().__init__(dataset, qids, sensitive)

    def prior_reid(self) -> float:
        """
        Calculates the prior vulnerability of Deterministic Re-identification attack.

        Returns:
            float: The prior vulnerability.
        """
        return 1 if self.data.num_rows == 1 else 0

    def post_reid(self, qids=None) -> float:
        """
        Calculates the posterior vulnerability of Deterministic Re-identification attack.

        Parameters:
            qids (list, optional): List of quasi-identifiers.
                If not provided (default is None), all the qids specified in the initialization of the class will be considered.

        Returns:
            float: The posterior vulnerability.
        """
        if qids is None:
            qids = self.qids

        partitions = self.data.dataframe.groupby(qids).size().to_numpy()
        partition_size, counts = np.unique(partitions, return_counts=True)
        post_prob = counts[np.where(partition_size == 1)] / self.data.num_rows
        return post_prob[0] if len(post_prob) >= 1 else 0

    def prior_ai(self) -> dict:
        """
        Calculates the prior vulnerability of Deterministic Attribute-inference attack.
        
        Returns:
            dict: A dictionary containing the prior vulnerability for each sensitive attribute.
        """
        if self.sensitive is None:
            raise ValueError("sensitive not defined")
        
        results = dict()
        for sens in self.sensitive:
            if len(self.data.dataframe[sens].unique()) > 1:
                results[sens] = 0
            else:
                results[sens] = 1
        return results

    def post_ai(self, qids=None) -> dict:
        """
        Calculates the posterior vulnerability of Deterministic Attribute-inference attack.
        
        Parameters:
            qids (list, optional): List of quasi-identifiers.
                If not provided (default is None), all the qids specified in the initialization of the class will be considered.

        Returns:
            dict: A dictionary containing the posterior vulnrability for each sensitive attribute.
        """
        if qids is None:
            qids = self.qids

        if self.sensitive is None:
            raise ValueError("sensitive not defined")
        
        results = dict()
        for sens in self.sensitive:
            partitions = self.data.dataframe.groupby(qids + [sens]).size().droplevel(self.sensitive).to_frame().rename(columns={0: "counts"})
            groupby_qids = partitions.groupby(qids)["counts"].agg(["max", "sum"]).reset_index()
            post_prob = groupby_qids[(groupby_qids["max"] == groupby_qids["sum"])]["max"].sum() / self.data.num_rows
            results[sens] = post_prob

        return results

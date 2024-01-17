import numpy as np
import pandas as pd
from infattacks.core.data import Data
from abc import ABC, abstractmethod
import itertools as it

class Attacks(ABC):
    """
    Abstract base class for privacy attacks on datasets.

    Parameters:
        - dataset (Data): The dataset to be attacked.
        - qids (list): The quasi-identifiers used for the attack.
        - sensitives (list, optional): List of sensitive attributes, default is None.

    Attributes:
        - dataset (Data): The dataset being attacked.
        - qids (list): The quasi-identifiers used for the attack.
        - sensitive (list): List of sensitive attributes.
    """

    def __init__(self, dataset: Data, qids: list, sensitives=None) -> None:
        """
        Initializes an instance of the Attacks class.

        Args:
            dataset (Data): The dataset to be attacked.
            qids (list): The quasi-identifiers used for the attack.
            sensitive (list, optional): List of sensitive attributes, default is None.
        """
        self.data = dataset
        self.qids = qids
        self.sensitive = sensitives
        self.num_qids = len(self.qids)
        self.num_sensitive = len(self.sensitive)
        self.results_powerset_reid = None
        self.results_powerset_ai = None

    @abstractmethod
    def prior_reid(self) -> float:
        """
        Abstract method for calculating the prior Re-identification attack probability.

        Returns:
            float: The prior Re-identification attack probability.
        """
        pass

    @abstractmethod
    def posterior_reid(self) -> float:
        """
        Abstract method for calculating the posterior Re-identification attack probability.

        Returns:
            float: The posterior Re-identification attack probability.
        """
        pass

    @abstractmethod
    def prior_ai(self) -> dict:
        """
        Abstract method for calculating the prior Attribute-Inference attack probabilities.

        Returns:
            dict: A dictionary containing the prior attack probabilities for each sensitive attribute.
        """
        pass

    @abstractmethod
    def posterior_ai(self) -> dict:
        """
        Abstract method for calculating the posterior Attribute-Inference attack probabilities.

        Returns:
            dict: A dictionary containing the posterior attack probabilities for each sensitive attribute.
        """
        pass

    def attack_power_set(self) -> None:
        """
        Perform re-identification and attribute-inference attacks for the power set of quasi-identifiers (qids).

        The method calculates the prior and posterior vulnerabilities for all possible combinations of qids, ranging
        from a single qid to the entire set. The results are stored in the class attributes `results_powerset_reid`
        and `results_powerset_ai`.

        Returns:
            None
        """
        prior_vul_reid = self.prior_reid()
        prior_vul_ai = self.prior_ai()

        posterior_vul_reid = {"qids": [], "posterior_vul": []}
        posterior_vul_ai = {"qids": [], "posterior_vul": []}

        # For all possible QIDs combinations
        for num_qids in np.arange(1, self.num_qids + 1):
            for qids_sel in it.combinations(self.qids, num_qids):
                qids_sel = list(qids_sel)

                # Re-identification attack
                posterior_vul_reid["qids"].append(", ".join(qids_sel))
                posterior_vul_reid["posterior_vul"].append(self.posterior_reid(qids_sel))

                # Attribute-Inference attack
                posterior_vul_ai["qids"].append(", ".join(qids_sel))
                posterior_vul_ai["posterior_vul"].append(self.posterior_ai(qids_sel))

        results_reid = {"prior": prior_vul_reid, "posterior": pd.DataFrame(posterior_vul_reid)}

        posterior_vul_ai = pd.DataFrame(posterior_vul_ai)
        # Transform the result of each sensitive attribute to a column
        for att in self.sensitive:
            posterior_vul_ai["posterior_vul_" + att] = posterior_vul_ai["posterior_vul"].apply(lambda x: x[att])
        posterior_vul_ai.drop("posterior_vul", axis=1, inplace=True)

        results_ai = {"prior": prior_vul_ai, "posterior": posterior_vul_ai}

        # Add number of qids
        results_reid["posterior"]["num_qids"] = results_reid["posterior"]["qids"].apply(lambda x: str(x).count(",") + 1)
        results_ai["posterior"]["num_qids"] = results_ai["posterior"]["qids"].apply(lambda x: str(x).count(",") + 1)

        self.results_powerset_reid = results_reid
        self.results_powerset_ai = results_ai

    def get_max_post_reid(self) -> pd.DataFrame:
        """
        Get the maximum posterior vulnerability of Re-identification attacks per number of attributes.

        Returns:
            pd.DataFrame: A DataFrame containing the maximum posterior vulnerabilities of Re-identification attacks per number of qids.
        """
        vul_column = "post_vul"
        max_post = pd.DataFrame(columns=["num_qids", "qids", vul_column])
        for num_att in np.arange(1, self.num_qids + 1):
            filter_results = self.results_powerset_reid["posterior"][
                self.results_powerset_reid["posterior"]["num_qids"] == num_att
            ]
            max_vul = filter_results.loc[filter_results[vul_column].idxmax()]
            max_post.loc[len(max_post)] = max_vul

        return max_post

    def get_max_post_ai(self, sensitive_attribute: str) -> pd.DataFrame:
        """
        Get the maximum posterior vulnerability of Attribute-Inference attacks per number of attributes for a given
        sensitive attribute.

        Args:
            sensitive_attribute (str): The sensitive attribute for which to take the maximum posterior vulnerability.

        Returns:
            pd.DataFrame: A DataFrame containing the maximum posterior vulnerabilities of Attribute-Inference attacks per number of qids.
        
        Raises:
            ValueError: If the specified sensitive attribute is not present in the dataset.
        """
        if sensitive_attribute not in self.data.columns:
            raise ValueError(f"Column {sensitive_attribute} is not in the dataset")

        vul_column = "post_vul_" + sensitive_attribute

        max_post = pd.DataFrame(columns=["num_qids", "qids", vul_column])
        for num_att in np.arange(1, self.num_qids + 1):
            posterior_results = self.results_powerset_ai["posterior"]
            filter_results = posterior_results[posterior_results["num_qids"] == num_att]
            max_vul = filter_results.loc[filter_results[vul_column].idxmax()]
            max_post.loc[len(max_post)] = max_vul

        return max_post

class ProbAttack(Attacks):
    """
    Concrete class for probability-based privacy attacks on datasets.

    Parameters:
        - dataset (Data): The dataset to be attacked.
        - qids (list): The quasi-identifiers used for the attack.
        - sensitives (list, optional): List of sensitive attributes, default is None.

    Inherits from Attacks.
    """

    def __init__(self, dataset: Data, qids: list, sensitives=None) -> None:
        """
        Initializes an instance of the ProbAttack class.

        Args:
            dataset (Data): The dataset to be attacked.
            qids (list): The quasi-identifiers used for the attack.
            sensitives (list, optional): List of sensitive attributes, default is None.
        """
        super().__init__(dataset, qids, sensitives)

    def prior_reid(self) -> float:
        """
        Calculates the prior Re-identification attack probability.

        Returns:
            float: The prior Re-identification attack probability.
        """
        return 1 / self.data.num_rows

    def posterior_reid(self, qids=None) -> float:
        """
        Calculates the posterior Re-identification attack probability.

        Args:
            qids (list, optional): List of quasi-identifiers used to perform the Re-identification attack.
                If not provided (default is None), all the qids specified in the initialization of the class will be considered.

        Returns:
            float: The posterior probability of a Re-identification attack.
            
        Returns:
            float: The posterior Re-identification attack probability.
        """
        if qids is None:
            qids = self.qids

        partitions = self.data.dataframe.groupby(qids).size().to_numpy()
        _, counts = np.unique(partitions, return_counts=True)
        posterior_prob = counts.sum() / self.data.num_rows
        return posterior_prob

    def prior_ai(self) -> dict:
        """
        Calculates the prior Attribute-Inference attack probabilities.

        Returns:
            dict: A dictionary containing the prior attack probabilities for each sensitive attribute.
        Raises:
            NameError: If sensitives is not defined.
        """
        if self.sensitive is None:
            raise NameError("sensitives is not declared.")
        
        results = dict()
        for sens in self.sensitive:
            results[sens] = self.data.dataframe[sens].value_counts().max() / self.data.num_rows
        return results

    def posterior_ai(self, qids=None) -> dict:
        """
        Calculates the posterior Attribute-Inference attack probabilities.

        Args:
            qids (list, optional): List of quasi-identifiers used to perform the Attribute-Inference attack.
                If not provided (default is None), all the qids specified in the initialization of the class will be considered.

        Returns:
            dict: A dictionary containing the posterior attack probabilities for each sensitive attribute.

        Raises:
            NameError: If sensitives is not defined.
        """
        if qids is None:
            qids = self.qids

        if self.sensitive is None:
            raise NameError("sensitives is not declared.")
        
        results = dict()
        for sens in self.sensitive:
            partitions = self.data.dataframe.groupby(
                qids+[sens]).size().droplevel(self.sensitive).to_frame().rename(columns={0: "counts"})
            groupby_qids = partitions.groupby(qids)["counts"].agg(["max", "sum"]).reset_index()

            posterior_prob = groupby_qids["max"].sum() / self.data.num_rows
            results[sens] = posterior_prob

        return results

class DetAttack(Attacks):
    """
    Concrete class for deterministic privacy attacks on datasets.

    Parameters:
        - dataset (Data): The dataset to be attacked.
        - qids (list): The quasi-identifiers used for the attack.
        - sensitives (list, optional): List of sensitive attributes, default is None.

    Inherits from Attacks.
    """

    def __init__(self, dataset: Data, qids: list, sensitives=None) -> None:
        """
        Initializes an instance of the DetAttack class.

        Args:
            dataset (Data): The dataset to be attacked.
            qids (list): The quasi-identifiers used for the attack.
            sensitives (list, optional): List of sensitive attributes, default is None.
        """
        super().__init__(dataset, qids, sensitives)

    def prior_reid(self) -> float:
        """
        Calculates the prior Re-identification attack probability.

        Returns:
            float: The prior Re-identification attack probability.
        """
        return 1 if self.data.num_rows == 1 else 0

    def posterior_reid(self, qids=None) -> float:
        """
        Calculates the posterior Re-identification attack probability.

        Args:
            qids (list, optional): List of quasi-identifiers used to perform the Re-identification attack.
                If not provided (default is None), all the qids specified in the initialization of the class will be considered.

        Returns:
            float: The posterior Re-identification attack probability.
        """
        if qids is None:
            qids = self.qids

        partitions = self.data.dataframe.groupby(qids).size().to_numpy()
        partition_size, counts = np.unique(partitions, return_counts=True)
        posterior_prob = counts[np.where(partition_size == 1)] / self.data.num_rows
        return posterior_prob[0] if len(posterior_prob) >= 1 else 0

    def prior_ai(self) -> dict:
        """
        Calculates the prior Attribute-Inference attack probabilities.
        
        Returns:
            dict: A dictionary containing the prior attack probabilities for each sensitive attribute.

        Raises:
            ValueError: If sensitives is not defined.
        """
        if self.sensitive is None:
            raise ValueError("sensitives not defined")
        
        results = dict()
        for sens in self.sensitive:
            if len(self.data.dataframe[sens].unique()) > 1:
                results[sens] = 0
            else:
                results[sens] = 1
        return results

    def posterior_ai(self, qids=None) -> dict:
        """
        Calculates the posterior Attribute-Inference attack probabilities.
        
        Args:
            qids (list, optional): List of quasi-identifiers used to perform the Attribute-Inferece attack.
                If not provided (default is None), all the qids specified in the initialization of the class will be considered.

        Returns:
            dict: A dictionary containing the posterior attack probabilities for each sensitive attribute.

        Raises:
            ValueError: If sensitives is not defined.
        """
        if qids is None:
            qids = self.qids

        if self.sensitive is None:
            raise ValueError("sensitives not defined")
        
        results = dict()
        for sens in self.sensitive:
            partitions = self.data.dataframe.groupby(self.qids + [sens]).size().droplevel(self.sensitive).to_frame().rename(columns={0: "counts"})
            groupby_qids = partitions.groupby(self.qids)["counts"].agg(["max", "sum"]).reset_index()
            posterior_prob = groupby_qids[(groupby_qids["max"] == groupby_qids["sum"])]["max"].sum() / self.data.num_rows
            results[sens] = posterior_prob

        return results

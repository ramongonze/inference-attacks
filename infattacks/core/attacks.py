import numpy as np
import pandas as pd
from humanize import intword
import matplotlib.pyplot as plt
from infattacks.core.data import Data
from abc import ABC, abstractmethod
import itertools as it

class Attacks(ABC):
    """
    Abstract base class for privacy attacks on datasets.

    Parameters:
        - dataset (Data): The dataset to be attacked.
        - qids (list): The quasi-identifiers used for the attack.
        - sensitive (list, optional): List of sensitive attributes, default is None.

    Attributes:
        - dataset (Data): The dataset being attacked.
        - qids (list): The quasi-identifiers used for the attack.
        - sensitive (list): List of sensitive attributes.
    """

    def __init__(self, data: Data, qids: list, sensitive=None) -> None:
        """
        Initialize an instance of the ReidentificationAttacker class.

        Args:
            dataset (Data): The dataset to be attacked.
            qids (list): The quasi-identifiers used for the attack.
            sensitive (list, optional): List of sensitive attributes, default is None.

        Attributes:
            - data (Data): The dataset being attacked.
            - qids (list): The quasi-identifiers used for the attack.
            - sensitive (list): List of sensitive attributes.
            - num_qids (int): The number of quasi-identifiers.
            - num_sensitive (int): The number of sensitive attributes.
            - results_powerset_reid (None or dict): Results of re-identification attacks on the power set of qids.
            - results_powerset_ai (None or dict): Results of attribute-inference attacks on the power set of qids.
            - type (str): Type of attack, "prob" or "det", for probabilistic or deterministic attack, respectivelly.

        Note:
            - The 'sensitive' parameter is optional. If not provided, it defaults to None.
            - 'results_powerset_reid' and 'results_powerset_ai' are initially set to None and can be populated
            after calling the 'attack_power_set' method.
        """
        self.data = data
        self.qids = qids
        self.sensitive = sensitive
        self.num_qids = len(self.qids)
        self.num_sensitive = len(self.sensitive)
        self.results_powerset_reid = None
        self.results_powerset_ai = None
        self.type = None

    @abstractmethod
    def _define_type(self) -> None:
        """
        Abstract method to define if the attack is Deterministic or Probabilistic.
        The attribute "type" must be set as "prob" or "det".
        """

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
        vul_column = "posterior_vul"
        max_post = pd.DataFrame(columns=["num_qids", "qids", vul_column])
        for num_att in np.arange(1, self.num_qids + 1):
            filter_results = self.results_powerset_reid["posterior"][
                self.results_powerset_reid["posterior"]["num_qids"] == num_att
            ]
            max_vul = filter_results.loc[filter_results[vul_column].idxmax()]
            max_post.loc[len(max_post)] = max_vul

        return max_post

    def get_max_post_ai(self, sensitive: str) -> pd.DataFrame:
        """
        Get the maximum posterior vulnerability of Attribute-Inference attacks per number of attributes for a given
        sensitive attribute.

        Args:
            sensitive (str): The sensitive attribute for which to take the maximum posterior vulnerability.

        Returns:
            pd.DataFrame: A DataFrame containing the maximum posterior vulnerabilities of Attribute-Inference attacks per number of qids.
        
        Raises:
            ValueError: If the specified sensitive attribute is not present in the dataset.
        """
        if sensitive not in self.data.columns:
            raise ValueError(f"Column {sensitive} is not in the dataset")

        vul_column = "posterior_vul_" + sensitive

        max_post = pd.DataFrame(columns=["num_qids", "qids", vul_column])
        for num_att in np.arange(1, self.num_qids + 1):
            posterior_results = self.results_powerset_ai["posterior"]
            filter_results = posterior_results[posterior_results["num_qids"] == num_att]
            max_vul = filter_results.loc[filter_results[vul_column].idxmax()]
            max_post.loc[len(max_post)] = max_vul

        return max_post

    def save_posterior_reid(self, file_name) -> None:
        """
        Save the results of posterior vulnerability generated by attack_power_set for Re-identification attacks.

        This function exports the posterior vulnerability results for Re-identification attacks, which were
        generated by the 'attack_power_set' method, to a CSV file.

        Args:
            file_name (str): The name of the CSV file to save the results.

        Returns:
            None


        Note:
            - The CSV file will contain columns such as 'num_qids', 'qids', and 'posterior_vul' with posterior vulnerability results.
        """    
        self.results_powerset_reid["posterior"].to_csv(
            file_name,
            float_format="%.8f",
            index=False
        )

    def save_posterior_ai(self, file_name) -> None:
        """
        Save the results of posterior vulnerability generated by attack_power_set for Attribute-Inference attacks.

        This function exports the posterior vulnerability results for Attribute-Inference attacks, which were
        generated by the 'attack_power_set' method, to a CSV file.

        Args:
            file_name (str): The name of the CSV file to save the results.

        Returns:
            None

        Note:
            - The CSV file will contain columns such as 'num_qids', 'qids', and 'posterior_vul' with posterior vulnerability results.
        """
        self.results_powerset_ai["posterior"].to_csv(
            file_name,
            float_format="%.8f",
            index=False
        )

    def save_posteriors(self, reid_file_name, ai_file_name) -> None:
        """
        Save both Re-identification and Attribute-Inference posterior vulnerability results.

        This function calls 'save_posterior_reid' and 'save_posterior_ai' to save the results of posterior vulnerability
        for Re-identification and Attribute-Inference attacks, respectively.

        Args:
            reid_file_name (str): The name of the CSV file to save Re-identification results.
            ai_file_name (str): The name of the CSV file to save Attribute-Inference results.

        Returns:
            None

        Note:
            - Each CSV file will contain columns such as 'num_qids', 'qids', and 'posterior_vul' with posterior vulnerability results.
        """
        self.save_posterior_reid(reid_file_name)
        self.save_posterior_ai(ai_file_name)
   
    def plot_graph_reid(self, show_graph=True, save_graph=None, format=None) -> None:
        """
        Plot a graph depicting the vulnerabilities in Re-identification attacks.

        Args:
            show_graph (bool, optional): Whether to display the graph. Default is True.
            save_graph (str, optional): File path to save the graph. If None, the graph won't be saved. Default is None.
            format (str, optional): The file format for saving the graph (e.g., 'png', 'pdf'). Applicable only if `save_graph` is provided.

        Note:
            - The graph illustrates the prior and posterior vulnerabilities for Re-identification attacks.
            - The x-axis represents the number of quasi-identifiers (QIDs), while the y-axis represents the vulnerability.
            - The graph includes both prior (baseline) and posterior (calculated) vulnerabilities.
            - The prior vulnerability is represented by a horizontal line labeled "Prior."
            - The posterior vulnerabilities are shown as scattered points, and the color intensity represents the vulnerability level.
            - The right y-axis indicates the corresponding number of individuals affected by the attack.
            - The graph can be displayed, saved to a file, or both based on the specified parameters.
        """
        prior = self.results_powerset_reid["prior"]
        plt.hlines(prior, 1, self.results_powerset_reid["posterior"]["num_qids"].max(), label="Prior")
        
        x = self.results_powerset_reid["posterior"]["num_qids"]
        y = self.results_powerset_reid["posterior"]["posterior_vul"]

        if self.type == "prob":
            title = "Probabilistic"
        elif self.type == "det":
            title = "Deterministic"

        ax1 = plt.subplot()
        plt.scatter(x, y, c=y, cmap="bwr", alpha=0.5, s=70, label="Posterior")
        plt.xlabel("Number of QIDs")
        plt.ylabel("Vulnerability")
        plt.ylim((0,1))
        plt.yticks(np.array(range(0,101,10))/100)

        ax2 = ax1.twinx()
        ax2.ticklabel_format(style="plain")
        ax2.scatter(x, np.array(y)*self.data.num_rows, alpha=0)
        plt.ylabel("Number of Individuals")
        plt.ylim((0,self.data.num_rows))

        plt.xticks(range(1,self.results_powerset_reid["posterior"]["num_qids"].max()+1))
        plt.title(title + " Re-identification Attack")
        ax1.legend()
        ax1.grid(.3, linestyle="--")
        
        fig = plt.gcf()
        fig.set_size_inches(10, 6)

        if save_graph:
            plt.savefig(save_graph, format=format, bbox_inches="tight")
        
        if show_graph:
            plt.show()

    def plot_graph_ai(self, sensitive:str, show_graph=True, save_graph=None, format=None) -> None:
        """
        Plot a graph for Attribute-Inference attacks.

        This function generates and displays a graph illustrating the prior and posterior vulnerability
        for Attribute-Inference attacks on a specific sensitive attribute across different numbers of quasi-identifiers (QIDs).

        Args:
            sensitive (str): The sensitive attribute for which to plot the graph.
            show_graph (bool, optional): Whether to display the graph. Default is True.
            save_graph (str, optional): File path to save the graph. If None, the graph won't be saved. Default is None.
            format (str, optional): The file format for saving the graph (e.g., 'png', 'pdf'). Applicable only if `save_graph` is provided.

        Note:
            - The graph includes the prior vulnerability (horizontal line) and the posterior vulnerability (scatter plot).
            - The x-axis represents the number of QIDs, and the y-axis represents the vulnerability.
            - The right y-axis shows the corresponding number of individuals affected.
        """
        prior = self.results_powerset_ai["prior"][sensitive]
        plt.hlines(prior, 1, self.results_powerset_ai["posterior"]["num_qids"].max(), label="Prior")
        
        x = self.results_powerset_ai["posterior"]["num_qids"]
        y = self.results_powerset_ai["posterior"]["posterior_vul_"+sensitive]

        if self.type == "prob":
            title = "Probabilistic"
        elif self.type == "det":
            title = "Deterministic"

        ax1 = plt.subplot()
        plt.scatter(x, y, c=y, cmap="bwr", alpha=0.5, s=70, label="Posterior")
        plt.xlabel("Number of QIDs")
        plt.ylabel("Vulnerability")
        plt.ylim((0,1))
        plt.yticks(np.array(range(0,101,10))/100)

        ax2 = ax1.twinx()
        ax2.ticklabel_format(style="plain")
        ax2.scatter(x, np.array(y)*self.data.num_rows, alpha=0)
        plt.ylabel("Number of Individuals")
        plt.ylim((0,self.data.num_rows))

        plt.xticks(range(1,self.results_powerset_ai["posterior"]["num_qids"].max()+1))
        plt.title(title + f" Attribute-Inference Attack - {sensitive}")
        ax1.legend()
        ax1.grid(.3, linestyle="--")
        
        fig = plt.gcf()
        fig.set_size_inches(10, 6)

        if save_graph:
            plt.savefig(save_graph, format=format, bbox_inches="tight")
        
        if show_graph:
            plt.show()

    def _gen_histogram(self, partition_vul, counts, hist_bin_size) -> np.ndarray:
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

    def plot_histogram(self, histogram_counts, title, x_label, y_label, hist_bin_size=5):
        histogram = dict()
        for i in range(len(histogram_counts)):
            if i < len(histogram_counts) - 1:
                histogram[f"[{i * (hist_bin_size / 100):.2f},{(i + 1) * (hist_bin_size / 100):.2f})"] = int(histogram_counts[i])
            else:
                histogram[f"[{i * (hist_bin_size / 100):.2f},{(i + 1) * (hist_bin_size / 100):.2f}]"] = int(histogram_counts[i])

        x_labels = histogram.keys()
        x = list(range(len(x_labels)))
        y = list(histogram.values())
        plt.bar(x,y)
        plt.xticks(x, x_labels, rotation=90)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        

class ProbAttack(Attacks):
    """
    Concrete class for probability-based privacy attacks on datasets.

    Parameters:
        - dataset (Data): The dataset to be attacked.
        - qids (list): The quasi-identifiers used for the attack.
        - sensitive (list, optional): List of sensitive attributes, default is None.

    Inherits from Attacks.
    """

    def __init__(self, dataset: Data, qids: list, sensitive=None) -> None:
        """
        Initializes an instance of the ProbAttack class.

        Args:
            dataset (Data): The dataset to be attacked.
            qids (list): The quasi-identifiers used for the attack.
            sensitive (list, optional): List of sensitive attributes, default is None.
        """
        super().__init__(dataset, qids, sensitive)
        self._define_type()

    def _define_type(self) -> None:
        """
        Defines the type of the attack.
        """
        self.type = "prob"

    def prior_reid(self) -> float:
        """
        Calculates the prior Re-identification attack probability.

        Returns:
            float: The prior Re-identification attack probability.
        """
        return 1 / self.data.num_rows

    def posterior_reid(self, qids=None, hist=False, hist_bin_size=5):
        """
        Calculate the posterior Re-identification attack probability for a given dataset.

        Parameters:
            - dataset (DataFrame): The dataset to be analyzed.
            - qids (list, optional): List of quasi-identifiers. If not provided, all columns will be used.
            - hist (bool, optional): Whether to generate a histogram of Re-identification probabilities. Default is False.
            - hist_bin_size (int, optional): Bin size for the histogram if hist is True. Default is 5.

        Returns:
            float or tuple: If hist is False, returns the expected posterior Re-identification attack probability.
            If hist is True, returns a tuple containing the expected posterior probability and a histogram dictionary.
        """
        if qids is None:
            qids = self.qids

        partitions = self.data.dataframe.groupby(qids).size().to_numpy()
        expected_posterior_prob = len(partitions) / self.data.num_rows

        if hist:
            histogram = self._gen_histogram(1/np.array(partitions), partitions, hist_bin_size)
            return expected_posterior_prob, histogram
        
        return expected_posterior_prob

    def prior_ai(self) -> dict:
        """
        Calculates the prior Attribute-Inference attack probabilities.

        Returns:
            dict: A dictionary containing the prior attack probabilities for each sensitive attribute.
        Raises:
            NameError: If sensitive is not defined.
        """
        if self.sensitive is None:
            raise NameError("sensitive is not declared.")
        
        results = dict()
        for sens in self.sensitive:
            results[sens] = self.data.dataframe[sens].value_counts().max() / self.data.num_rows
        return results

    def posterior_ai(self, qids=None, hist=False, hist_bin_size=5):
        if qids is None:
            qids = self.qids

        if self.sensitive is None:
            raise NameError("sensitive is not declared.")
        
        # max = size of largest partition
        # sum = number of people in all partitions (for that combination of qids + [sens])
        expectations = dict()
        histograms = dict()
        for sens in self.sensitive:
            partitions = self.data.dataframe.groupby(
                qids+[sens]).size().droplevel(self.sensitive).to_frame().rename(columns={0: "counts"})
            groupby_qids = partitions.groupby(qids)["counts"].agg(["max", "sum"]).reset_index()
            posterior_prob = groupby_qids["max"].sum() / self.data.num_rows
            if hist:
                counts = groupby_qids["sum"].to_numpy()
                partition_vul = groupby_qids["max"].to_numpy() / counts
                histograms[sens] = self._gen_histogram(partition_vul, counts, hist_bin_size)

            expectations[sens] = posterior_prob

        if hist:
            return expectations, histograms
        
        return expectations

class DetAttack(Attacks):
    """
    Concrete class for deterministic privacy attacks on datasets.

    Parameters:
        - dataset (Data): The dataset to be attacked.
        - qids (list): The quasi-identifiers used for the attack.
        - sensitive (list, optional): List of sensitive attributes, default is None.

    Inherits from Attacks.
    """

    def __init__(self, dataset: Data, qids: list, sensitive=None) -> None:
        """
        Initializes an instance of the DetAttack class.

        Args:
            dataset (Data): The dataset to be attacked.
            qids (list): The quasi-identifiers used for the attack.
            sensitive (list, optional): List of sensitive attributes, default is None.
        """
        super().__init__(dataset, qids, sensitive)
        self._define_type()

    def _define_type(self) -> None:
        """
        Defines the type of the attackl.
        """
        self.type = "det"

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
            ValueError: If sensitive is not defined.
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

    def posterior_ai(self, qids=None) -> dict:
        """
        Calculates the posterior Attribute-Inference attack probabilities.
        
        Args:
            qids (list, optional): List of quasi-identifiers used to perform the Attribute-Inferece attack.
                If not provided (default is None), all the qids specified in the initialization of the class will be considered.

        Returns:
            dict: A dictionary containing the posterior attack probabilities for each sensitive attribute.

        Raises:
            ValueError: If sensitive is not defined.
        """
        if qids is None:
            qids = self.qids

        if self.sensitive is None:
            raise ValueError("sensitive not defined")
        
        results = dict()
        for sens in self.sensitive:
            partitions = self.data.dataframe.groupby(qids + [sens]).size().droplevel(self.sensitive).to_frame().rename(columns={0: "counts"})
            groupby_qids = partitions.groupby(qids)["counts"].agg(["max", "sum"]).reset_index()
            posterior_prob = groupby_qids[(groupby_qids["max"] == groupby_qids["sum"])]["max"].sum() / self.data.num_rows
            results[sens] = posterior_prob

        return results

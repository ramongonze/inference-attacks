import os
import numpy as np
from humanize import intword
import matplotlib.pyplot as plt
import infattacks
from matplotlib.colors import Normalize, LinearSegmentedColormap

class VisualizeRisk:
    """
    A class for visualizing risk data.

    This class provides methods to visualize data generated by an Attack class. It's possible to make scatter plots to see the vulnerability for different combinations of QIDs and bar plot for the individual posterior vulnerabilities  for a given combination of QIDs.
    """

    def __init__(self, attack:infattacks.attacks.Attack):
        """Initialize the VisualizeRisk object.

        Parameters:
            attack (Attack): Attack object.
        """
        self.attack = attack
    
    def _build_graph(self, title, atk_type, x, y, show_graph, save_graph, language):
        cmap = LinearSegmentedColormap.from_list("red_to_blue", [(1, 0, 0), (0, 0, 1)])

        # Plot for left y-axis (proportion of people)
        y_left = y
        ax1 = plt.subplot()
        cmap = LinearSegmentedColormap.from_list("blue_to_red", [(0, 0, 1), (1, 0, 0)])
        norm = Normalize(vmin=min(y_left), vmax=max(y_left))
        plt.scatter(x, y_left, c=y_left, cmap=cmap, norm=norm, alpha=0.5, s=70, label="Posterior")
        if language == "en":
            plt.xlabel("Number of QIDs")
            if atk_type == "det":
                plt.ylabel("Proportion of individuals")
            else:
                plt.ylabel("Vulnerability")
        elif language == "pt":
            plt.xlabel("Número de QIDs")
            if atk_type == "prob":
                plt.ylabel("Proporção de indivíduos")
            else:
                plt.ylabel("Vulnerabilidade")
                
        plt.ylim((0,1))
        plt.yticks(np.array(range(0,101,10))/100)

        # Plot for right y-axis (number of people)
        y_right = np.array(y)*self.attack.data.num_rows
        ax2 = ax1.twinx()
        ax2.ticklabel_format(style="plain")
        ax2.scatter(x, y_right, alpha=0)
        if language == "en":
            if atk_type == "det":
                plt.ylabel("Number of individuals")

        elif language == "pt":
            if atk_type == "det":
                plt.ylabel("Número de indivíduos")

        plt.ylim((0,self.attack.data.num_rows))
        plt.xticks(range(x.min(),x.max()+1))

        plt.title(title)
        ax1.legend()
        ax1.grid(.3, linestyle="--")
        
        if save_graph:
            _, format = os.path.splitext(save_graph)
            plt.savefig(save_graph, format=format[1:].lower(), bbox_inches="tight")
        
        if show_graph:
            plt.show()

        plt.cla()
        plt.clf()

    def _build_hist(self, title, hist_bin_size, hist_counts, show_hist, save_hist, language):
        histogram = dict()
        for i in range(len(hist_counts)):
            if i < len(hist_counts) - 1:
                histogram[f"[{i * (hist_bin_size / 100):.2f},{(i + 1) * (hist_bin_size / 100):.2f})"] = int(hist_counts[i])
            else:
                histogram[f"[{i * (hist_bin_size / 100):.2f},{(i + 1) * (hist_bin_size / 100):.2f}]"] = int(hist_counts[i])

        x_labels = histogram.keys()
        x = list(range(len(x_labels)))
        y_right = np.array(list(histogram.values())) # Absolute 
        y_left = y_right/y_right.sum() # Proportions

        # Plot for left y-axis (proportion of people)
        ax1 = plt.subplot()
        ax1.grid(.3, linestyle="--")
        cmap = LinearSegmentedColormap.from_list("blue_to_red", [(0, 0, 1), (1, 0, 0)])
        norm = Normalize(vmin=0, vmax=len(x_labels))
        plt.bar(x, y_left, color=cmap(norm(x)))
        if language == "en":
            plt.xlabel("Posterior vulnerability")
            plt.ylabel("Proportion of individuals")
        elif language == "pt":
            plt.xlabel("Vulnerabilidade a posteriori")
            plt.ylabel("Proporção de indivíduos")
        plt.xticks(x, x_labels, rotation=90)
        plt.ylim((0,1))
        plt.yticks(np.array(range(0,101,10))/100)

        # Plot for right y-axis (number of people)
        ax2 = ax1.twinx()
        # ax2.ticklabel_format(style="plain")
        ax2.bar(x,y_left, alpha=0)
        
        if language == "en":
            plt.ylabel("Number of individuals")
        elif language == "pt":
            plt.ylabel("Número de indivíduos")
        plt.ylim((0,self.attack.data.num_rows))

        plt.title(title)

        fig = plt.gcf()
        fig.set_size_inches(10, 6)

        if save_hist:
            _, format = os.path.splitext(save_hist)
            plt.savefig(save_hist, format=format[1:].lower(), bbox_inches="tight")
        
        if show_hist:
            plt.show()

        plt.cla()
        plt.clf()

    def plot_comb_qids_reid(self, title:str, atk_type=str, show_graph=True, save_graph=None, language="en") -> None:
        """
        Plot a graph depicting the vulnerabilities in Re-identification attacks generated by Attack.post_comb_qids_reid.

        Parameters:
            title (str): Graph's title.
            atk_type (str): Must be "prob" for probabilistic attacks and "det" for deterministic attacks. 
            show_graph (bool, optional): Whether to display the graph on the screen. Default is True.
            save_graph (str, optional): File path to save the graph (including the format). If None, the graph won't be saved. Default is None.
            language (str, optinal): Language for axis' labels. Default "en", accepts also "pt".

        Note:
            - The graph illustrates the prior and posterior vulnerabilities for Re-identification attacks.
            - The x-axis represents the number of quasi-identifiers (QIDs), while the y-axis represents the vulnerability.
            - The graph includes both prior (baseline) and posterior (calculated) vulnerabilities.
            - The graph can be displayed, saved to a file, or both based on the specified parameters.
        """
        prior = self.attack.prior_reid()
        plt.hlines(
            prior,
            self.attack.result_comb_qids_reid["num_qids"].min(),
            self.attack.result_comb_qids_reid["num_qids"].max(),
            label="Prior"
        )
        
        x = self.attack.result_comb_qids_reid["num_qids"]
        y = self.attack.result_comb_qids_reid["post_vul"]
        self._build_graph(title, atk_type, x, y, show_graph, save_graph, language)

    def plot_comb_qids_ai(self, title:str, atk_type:str, sensitive:str, show_graph=True, save_graph=None, language="en") -> None:
        """
        Plot a graph for Attribute-inference attacks.

        This function generates and displays a graph illustrating the prior and posterior vulnerability
        for Attribute-inference attacks on a specific sensitive attribute across different numbers of quasi-identifiers (QIDs).

        Parameters:
            title (str): Graph's title.
            atk_type (str): Must be "prob" for probabilistic attacks and "det" for deterministic attacks. 
            sensitive (str): The sensitive attribute for which to plot the graph.
            show_graph (bool, optional): Whether to display the graph on the screen. Default is True.
            save_graph (str, optional): File path to save the graph (including the format). If None, the graph won't be saved. Default is None.
            language (str, optinal): Language for axis' labels. Default "en", accepts also "pt".

        Note:
            - The graph illustrates the prior and posterior vulnerabilities for Attribute-inference attacks.
            - The x-axis represents the number of quasi-identifiers (QIDs), while the y-axis represents the vulnerability.
            - The graph includes both prior (baseline) and posterior (calculated) vulnerabilities.
            - The graph can be displayed, saved to a file, or both based on the specified parameters.
        """
        if self.attack.num_sensitive == 0:
            raise ValueError("No sensitive attribute was defined")
        
        prior = self.attack.prior_ai()[sensitive]
        plt.hlines(
            prior,
            self.attack.result_comb_qids_ai["num_qids"].min(),
            self.attack.result_comb_qids_ai["num_qids"].max(),
            label="Prior"
        )
        
        x = self.attack.result_comb_qids_ai["num_qids"]
        y = self.attack.result_comb_qids_ai["post_vul_"+sensitive]
        self._build_graph(title, atk_type, x, y, show_graph, save_graph, language)

    def plot_hist_reid(self, title:str, qids, hist_bin_size=5, show_hist=True, save_hist=None, language="en"):
        """
        Plot a histogram of individual posterior vulnerabilities for Re-identification attacks for a given set of QIDs.

        Parameters:
            title (str): Histogram's title.
            qids (list, tuple, numpy.array): List of qids to make the attack and plot the histogram.
            hist_bin_size (int, optional): Size of each histogram bin, in percentage. Defaults to 5.
            show_hist (bool, optional): Whether to display the histogram on the screen. Default is True.
            save_hist (str, optional): File path to save the histogram (including the format). If None, the histogram won't be saved. Default is None.
            language (str, optinal): Language for axis' labels. Default "en", accepts also "pt".

        Note:
            - The x-axis labels represent the posterior vulnerability (i.e., intervals of probabilities).
            - The y-axis represents the number of people with the respective posterior vulnerability in a given bin.
        """
        _, hist_counts = self.attack.post_reid(qids=qids, hist=True)
        self._build_hist(title, hist_bin_size, hist_counts, show_hist, save_hist, language)

    def plot_hist_ai(self, title:str, qids, sensitive:str, hist_bin_size=5, show_hist=True, save_hist=None, language="en"):
        """
        Plot a histogram of individual posterior vulnerabilities for Attribute-Inference attack for a given set of QIDs and a given sensitive attribute.

        Parameters:
            title (str): Histogram's title.
            qids (list, tuple, numpy.array): List of qids to make the attack and plot the histogram.
            sensitive (str): Sensitive attribute.
            hist_bin_size (int, optional): Size of each histogram bin, in percentage. Defaults to 5.
            show_hist (bool, optional): Whether to display the histogram on the screen. Default is True.
            save_hist (str, optional): File path to save the histogram (including the format). If None, the histogram won't be saved. Default is None.
            language (str, optinal): Language for axis' labels. Default "en", accepts also "pt".

        Note:
            - The x-axis labels represent the posterior vulnerability (i.e., intervals of probabilities).
            - The y-axis represents the number of people with the respective posterior vulnerability in a given bin.
        """
        if self.attack.num_sensitive == 0:
            raise ValueError("No sensitive attribute was defined")
        
        _, hist_counts = self.attack.post_ai(qids=qids, hist=True)
        hist_counts = hist_counts[sensitive]
        self._build_hist(title, hist_bin_size, hist_counts, show_hist, save_hist, language)

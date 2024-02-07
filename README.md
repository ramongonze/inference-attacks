# Inference Attacks Library

## Overview

Welcome to the Inference Attacks library, a comprehensive toolset designed to facilitate the implementation and analysis of privacy inference attacks on datasets. This library offers a robust and extensible framework to conduct various types of privacy attacks, providing users with insights into potential vulnerabilities within their datasets.

## Installation

To integrate the Inference Attacks library into your project, use the following pip command:

```bash
pip install infattacks
```

## Example of usage
```python
import infattacks

# Example: Create a Dataset instance from a CSV file
dataset = infattacks.Data(file_name="path/to/your/dataset.csv", sep_csv=",")

# Attack
qids = ["age", "education"]
sensitive = ["income"]

attack = infattacks.Probabilistic(dataset, qids=qids, sensitive=sensitive)
prior_reid = attack.prior_reid()
prior_ai = attack.prior_ai()
exp_reid, hist_reid = attack.post_reid(qids=qids, hist=True)
exp_ai, hist_ai = attack.post_ai(qids=qids, hist=True)

print("[Re-identification]")
print("Prior vulnerability: %.6f"%(prior_reid))
print("Posterior vulnerability: %.6f"%(exp_reid))

print("\n[Attribute-Inference]")
for att in sensitive:
    print("Prior vulnerability [%s]: %.6f"%(att, prior_ai[att]))
    print("Posterior vulnerability [%s]: %.6f"%(att, exp_ai[att]))

vis = infattacks.VisualizeRisk(attack)
vis.plot_hist(hist_reid, "Re-identification - QIDs: [age,education]")
vis.plot_hist(hist_ai["income"], "Attribute-Inference - QIDs: [age,education] - sensitive: income")
```

## License

This library is distributed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it according to the terms outlined in the license.
# paper2synthesis

# Overview

It’s crucial for lawmakers, regulators, and policymakers to base climate decisions on scientific evidence and to assess current climate policies through the lens of this evidence. Evaluating climate policies helps improve regulations, fosters transparency, and builds public trust. It also motivates both public and private sectors to make commitments and take stronger actions.

Academic research offers valuable insights for climate policy evaluation. With the vast amount of scientific studies available, it's important to systematically review and summarize the current scientific understanding. The first step in this process is to identify the most relevant studies for specific policies.

In this notebook, I'll demonstrate how Natural Language Processing (NLP) can be used to identify and map climate-related research using a supervised learning approach.

Here’s the structure of the notebook:
1. First, I’ll explore the dataset and create a basic machine learning pipeline.
2. Next, I’ll train a more advanced model using a supervised SVM classifier.
3. Finally, I’ll train and evaluate pretrained transformer models.

## Problem: The Challenge of Comprehensive Assessments

In the past, early IPCC reports dealt with only a few hundred or thousand studies on climate change. However, in the most recent assessment, over 300,000 studies were published on the subject ([Callaghan et al., 2020](https://www.nature.com/articles/s41558-019-0684-5)). While these reports have expanded in terms of authors, pages, and references, they haven't kept pace with the sheer growth in the available literature. As a result, the IPCC now cites a much smaller portion of the relevant studies, raising important questions about which studies are chosen and how they are selected.
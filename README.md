# Base64 command line token detector

One of the evation techniques that the threat actors often use is to encode the malicious payload/script in the command line with base64 format (see [here](https://redcanary.com/blog/investigating-powershell-attacks/) or [here](https://azure.microsoft.com/en-us/blog/learning-from-cryptocurrency-mining-attack-scripts-on-linux/)). In this project, we try to train a machine learning model to detect whether such base64 token appears in the command line string.

## Approach

- Self-supervised learning
- Model: recurrent neural networks (in particular, GRU), implemented using PyTorch

## Data

(TBA)

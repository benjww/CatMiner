This directory (specifically the main.py file) demonstrates how multiple operating conditions can be specified and targeted for extraction in a single CatMiner run. In the main.py file, you can see that both "reaction operating temperature" and "gas hourly space velocity" are each defined in the conditions list. 

Before running main.py to generate it, you must:

1. Populate the corresponding api.env file with the necessary information, including a valid API key, as described below.
2. Copy preprocessed text files that make up the source text into the corresponding /OCM-papers/ directory. These files can be obtained by, e.g., running the scripts and following the instructions included in the [/download_papers/test_set/](https://github.com/benjww/text_mining_prep/tree/main/download_papers/test_set/) and [/preprocessing/test_set/](https://github.com/benjww/text_mining_prep/tree/main/preprocessing/test_set/) directories in our sibling [text_mining_prep repository](https://github.com/benjww/text_mining_prep/tree/main/).

Llama 3.1 405B is employed in this example. It is called using the Amazon Bedrock service provided by Amazon Web Services. Documentation for this cloud service can be found [here](https://docs.aws.amazon.com/bedrock/). In the provided api.env files you must enter your desired model ID, AWS Region, and two confidential API keys that can be requested upon creation of an AWS account. Model IDs and supported AWS Regions can be found [here](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html). 

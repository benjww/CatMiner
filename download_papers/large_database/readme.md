The scripts here can be ran to identify the publishers associated with the 1,029 papers in our full OCM corpus and download the subset of Elsevier papers in XML format. For information about downloading Royal Society of Chemistry papers, see the instructions in the [parent directory](https://github.com/benjww/text_mining_prep/tree/main/download_papers/). 

Run these scripts in the following order:

1. get-pub-info-from-dois.py
2. parse-elsevier-dois.py
3. get-xml-from-elsevier-dois.py

Script 3 requires an Elsevier text mining API key to execute, which can be requested via the developer portal [here](https://dev.elsevier.com/). Once you obtain an API key, please replace the text that reads PLACEHOLDER with the appropriate key string. 

Script 1 employs the Crossref API. Notably, this does not require a key. However, best practices should be obeyed and can be found [here](https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/).

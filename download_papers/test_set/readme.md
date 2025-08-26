These three scripts are used to download the open-access Elsevier and Springer Nature papers contained in our test set, from which we extract our handmade database. For information about downloading Royal Society of Chemistry papers, see the instructions in the [parent directory](https://github.com/benjww/text_mining_prep/tree/main/download_papers/). 

Run these scripts in the following order:

1. get-pub-info-from-dois.py
2. get-xml-from-elsevier-dois.py
3. get-xml-from-sn-dois.py

Scripts 2 and 3 require Elsevier and Springer Nature text mining API keys to execute, respectively, which can be requested via the respective developer portals [here](https://dev.elsevier.com/) and [here](https://dev.springernature.com/). Once you obtain API keys, please replace the text that reads PLACEHOLDER with the appropriate key string. 

Note that in Script 3, we specifically employ the Springer Nature Open Access API, as the one API that we had access to at time of writing. 

Script 1 employs the Crossref API. Notably, this does not require a key. However, best practices should be obeyed and can be found [here](https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/).

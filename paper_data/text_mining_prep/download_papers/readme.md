These directories contain the code we used to download the papers from which we extract our 40-paper test database and 1029-paper full database. The scripts they contain are used to obtain HTML and XML format copies of these papers. 

Note we do not provide scripts to download any Royal Society of Chemistry (RSC) papers as those were downloaded manually from the web. 

HTML copies of the Royal Society of Chemistry (RSC) papers can be downloaded with permission from RSC via web crawling, or manually from the web. We provide some guidelines here:
 - For more information about web crawling permissions, RSC provides a "Get in touch" form linked [here](https://www.rsc.org/journals-books-databases/research-tools/text-and-data-mining/tdm-free-resources-and-open-source-software/).
 - To manually download an RSC paper, first view it in your web browser, e.g., by entering doi.org/ followed by the paper DOI. Then, click the "Article HTML" button on the righthand side of the page. Then right-click, select "save as", and save the file as "Webpage, HTML only". We have only verified this in Google Chrome. 
 - Older RSC papers read online often lack an associated "Article HTML" button. The article HTML webpage may still be accessed by directly entering its URL into your browser. Specifically, the associated URL has the form "pubs.rsc.org/{language}/content/articlehtml/{year}/{journal}/{doi_suffix}".
   - {language} is a two-character code indicating the language, e.g., "en"
   - {year} is the year the article was published, e.g., "1990"
   - {journal} is a two-character code associated with the journal, e.g., "c3". Codes can be found for specific journals by clicking on their associated link [here](https://pubs.rsc.org/en/journals?key=title&value=current) and inspecting the URL. 
   - {doi_suffix} is the DOI suffix, i.e., the portion of the DOI proceeding the forward slash. 

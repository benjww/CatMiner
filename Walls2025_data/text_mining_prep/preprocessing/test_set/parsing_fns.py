from chemdataextractor.reader import RscHtmlReader, ElsevierXmlReader, NlmXmlReader
from chemdataextractor import Document
import os

def rsc_to_sentences(source_path, target_path):

    ### convert RSC HTML files into lists of sentences

    ### inputs: 
    # source_path: path to a directory containing RSC HTML files [str]
    # target_path: path to a directory where sentences should be written [str]

    # define list of RSC HTML file names
    filenames = os.listdir(source_path)

    # initiate counter of times we parse an empty string
    none_parsed_count = 0
    no_ft_counter = 0
    failed_counter = 0

    print(f'Beginning preprocessing of {len(filenames)} RSC HTML files')

    for i in range(len(filenames)):

        # define file path
        file_path = source_path + filenames[i]
        #print(file_path)

        # read as Document
        try:
            doc = Document.from_file(file_path, readers=[RscHtmlReader()])
        except:
            failed_counter += 1
            print(f'Failed to parse with RscHtmlReader. The HTML file may not be valid. This has happened {failed_counter} times.') 
            continue

        for e in range(len(doc.elements)):
            if doc.elements[e].__class__.__name__ == 'Title':
                title = " ".join(str(doc.elements[e]).split())
            if "The full text of this document is currently only available in the PDF" in str(doc.elements[e]):
                no_ft_counter += 1
                print(f'Found an RSC HTML file that likely only contains the abstract. This has happened {no_ft_counter} times.')

        # determine first element to include in full text string
        start = next((i for i, element in enumerate(doc.elements) if str(element) == 'Abstract'), None) + 1

        # determine last element to include in full text string
        try:
            end = next((i for i, element in enumerate(doc.elements) if element.__class__.__name__ == 'Citation'), None) - 1
        except:
            end = len(doc.elements) - 1

        # convert all relevant doc elements to a single full text string
        fulltext = _get_fulltext_from_elements(doc.elements, start, end)

        # tokenize the full text into sentences and write to a text file
        none_parsed_count = _string_to_sentences(fulltext, filenames[i], target_path, title, none_parsed_count)

        print(f'Completed RSC HTML file {i}')


def elsevier_to_sentences(source_path, target_path):

    ### convert Elsevier XML files into lists of sentences

    ### inputs:
    # source_path: path to a directory containing Elsevier XML files [str]
    # target_path: path to a directory where sentences should be written [str]

    # define list of RSC HTML file names
    filenames = os.listdir(source_path)

    # initiate counter of times we parse an empty string
    none_parsed_count = 0

    print(f'Beginning preprocessing of {len(filenames)} Elsevier XML files')

    # initialize count of papers that lack full text for debugging purposes
    no_ft_counter = 0
    fail_counter = 0

    for i in range(len(filenames)):

        #print(filenames[i])

        # define file path
        file_path = source_path + filenames[i]

        # read as Document
        try:
            doc = Document.from_file(file_path, readers=[ElsevierXmlReader()])
        except:
            fail_counter += 1
            print(f'Failed to parse with ElsevierXmlReader. The XML file may not be valid. This has happened {fail_counter} times.')
            continue

        for e in range(len(doc.elements)):
            if doc.elements[e].__class__.__name__ == 'Title':
                title = " ".join(str(doc.elements[e]).split())

        # Determine first element to include in full text string

        # Docs parsed with ElsevierXmlReader often include the title once at the beginning, labelled as a title element...
        # and once later, immediately before the abstract. Here, we attempt to find the index of the second title element.
        # If we cannot find it, we default to starting after the single MetaData element, which appears to always be present. 
        occurrence_count = 0
        try:
            title_idx = next((j for j, element in enumerate(doc.elements) if element.__class__.__name__ == 'Title'), None)
            title = " ".join(str(doc.elements[title_idx]).split())
            for j, element in enumerate(doc.elements):
                if " ".join(str(element).split()) == title:
                    occurrence_count += 1
                    if occurrence_count == 2:
                        start = j + 1
        except: 
            print('Elsevier title element not found, starting from MetaData instead')
            start = next((j for j, element in enumerate(doc.elements) if element.__class__.__name__ == 'MetaData'), None) + 1
        
        if occurrence_count < 2: 
            print('Could not find the repeated title, starting from MetaData instead')
            start = next((j for j, element in enumerate(doc.elements) if element.__class__.__name__ == 'MetaData'), None) + 1

        # determine last element to include in full text string
        # Docs parsed with the ElseverXmlReader automatically exclude citations.
        end = len(doc.elements) - 1

        # check if the XML file contains the full text or only the abstract
        if start > end-3:
            no_ft_counter += 1
            print(f'Found an Elsevier XML file that likely only contains the abstract. This has happened {no_ft_counter} times.')

        #print(f'start: {start}')
        #print(f'end: {end}')

        # convert all relevant doc elements to a single full text string
        fulltext = _get_fulltext_from_elements(doc.elements, start, end)

        # tokenize the full text into sentences and write to a text file
        none_parsed_count = _string_to_sentences(fulltext, filenames[i], target_path, title, none_parsed_count)

        print(f'Completed Elsevier XML file {i}')


def sn_to_sentences(source_path, target_path):

    ### convert Nature XML JATS files into lists of sentences

    ### inputs: 
    # source_path: path to a directory containing Nature XML files [str]
    # target_path: path to a directory where sentences should be written [str]

    # define list of SN XML file names
    filenames = os.listdir(source_path)

    # initiate counter of times we parse an empty string
    none_parsed_count = 0

    print(f'Beginning preprocessing of {len(filenames)} SpringerNature XML files')

    for i in range(len(filenames)):

        # define file path
        file_path = source_path + filenames[i]

        # read as Document
        try:
            doc = Document.from_file(file_path, readers=[NlmXmlReader()])
        except:
            print('Failed to parse with NlmXmlReader. The XML file may not be valid.')
            continue

        for e in range(len(doc.elements)):
            if doc.elements[e].__class__.__name__ == 'Title':
                title = " ".join(str(doc.elements[e]).split())

        # determine first element to include in full text string
        start = next((i for i, element in enumerate(doc.elements) if str(element) == 'Abstract'), None) + 1
        #print(f'start: {start}')

        # determine last element to include in full text string
        end = next((i for i, element in enumerate(doc.elements) if element.__class__.__name__ == 'Citation'), None) - 1
        #print(f'end: {end}')

        # convert all relevant doc elements to a single full text string
        fulltext = _get_fulltext_from_elements(doc.elements, start, end)
        #print('got full text')

        # tokenize the full text into sentences and write to a text file
        none_parsed_count = _string_to_sentences(fulltext, filenames[i], target_path, title, none_parsed_count)
        
        print(f'Completed SpringerNature XML file {i}')


def _get_fulltext_from_elements(elements, start, end, figures=False, tables=False): 

    ### convert a doc.elements object from CDE2 into a full text string

    ### inputs:
    # elements: doc.elements object [list]
    # start: the position of the first element to include in the fulltext string [int]
    # end: the position of the last element to include in the fulltext string [int]
    # figures: if figure elements should be included in the fulltext string [bool] (default: False)
    # tables: if table elements should be included in the fulltext string [bool] (default: False)

    ### outputs:
    # fulltext: a single continuous text string of all relevant elements from the provided doc [str]

    # initiate full text string
    fulltext = ''

    #print(f'start: {start}')
    #print(f'end: {end}')

    for i in range(end-start+1):

        # define next element to add 
        element = elements[i+start]

        # check if it is a figure or table
        if element.__class__.__name__ == 'Figure':
            if figures == False:
                #print('Skipping Figure element')
                continue

        if element.__class__.__name__ == 'Table':
            if tables == False:
                #print('Skipping Table element')
                continue

        # convert to string, removing line breaks and extraneous white space
        text = " ".join(str(element).split())

        # add to the full text string
        fulltext = fulltext + ' ' + text

    return fulltext


def _string_to_sentences(text, id, dir_path, title, none_parsed_count, write=True, clean=False, abstract=None):

    ### takes some string of text and splits it into sentences using CDE2

    ### inputs:
    # text: natural language text to tokenize into chemistry-aware sentences [str]
    # id: a unique identifier for this particular document, used for writing file names [str]
    # dir_path: the path to the directory where files will be written to (string)
    # title: the title of the source (it will be written at the top of the generated txt file) [str]
    # none_parsed_count: the number of times thus far we have attempted to convert a blank string to sentences [int]
    # write: True if we should write sentences line by line to a text file, False if not [Bool] (default: True)
    # clean: True if we need to remove extraneous reference/introductory text, False if not [Bool] (default: False)
    # abstract: the abstract of that paper, only necessary if clean=True [str] (default: None)

    ### outputs:
    # none_parsed_count: the number of times thus far we have attempted to convert a blank string to sentences [int]

    # if we want to remove extraneous introductory text and have access to the abstract... (i.e., working with Elsevier full texts from API)
    if clean == True:

        #print(f'text: {text}')
        #print(f'type(text): {type(text)}')
        #print(f'len(text): {len(text)}')

        #remove text before the abstract
        abstract_idx = text.find(abstract[0:100]) # being safe and just using first 100 characters 
        text = text[abstract_idx:]

        # remove references and below
        ref_idx = text.rfind('Reference') ## TODO: change to account for more ways of delineating a reference section
        if ref_idx == -1:
            print('Could not find the references section; leaving unchanged.')
        else: 
            text = text[:ref_idx] 

        #print(f'text: {text}')
        #print(f'type(text): {type(text)}')
        #print(f'len(text): {len(text)}')

    # write to a text file 
    h = open(f'{dir_path}/{id}-continuous.txt', mode='a', encoding='utf-8') 
    h.write(text)
    h.close()

    # for TXT
    f = open(f'{dir_path}/{id}-continuous.txt', mode='rb')
    try:
        doc = Document.from_file(f) 
    except TypeError:
        none_parsed_count += 1
        print(f'{id} had no text parsed from it; skipping. This has happened {none_parsed_count} times.')
        return none_parsed_count

    # check that it's only one big paragraph
    if len(doc.elements) == 1:
        para = doc.elements[0]

        if write == True: ### write sentences line-by-line to a text file
            
            # append title
            g = open(f'{dir_path}/sentences-{id}.txt', mode='a', encoding='utf-8')
            g.write(f'{title}\n')
            g.close()

            skipnext = False
            for j in range(len(para.sentences)):

                # check if we already wrote this sentence with the prior one
                if skipnext:
                    skipnext = False
                    continue
            
                g = open(f'{dir_path}/sentences-{id}.txt', mode='a', encoding='utf-8')
                
                # if the sentence is less than 10 characters (arbitrary), combine it with the next sentence
                if len(str(para.sentences[j])) < 10:
                    try:
                        g.write(str(para.sentences[j])+' '+f'{str(para.sentences[j+1])}\n')
                    except IndexError: # there might not be a next sentence to combine it with
                        g.write(f'{str(para.sentences[j])}\n')
                    g.close()
                    skipnext = True
                else:
                    g.write(f'{str(para.sentences[j])}\n')
                    g.close()
                    skipnext = False
        
        #print(type(para.sentences))
        return none_parsed_count
        
    else: 
        print('Full text did not process as a single doc element. Skipping this paper.')
        return

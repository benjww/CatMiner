from fireworks.client import Fireworks
from openai import AzureOpenAI
import itertools
import boto3
import time
import os


def define_client(client_type):

    ### define the client variable based on the model type

    ### inputs:
    # client_type: the service we are using to host the model (supported options: Azure, Bedrock, Fireworks) [str]

    ### outputs:
    # client: the LLM client 

    if client_type == 'Azure':
        
        client = AzureOpenAI(
            api_key=os.environ['API_KEY'],  
            api_version=os.environ['API_VERSION'],
            azure_endpoint = os.environ['AZURE_ENDPOINT'],
            organization = os.environ['ORGANIZATION_ID']
            )
        
    elif client_type == 'Bedrock':

        client = boto3.client(service_name='bedrock-runtime', 
            region_name=os.environ['AWS_REGION']
            )
        
    elif client_type == 'Fireworks':

        client = Fireworks(api_key=os.environ['FIREWORKS_API_KEY'])

    else:
        print('Provided client type is not supported. Supported client types are Azure, Bedrock, and Fireworks.')

    return client


def read_sentences(file_path):

    ### parse title and sentences from a preprocessed article text file

    ### input
    # file_path: the path to a preprocessed paper text file [str]

    ### outputs
    # sentences: the sentences contained in the preprocessed paper [list]
    # title: the title of that preprocessed paper [str]

    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]

    sentences = lines[1:None]
    title = lines[0]

    return sentences, title


def _append_context(context, model_type, role, message):

    ### append a user prompt or LLM response to the context dictionary

    ### inputs:
    # context: all previous messages in the conversation [list of dict]
    # model_type: type of LLM we are expecting (supported options are 'OpenAI' and 'Meta') [str]
    # role: who delivered the message (supported options are "assistant" or "user") [str]
    # message: the message to append [str]

    ### outputs:
    # context: the conversation with the newest message appended [list of dict]

    if model_type == 'OpenAI':
        context.append({"role": role, "content": message})

    if model_type == 'Meta':
        context.append({"role": role, "content": [{"text": message}]})

    return context


def _get_ans(model_type, client, context, sysprompt=None, sleep_time=0.0):

    ### get the LLM response and token counts

    ### inputs: 
    # model_type: type of LLM we are expecting (supported options are 'OpenAI' and 'Meta') [str]
    # client: the LLM client (Azure, Bedrock, or Fireworks)
    # context: the query along with all previous messages in the conversation [list of dict]
    # sysprompt: (if using Meta) the system prompt [str] (default None)
    # sleep_time: delay in seconds imposed after a model call. Can be used to obey API rate limits [float] (default 0.0)

    ### outputs:
    # ans: the LLM response [str]
    # new_in_tkns: the amount of input tokens passed by this prompt [int]
    # new_out_tkns: the amount of output tokens produced by this prompt [int]

    if model_type == 'OpenAI':
    
        response = client.chat.completions.create(
            model=os.environ['model'],
            messages=context,
            temperature=0,
            max_tokens=100, 
            frequency_penalty=0,
            presence_penalty=0
        )

        ans = response.choices[0].message.content
        new_in_tkns = response.usage.prompt_tokens
        new_out_tkns = response.usage.completion_tokens

    if model_type == 'Meta':

        # save response 
        response = client.converse(
            modelId=os.environ['MODEL_ID'],
            messages=context,
            system=[{"text": sysprompt}],
            inferenceConfig={"temperature": 0.0, "topP": 0.0, "maxTokens": 100},
            performanceConfig={"latency": "optimized"}
        )

        ans = response['output']['message']['content'][0]['text']
        new_in_tkns = response['usage']['inputTokens']
        new_out_tkns = response['usage']['outputTokens']

    ans = ans.strip()

    time.sleep(sleep_time)

    return ans, new_in_tkns, new_out_tkns


def prompt(model_type, client, context, chat, sysprompt, user_message, in_tkn, out_tkn, log, append):
    
    ### prompt the LLM and update the conversation log and the token count
   
    ### inputs:
    # model_type: type of LLM we are expecting (supported options are 'OpenAI' and 'Meta') [str]
    # client: the LLM client (Azure, Bedrock, or Fireworks)
    # context: the query along with all previous messages in the conversation [list of dict]
    # chat: whether chat-like memory is enabled [Bool]
    # sysprompt: the system prompt [str]
    # user_message: the prompt given by the user [str]
    # in_tkn: the number of input tokens passed thus far [int]
    # out_tkn: the number of output tokens produced thus far [int]
    # log: the current log of all relevant outputs [list of dicts]
    # append: whether we should append the model answer to our conversation [Bool]

    ### outputs:
    # ans: the LLM response [str]
    # in_tkn: the number of input tokens passed thus far [int]
    # out_tkn: the number of output tokens produced thus far [int]

    # update context as needed
    if chat == True:
        context = _append_context(context, model_type, "user", user_message)
    if chat == False:
        context = [] # tabula rasa
        if model_type == 'OpenAI': # append system prompt to context if we're using OpenAI
            context.append({"role": "system", "content": sysprompt})
        context = _append_context(context, model_type, "user", user_message)

    # query model and count tokens
    ans, new_in_tkns, new_out_tkns = _get_ans(model_type, client, context, sysprompt)

    # update total token counts
    in_tkn = in_tkn + new_in_tkns
    out_tkn = out_tkn + new_out_tkns

    # sometimes, the API outputs a NoneType object instead of a response string as expected
    if isinstance(ans,str) == False:
        raise TypeError(f"Expected a string, but got {type(ans).__name__}")
    
    if append == True:
        context = _append_context(context, model_type, "assistant", ans)

    return ans, context, in_tkn, out_tkn, log


def write_log(context, log, message=None, verbose=False):

    ### update the CatMiner log
    
    ### inputs:
    # log: the current log of all relevant outputs [list of dict]
    # context: the content to be appended to it [list of dict]
    # message: any additional notes to include following the appended chat log [str] (default None)
    # verbose: whether the context and message should be printed [Bool] (default False)

    ### output:
    # log: the log as input + the appended context and message [list of dict]

    for c in range(len(context)):
        log.append(context[c])
    log.append(message)

    #print(f'log so far: {log}')

    if verbose==True:
        print(context, message)

    return log


def getexcerpt(title, sentences, s, params):

    ### affix additional context to a target sentence to construct an excerpt

    ### inputs:
    # title: the title of the paper from which the sentence comes from [str]
    # sentences: a CDE2-style collection of all sentences in the document [list]
    # s: the index of the target sentence in the document [int]
    # params: an object containing the context bounds and specifying if the title should be included [dict]

    ### outputs:
    # excerpt: a passage of text containing the target sentence and all specified context [str]

    # read input parameters
    P = params['Bounds'][0]
    F = params['Bounds'][1]
    T = params['Title']

    # initiate excerpt
    if T == True:
        excerpt = title + '. '
    elif T == False:
        excerpt = ''

    # run for one iteration per sentence in the desired context
    for i in range(P+F+1):
        idx = s-P+i

        # skip the sentence if the index is out of bounds
        if ((idx < 0) or (idx > len(sentences))):
            continue
        else:
            excerpt = excerpt + str(sentences[idx]) + ' '

    return excerpt


def obtain_acronym_defs(sentences, phrase, delimiters=['/', ' ', '-', '–', '@']):

    ### retrieve the sentences that are likely to contain acronym definitions

    ### inputs:
    # sentences: a list of sentences in the source text to choose from [list]
    # phrase: a string that has been classified as either being or containing an acronym [str]
    # delimiters: characters used to tokenize the provided phrase and check for subphrase acronyms [list]
    # min_phrase_length: the minimum length required to retrieve a source sentence for a subphrase [int]

    ### outputs:
    # defs: a collection of sentences that may contain the definition of the provided acronym(s) [str]
    
    # initialize subphrase dictionary and delimiter permutation count
    perm_dict = {}
    perm_count = 0

    # obtain all possible orderings of our delimiters
    delimiter_orders = itertools.permutations(delimiters,len(delimiters))

    for perm in delimiter_orders:
        
        delimiter_permutation = list(perm)

        # reset initial subphrase level
        subphrases = [[phrase]]

        for i in range(len(delimiter_permutation)): 

            subphrases.append([]) # define next level of subphrases

            for subphrase in subphrases[i]: # for each phrase in the current level...
                for j in range(len(subphrase.split(delimiter_permutation[i]))): 
                    subphrases[i+1].append(subphrase.split(delimiter_permutation[i])[j]) # add its subphrases to the next level

        # append these subphrases to a dictionary and then try the next permutation of delimiters
        perm_dict[perm_count] = subphrases
        perm_count += 1

    # combine all subphrases into a single list and remove duplicates
    all_subphrases = []
    for key in range(perm_count):
        for level in range(len(delimiters)+1):
            for i in range(len(perm_dict[key][level])):
                all_subphrases.append(perm_dict[key][level][i])

    # remove duplicates
    unique_subphrases = list(set(all_subphrases))
    
    # get the first sentence in the source text that includes each (sub)phrase
    # initialize list of definition sentences
    defs_list = []

    for subphrase in unique_subphrases:

        if len(subphrase) > 2:
            try:
                index = [idx for idx, s in enumerate(sentences) if subphrase in s][0]
                defs_list.append(sentences[index])
            except:
                continue       
        else:
            continue

    # remove duplicate definition sentences
    defs_list = list(set(defs_list))

    # initialize single string of definition sentences
    defs = ''
    # collate into a single string with new lines in between each sentence
    for sentence in defs_list:
        defs += sentence
        defs += '\n\n'

    return defs


def filter_sentences(sentences, s, required_phrases=None):
    
    ### append all the candidate sentences for far-field NERRE in a position-aware manner

    ### inputs
    # sentences: ordered list of all sentences contained in the source document [list]
    # s: the index of the current target sentence in the source document [int]
    # required_phrases: strings that must be present in a retrieved excerpt to consider scoring it [list] (default None)

    ### outputs
    # context: the new context to be supplied to the LLM for inter-paragraph search [str]

    filtered_sentences = []

    for sentence in sentences: 
        if any(x in sentence for x in required_phrases):
            filtered_sentences.append(sentence)
        else:
            filtered_sentences.append('...')

    # indicate where our current target sentence is
    filtered_sentences[s] = sentences[s] + ' <-- We are here'

    filtered_sentences_cleaned = [filtered_sentences[0]]

    for sentence in filtered_sentences[1:]:
        if sentence != filtered_sentences_cleaned[-1]:
            filtered_sentences_cleaned.append(sentence)
        
    context = ''

    for sentence in filtered_sentences_cleaned:
        context += sentence + '\n'

    return context

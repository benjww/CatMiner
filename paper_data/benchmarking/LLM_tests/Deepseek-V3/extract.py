from functions import getexcerpt, write_log, prompt, read_sentences, filter_sentences
from prompts import (
    prompt1,
    prompt2,
    prompt3,
    prompt4,
    prompt4_ips,
    promptf1,
    promptf2,
    promptf3,
    promptf3_nochat,
    promptf4,
    promptf4_nochat
)
import pandas as pd


def catminer_test_mode(file_path, client, target_dict, MODEL_TYPE, sp_path=None, log_path=None, log_bool=True, SYSPROMPT=True, FOLLOWUP=True, IPS=True, CHAT=True):
    
    ### this is a test mode of CatMiner that applies and records the results of all follow-up prompts without throwing out any records
    # this enables us to see the performance of any arbitrary combination of follow-up prompts

    # TODO: read all the hyperparameters in from an input file
    
    ### inputs
    # file_path: the path to a text file that obeys CatMiner input format (i.e., title in the first line, each following line is a new sentence) [str]
    # client: LLM client defined using our environmental variables
    # target_dict: a dictionary that defines all target variables and the context windows associated with each one [dict]
    # MODEL_TYPE: type of LLM we are expecting (supported values are 'OpenAI' and 'Meta') [str]
    # sp_path: the path to a text file that contains the user's desired system prompt [str] (default None)
    # log_path: the path to a csv file that the log should be written to [str] (default None)
    # write_log: True if we should write the LLM conversation to a CSV file, False if not [Bool] (default True)
    # SYSPROMPT: True if we should use the extraction system prompt, False if not [Bool] (default True)
    # FOLLOWUP: True if we should apply follow-up questions to interrogate material-property pairs, False if not [Bool] (default True)
    # IPS: True if we should use inter-paragraph search as a backup if nested properties are not found [Bool] (default True)

    ### outputs
    # extracted_records: all the pairs/triplets that were extracted from the provided sentences [dict]
    # in_tkn: the total # of input tokens passed [int]
    # out_tkn: the total # of output tokens produced [int]

    # read title and list of sentences from input text file
    sentences, title = read_sentences(file_path) # assumes first line is the title of the source

    # read system prompt
    if SYSPROMPT == True: 
        f = open(sp_path, mode="r", encoding="utf8")
        system_prompt = f.read()
    else: 
        if MODEL_TYPE == 'OpenAI':
            system_prompt = ' '
        if MODEL_TYPE == 'Meta':
            system_prompt = 'You are a helpful assistant.'

    # read target parent properties and optional required phrases to aid in extraction
    property = target_dict['Properties'][0]['Name']
    required_property_phrases = target_dict['Properties'][0]['Required Phrases'] 
    
    operating_conditions = [] 
    required_cond_phrases = []
    for i in range(len(target_dict['Operating Conditions'])):
        operating_conditions.append(target_dict['Operating Conditions'][i]['Name'])
        required_cond_phrases.append(target_dict['Operating Conditions'][i]['Required Phrases'])

    # initialize dictionary of outputs and log
    extracted_records = {'Source': [], 'Sentence': [], 'Property': [], 
                         'Property Value': [], 'Material': [], 'F1 Response': [], 
                         'F2 Response': [], 'F3 Response': []}
    
    for i, operating_condition in enumerate(operating_conditions):
        extracted_records[f'Operating Condition {i+1}'] = []
        extracted_records[f'Operating Condition {i+1} Value'] = []
        extracted_records[f'Operating Condition {i+1} Value (IPS)'] = []
        extracted_records[f'Operating Condition {i+1} F4 Response'] = []
        extracted_records[f'Operating Condition {i+1} F4-IPS Response'] = []

    log = []

    # initialize token and record counters
    in_tkn = 0
    out_tkn = 0
    rcounts = 0

    print(f'Beginning extraction from {len(sentences)} sentences...')

    for s in range(len(sentences)):

        sentence = sentences[s]

        # if we don't have any of the required phrases, continue to the next sentence
        if not any(phrase in sentence for phrase in required_property_phrases):
            continue

        # reset the context window with or without a system prompt
        context = []
        if MODEL_TYPE == 'OpenAI': # append system prompt to context if we're using OpenAI
            context.append({"role": "system", "content": system_prompt})

        # define excerpts for Prompts 3 and 4 
        excerpt_p3 = getexcerpt(title, sentences, s, target_dict['Properties'][0]['Context Params'])
        excerpts_p4 = []
        for i in range(len(target_dict['Operating Conditions'])):
            excerpt_p4 = getexcerpt(title, sentences, s, target_dict['Operating Conditions'][i]['Context Params'])
            excerpts_p4.append(excerpt_p4)

        ### PROMPT 1
        user_message = prompt1.format(property=property) + sentence
        try:
            p1_ans, context, in_tkn, out_tkn, log = prompt(MODEL_TYPE, client, context, CHAT, system_prompt, user_message, in_tkn, out_tkn, log, append=True)
        except Exception as e:
            log = write_log(context, log, message=f"Prompt 1 encountered some error: {e}. skipping to the next sentence.", verbose=True)
            continue

        # skip if negatively classified
        if 'no' in p1_ans.strip().lower(): 
            log = write_log(context,log,message="Moving to next sentence...")
            log.append(" ")
            continue

        ### PROMPT 2
        user_message = prompt2.format(property=property) + sentence
        try:
            p2_ans, context, in_tkn, out_tkn, log = prompt(MODEL_TYPE, client, context, CHAT, system_prompt, user_message, in_tkn, out_tkn, log, append=True)
        except Exception as e:
            log = write_log(context, log, message=f"Prompt 2 encountered some error: {e}. skipping to the next sentence.", verbose=True)
            continue

        # if no property values are extracted, skip the sentence
        if 'none' == p2_ans.strip().lower(): 
            log = write_log(context,log,message=f'Skipping sentence {s} because no {property} was extracted')
            log.append(" ")
            continue

        # save checkpoint
        context_p2 = context.copy() # context checkpoint

        # parse property values
        vals = p2_ans.split(';')

        # process the list of values to remove duplicates
        for val in vals:
            val = val.strip()
        vals = list(set(vals)) 

        # for each value...
        for val in vals:

            # screen out values that are named "None" or blank spaces
            # this often happens with, e.g., GPT-3.5 Turbo
            if val.strip().lower() == 'none' or '': 
                log = write_log(context,log,message=f'Skipped an invalid property value')
                log.append(" ")
                continue

            # load context from last checkpoint
            context = context_p2.copy() 

            ### PROMPT 3
            user_message = prompt3.format(property=property, property_value=val) + excerpt_p3
            try:
                p3_ans, context, in_tkn, out_tkn, log = prompt(MODEL_TYPE, client, context, CHAT, system_prompt, user_message, in_tkn, out_tkn, log, append=True)
            except Exception as e: 
                log = write_log(context, log, message=f"Prompt 3 encountered some error: {e}. skipping to the next property value.", verbose=True)
                continue

            # save checkpoint
            context_p3 = context.copy() 

            # parse material names
            mats = p3_ans.split(';') 

            # if there are no materials associated with this value, go to the next one
            if 'none' == p3_ans.strip().lower():
                log = write_log(context,log,message=f'Skipped {val} because there are no materials.')
                continue

            # for each material name...
            for mat in mats:

                # count the number of materials extracted
                rcounts += 1
                print('# record:', rcounts)

                # initiate lists of extracted operating condition values, contexts, and excerpts
                p4_responses = []
                p4_ips_responses = []
                p4_contexts = []
                p4_ips_contexts = []
                pf4_responses = []
                pf4_ips_responses = []
                ips_excerpts = []

                # for each target operating condition...
                for i, operating_condition in enumerate(operating_conditions):

                    # load context from last checkpoint
                    context = context_p3.copy() 
        
                    ### PROMPT 4 NO IPS 
                    user_message = prompt4.format(operating_condition=operating_condition, material=mat, property=property, property_value=val) + excerpts_p4[i]
                    try:
                        p4_ans, context, in_tkn, out_tkn, log = prompt(MODEL_TYPE, client, context, CHAT, system_prompt, user_message, in_tkn, out_tkn, log, append=True)
                        p4_responses.append(p4_ans)
                    except Exception as e:
                        log = write_log(context, log, message=f"Prompt 4 encountered some error: {e}. skipping to the next operating condition.", verbose=True)
                        p4_responses.append('Error')
                        continue

                    # save checkpoint
                    p4_context = context.copy()
                    p4_contexts.append(p4_context)

                    # if we extracted nothing...
                    if IPS == True:
                        if 'none' == p4_ans.strip().lower():

                            excerpt_ips = filter_sentences(sentences, s, required_cond_phrases[i])
                            ips_excerpts.append(excerpt_ips)

                            # load context from last checkpoint
                            context = context_p3.copy()

                            ### PROMPT 4 IPS -- using a different prompt than the original Prompt 4
                            user_message = prompt4_ips.format(operating_condition=operating_condition, material=mat, property=property, property_value=val) + ips_excerpts[i]
                            try:
                                p4_ips_ans, context, in_tkn, out_tkn, log = prompt(MODEL_TYPE, client, context, CHAT, system_prompt, user_message, in_tkn, out_tkn, log, append=True)
                                p4_ips_responses.append(p4_ips_ans)
                            except Exception as e:
                                log = write_log(context, log, message=f"Prompt 4-IP encountered some error: {e}. skipping to the next material.", verbose=True)
                                p4_ips_responses.append('Error')
                                continue

                            # save checkpoint
                            p4_ips_context = context.copy()
                            p4_ips_contexts.append(p4_ips_context)

                        else:
                            p4_ips_responses.append('None')
                            pf4_ips_responses.append('None')
                            ips_excerpts.append('None')

                # apply follow-up prompts TODO support multiple operating conditions here; I *think* I've got it working above but need to do some testing
                if FOLLOWUP == True:

                    # load context from P3 checkpoint
                    context = context_p3.copy() 

                    ### FOLLOW-UP PROMPT 1
                    user_message = promptf1.format(material=mat)
                    try:
                        pf1_ans, context, in_tkn, out_tkn, log = prompt(MODEL_TYPE, client, context, CHAT, system_prompt, user_message, in_tkn, out_tkn, log, append=False)                        
                    except Exception as e:
                        log = write_log(context, log, message=f"Follow-up 1 encountered some error: {e}. skipping to the next material.", verbose=True)
                        continue
                    if 'no' in pf1_ans.strip().lower(): # if we are...
                        log = write_log(context,log,message=f'Skipped {mat} because it is incomplete.')

                    # load context from P3 checkpoint
                    context = context_p3.copy() 

                    ### FOLLOW-UP PROMPT 2
                    user_message = promptf2.format(material=mat)
                    try:
                        pf2_ans, context, in_tkn, out_tkn, log = prompt(MODEL_TYPE, client, context, CHAT, system_prompt, user_message, in_tkn, out_tkn, log, append=False)                        
                    except Exception as e:
                        log = write_log(context, log, message=f'Follow-up 2 encountered some error: {e}. skipping to the next material.', verbose=True)
                        continue
                    if 'no' in pf2_ans.strip().lower(): # if we are...
                        log = write_log(context,log,message=f'Skipped {mat} because it is not specific.')

                    # load context from P3 checkpoint
                    context = context_p3.copy() 

                    # FOLLOW-UP PROMPT 3
                    if CHAT == True: 
                        user_message = promptf3.format(material=mat, property=property, property_value=val)
                    if CHAT == False:
                        user_message = promptf3_nochat.format(material=mat, property=property, property_value=val) + excerpt_p3
                    
                    try:
                        pf3_ans, context, in_tkn, out_tkn, log = prompt(MODEL_TYPE, client, context, CHAT, system_prompt, user_message, in_tkn, out_tkn, log, append=False)  
                    except Exception as e:
                        log = write_log(context, log, message=f'Follow-up 3 encountered some error: {e}. skipping to the next material.', verbose=True)
                        continue
                    if 'no' in pf3_ans.strip().lower(): # if the pair doesn't match... 
                        log = write_log(context,log,message=f'Threw out {mat}, {val}.')

                    # try the last follow-up if we had any operating conditions extracted with or without IPS...
                    # for each target operating condition...
                    for i, operating_condition in enumerate(operating_conditions):
                        if not 'none' == p4_responses[i].strip().lower(): 

                            # load context from P4 checkpoint
                            context = p4_contexts[i].copy()

                            ### FOLLOW-UP PROMPT 4 NO IPS
                            if CHAT == True:
                                user_message = promptf4.format(material=mat, property=property, property_value=val, operating_condition=operating_condition, operating_condition_value=p4_responses[i])
                            if CHAT == False:
                                user_message = promptf4_nochat.format(material=mat, property=property, property_value=val, operating_condition=operating_condition, operating_condition_value=p4_responses[i]) + excerpts_p4[i]
                            
                            try:
                                pf4_ans, context, in_tkn, out_tkn, log = prompt(MODEL_TYPE, client, context, CHAT, system_prompt, user_message, in_tkn, out_tkn, log, append=False) 
                                pf4_responses.append(pf4_ans)
                            except Exception as e:
                                log = write_log(context, log, message=f'Follow-up 4 encountered some error: {e}. skipping to the next material.', verbose=True)
                                pf4_responses.append('Error')
                                continue
                            if 'no' in pf4_ans.strip().lower(): # if the nested property doesn't match...
                                log = write_log(context,log,message=f'Threw out {p4_ans}.')

                        else:
                            pf4_responses.append('None')

                        # if IPS retrieved an operating condition...
                        if not 'none' == p4_ips_responses[i].strip().lower():

                            # load context from IPS checkpoint 
                            context = p4_ips_contexts[i].copy()
                            
                            ### FOLLOW-UP PROMPT 4 IPS
                            if CHAT == True:
                                user_message = promptf4.format(material=mat, property=property, property_value=val, operating_condition=operating_condition, operating_condition_value=p4_ips_responses[i])
                            if CHAT == False:
                                user_message=promptf4_nochat.format(material=mat, property=property, property_value=val, operating_condition=operating_condition, operating_condition_value=p4_ips_responses[i]) + ips_excerpts[i]

                            try:
                                pf4_ips_ans, context, in_tkn, out_tkn, log = prompt(MODEL_TYPE, client, context, CHAT, system_prompt, user_message, in_tkn, out_tkn, log, append=False)  
                                pf4_ips_responses.append(pf4_ips_ans)
                            except Exception as e:
                                log = write_log(context, log, message=f'Follow-up 4 encountered some error: {e}. skipping to the next material.', verbose=True)
                                pf4_ips_responses.append('Error')
                                continue

                        else:
                            pf4_ips_responses.append('None')
     
                # update the record list and counter
                extracted_records['Source'].append(file_path)
                extracted_records['Sentence'].append(sentence)
                extracted_records['Property'].append(property)
                extracted_records['Property Value'].append(val.strip())
                extracted_records['Material'].append(mat.strip())
                extracted_records['F1 Response'].append(pf1_ans.strip())
                extracted_records['F2 Response'].append(pf2_ans.strip())
                extracted_records['F3 Response'].append(pf3_ans.strip())

                for i, operating_condition in enumerate(operating_conditions):
                    extracted_records[f'Operating Condition {i+1}'].append(target_dict['Operating Conditions'][i]['Name'])
                    extracted_records[f'Operating Condition {i+1} Value'].append(p4_responses[i])
                    extracted_records[f'Operating Condition {i+1} Value (IPS)'].append(p4_ips_responses[i])
                    extracted_records[f'Operating Condition {i+1} F4 Response'].append(pf4_responses[i])
                    extracted_records[f'Operating Condition {i+1} F4-IPS Response'].append(pf4_ips_responses[i])

                # update log 
                log = write_log(context,log,message=f'Extracted {mat}, {val}.')        

        log.append(f'Extracted sentence {s}')
        log.append(" ")

    print(f'Completed extraction of {rcounts} records')
    
    # write chat to CSV log file
    if log_bool == True:
        logfile_dict = {'Chats': log}
        logfile_df = pd.DataFrame(data=logfile_dict)
        logfile_df.to_csv(log_path, mode='a', index=False)

    return extracted_records, in_tkn, out_tkn

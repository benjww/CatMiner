import os
import re

def _get_ans(client, context, sysprompt=None):

    ### specific function to get the LLM response and token counts

    ### inputs: 
    # client: the LLM client (currently only Bedrock supported) 
    # context: the query along with all previous messages in the conversation [list of dict]
    # sysprompt: (if using Llama) the system prompt [str]

    ### outputs:
    # ans: the LLM response [str]
    # new_in_tkns: the amount of input tokens passed by this prompt [int]
    # new_out_tkns: the amount of output tokens produced by this prompt [int]

    # save response 
    response = client.converse(
        modelId=os.environ['MODEL_ID'],
        messages=context,
        inferenceConfig={"temperature": 0.0, "topP": 0.0, "maxTokens": 100},
        performanceConfig={"latency": "optimized"}
    )

    ans = response['output']['message']['content'][0]['text']
    new_in_tkns = response['usage']['inputTokens']
    new_out_tkns = response['usage']['outputTokens']

    ans = ans.strip()

    return ans, new_in_tkns, new_out_tkns


def normalize_names(data, client, header, total_tokens=0):
    """
    Normalize the names of the materials in the data
    """
    # Get the list of materials
    materials = data[header].tolist()
    
    # Initialize counter
    counter = 0

    # Define prompt
    prompt = """Given the following inorganic material, please list the chemical elements it contains in a semicolon-separated list. 
            For example, given 'Manganese Sodium Tungstenate on Silicon Dioxide', you should respond with 'Mn; Na; W; Si; O': \n ""","""\n
            Please ONLY include atomic symbols in your response (e.g., 'Si', not 'Silicon').
            If you cannot identify the material, please just respond with 'None'.
            Please respond ONLY with the list of elements or None, as we are parsing the output. 
    """

    print(f'Normalizing the names of {len(materials)} materials...')

    # Normalize material names
    for material in materials:

        # Initialize context
        context = []
        user_message = prompt[0] + material + prompt[1]
        context.append({"role": "user", "content": [{"text": user_message}]})

        # Save response and update token count
        normalized_name, new_in_tkns, new_out_tkns = _get_ans(client, context, sysprompt=None)
        
        # If newlines are in the response, save only the line that contains a semicolon
        if '\n' in normalized_name: 
            lines = normalized_name.split('\n')
            for line in lines:
                if ';' in line:
                    normalized_name = line
                    break
        # if the name isn't None, transform into a list of atomic symbols 
        if normalized_name != 'None':
            element_list = str(normalized_name).split(';') # Transform into a list of atomic symbols 
            element_list = [x.strip() for x in element_list] # Remove blank space from each list element
            element_set = set(element_list) # Transform into a set
            normalized_name = str(element_set) # Transform back into a string to write to the data

        # Update total token count
        total_tokens += new_in_tkns + new_out_tkns

        # Update the material name in the data
        data.loc[counter, header] = normalized_name

        # Print progress
        counter += 1
        print(f'{counter}/{len(materials)} materials normalized')

    return data


def normalize_percentages(data, header, min_value=None, max_value=None):
    """
    Normalize the property values of the materials in the data
    """
    # Get the list of property values
    property_values = data[header].tolist()

    # Initialize counter
    counter = 0

    # Normalize the property values
    for property_value in property_values:

        # if the property value isn't a string, print progress and skip
        if not isinstance(property_value, str):
            counter += 1
            print(f'{counter}/{len(property_values)} property values normalized')
            continue

        # 1. Remove any text after the LAST percent symbol
        property_value = re.sub(r'%(?!.*%).*$', '%', property_value)

        # 2. Remove any text to the right of ± or +/-
        property_value = re.sub(r'(?:±|\+/-|±).*$', '', property_value)

        # 3. Remove any text to the left of ≈
        property_value = re.sub(r'.*≈', '', property_value)

        # 4. Replace ∼ with - if it does not appear in the first position
        if property_value[0] != '∼':
            property_value = re.sub(r'∼', '-', property_value)

        # 5. Replace several other characters with "-"
        property_value = re.sub(r'to|‒|–|−|< T <', '-', property_value)

        # 6. Replace commas with periods
        property_value = re.sub(r',', '.', property_value)

        # 7. Remove blank spaces adjacent to hyphens 
        property_value = re.sub(r'\s*(-)\s*', r'\1', property_value)

        # 8. Remove all percentage symbols
        property_value = re.sub(r'%', '', property_value)

        # 9. Remove any instances of decimals and hyphens that are not preceded AND followed by a digit
        property_value = re.sub(r'(?<!\d)[.-](?=\d)|(?<=\d)[.-](?!\d)|(?<!\d)[.-](?!\d)', '', property_value)

        # 10. Remove all characters except for 0-9, decimals, and hyphens
        property_value = re.sub(r'[^0-9.-]', '', property_value)

        # 11. Strip whitespace
        property_value = property_value.strip()

        # If the property value is a range, take the average
        if '-' in property_value:
            delimiter = '-'
            range = True
        else:
            range = False

        if range:
            range_boundaries = property_value.split(delimiter)
            property_value = str(round((float(range_boundaries[0])+float(range_boundaries[1]))/2.0, 2))

        # If the property value is an empty string, set it to None
        if property_value == "":
            property_value = 'None'

        # If the property value is outside the bounds, set it to None
        if min_value is not None and float(property_value) < min_value:
            property_value = 'None'
        if max_value is not None and float(property_value) > max_value:
            property_value = 'None'

        # Update the property value in the data
        data.loc[counter, header] = property_value

        # Print progress
        counter += 1
        print(f'{counter}/{len(property_values)} property values normalized')

    return data
    

def normalize_temperatures(data, header, min_temp=None, max_temp=None):
    """
    Normalize the temperatures of the materials in the data
    """
    # Get the list of temperatures
    temperatures = data[header].tolist()

    # Initialize counter
    counter = 0

    # Normalize the temperatures
    for temperature in temperatures:

        # if the temperature isn't a string, print progress and skip
        if not isinstance(temperature, str):
            counter += 1
            print(f'{counter}/{len(temperatures)} temperatures normalized')
            continue

        # Check for units
        if re.search(r'[CС℃]', temperature):
            unit = 'C'
        elif re.search(r'[F℉]', temperature):
            unit = 'F'
        elif re.search(r'[K]', temperature):
            unit = 'K'
        else:
            unit = None

        # If there is no unit, set equal to None, print progress, and skip
        if unit is None:
            data.loc[counter, header] = 'None'
            counter += 1
            print(f'{counter}/{len(temperatures)} temperatures normalized')
            continue

        # 1. Remove any text to the right of ± or +/-
        temperature = re.sub(r'(?:±|\+/-|±).*$', '', temperature)

        # 2. Replace commas with semicolons
        temperature = re.sub(r',', ';', temperature)

        # 3. Remove any text to the right of ;
        temperature = re.sub(r';.*$', '', temperature)

        # 4. Remove any text to the left of ≈
        temperature = re.sub(r'.*≈', '', temperature)

        # 5. Replace ∼ with - if it does not appear in the first position
        if temperature[0] != '∼':
            temperature = re.sub(r'∼', '-', temperature)

        # 6. Replace several other characters with "-"
        temperature = re.sub(r'to|‒|–|−|< T <', '-', temperature)

        # 7. Remove blank spaces adjacent to hyphens 
        temperature = re.sub(r'\s*([-])\s*', r'\1', temperature)

        # 8. Remove any instances of decimals and hyphens that are not preceded AND followed by a digit
        temperature = re.sub(r'(?<!\d)[.-](?=\d)|(?<=\d)[.-](?!\d)|(?<!\d)[.-](?!\d)', '', temperature)

        # 9. Remove all characters except for 0-9, decimals, and hyphens
        temperature = re.sub(r'[^0-9.-]', '', temperature)

        # 10. Strip whitespace
        temperature = temperature.strip()

        # If the temperature is a range, take the average
        if '-' in temperature:
            delimiter = '-'
            range = True
        else:
            range = False

        if range:
            range_boundaries = temperature.split(delimiter)
            temperature = str(round((float(range_boundaries[0])+float(range_boundaries[1]))/2.0, 2))

        # If the temperature is an empty string, set it to None
        if temperature == "":
            temperature = 'None'
        
        # If the temperature is outside the bounds, set it to None
        if min_temp is not None and float(temperature) < min_temp:
            temperature = 'None'
        if max_temp is not None and float(temperature) > max_temp:
            temperature = 'None'

        # Normalize to Celsius
        if unit == 'F':
            temperature = str(round((float(temperature)-32.0)*(5.0/9.0), 0))
        elif unit == 'K':
            temperature = str(round(float(temperature)-273.15, 0))

        # Update the temperature in the data
        data.loc[counter, header] = temperature

        # Print progress
        counter += 1
        print(f'{counter}/{len(temperatures)} temperatures normalized')

    return data

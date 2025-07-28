prompt1 = ('Answer "Yes" or "No" only. Does the following text contain a value of {property}?\n\n')

prompt2 = ('Use only data present in the sentence. If data is not present in the sentence, type '
           '"None". Please list each {property} value reported in the following sentence in a '
           'single semicolon-separated line with no additional text. Modifiers such as >, <, ≈, '
           '—, and ~ are allowed.\n\n')

prompt3 = ('Please list the catalyst(s) that give a {property} of {property_value}. If there are '
           'none, please say “None”. If multiple, please return them as a semicolon-separated '
           'list. If dopants, supports, or promoters are mentioned, include them in the '
           'material name. If the catalyst is a hybrid material, include all components in a '
           'single name. Do not use a full sentence and base your answer only on the following '
           'passage:\n\n')

prompt4 = ('What is the {operating_condition} when {material} gives a {property} of '
           '{property_value}? If none is given, please say "None". Modifiers such as >, <, ≈, —, '
           'and ~ are allowed. Do not use a full sentence and base your answer only on the '
           'following passage:\n\n')

prompt4_ips = ('What is the {operating_condition} when {material} gives a {property} of '
               '{property_value}? If none is given, please say "None". Modifiers such as >, <, '
               '≈, —, and ~ are allowed. Do not use a full sentence. Base your answer on this '
               'filtered version of the source article, provided below. Irrelevant paragraphs '
               'have been replaced with ellipses and the relative position of our current target '
               'sentence is denoted "<-- We are here":\n\n')

promptf1 = ('Is "{material}" a complete catalyst name? If dopants, supports, or promoters are '
            'mentioned, they should be included. Acronyms and variable stoichiometric '
            'coefficients are acceptable. Answer "Yes" or "No" only.')

promptf2 = ('Does the name "{material}" refer to a *specific* material? If it refers to a broad '
            'type, class, or family of materials, it does not count. Acronyms and variable '
            'stoichiometric coeffcients are acceptable. Answer "Yes" or "No" only.')

promptf3 = ('You said that {material} is a catalyst that gives a {property} value of '
            '{property_value}. Is this true? Answer "Yes" or "No" only. It is possible the '
            'information you extracted is wrong. Base your answer on the same passage as before.')

promptf3_nochat = ('You said that {material} is a catalyst that gives a {property} value of '
                   '{property_value}. Is this true? Answer "Yes" or "No" only. It is possible '
                   'the information you extracted is wrong. Base your answer only on the '
                   'following passage:\n\n')

promptf4 = ('You said that {material} is a catalyst that gives a {property} of {property_value} when '
            'the {operating_condition} is {operating_condition_value}. Is this true? Answer "Yes" '
            'or "No" only. It is possible the information you extracted is wrong. Base your answer '
            'on the same passage as before.')

promptf4_nochat = ('You said that {material} is a catalyst that gives a {property} of '
                   '{property_value} when the {operating_condition} is {operating_condition_value}. '
                   'Is this true? Answer "Yes" or "No" only. It is possible the information you '
                   'extracted is wrong. Base your answer only on the following passage:\n\n')

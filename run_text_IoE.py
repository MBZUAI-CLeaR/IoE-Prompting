'''
    This is the code for evaluating self-correction of LLM.
    Please pay attention to some key parameters, such as:
        model: 'gpt-3.5-turbo' or 'gpt-4'  gpt-3.5-turbo-1106
        request_timeout: when time exceeds, the code will continue to avoid getting stuck.
        temperature: [0,2], showing the variation of the answer.
'''

import openai
import json 
import time 
import re 
import numpy as np 
import os 
from tqdm import tqdm

openai.api_key = "XXX"


def read_data(file):
    with open(file) as f:
        data = [json.loads(line) for line in f]
    return data

def save_result(messages, path):
    f = open(path, 'a+')
    json.dump(messages, f)
    f.write('\n')
    f.close()

def normalize_answer(ans):
    ans = ans.lower()
    ans = ans.replace(',', '')
    ans = ans.replace('.', '')
    ans = ans.replace('?', '')
    ans = ans.replace('!', '')
    ans = ans.replace('\'', '')
    ans = ans.replace('\"', '')
    ans = ans.replace(';', '')
    ans = ans.replace(':', '')
    ans = ans.replace('-', '')
    ans = ans.replace('_', '')
    ans = ans.replace('(', '')
    ans = ans.replace(')', '')
    ans = ans.replace('[', '')
    ans = ans.replace(']', '')
    ans = ans.replace('{', '')
    ans = ans.replace('}', '')
    ans = ans.replace('/', '')
    ans = ans.replace('\\', '')
    ans = ans.replace('|', '')
    ans = ans.replace('<', '')
    ans = ans.replace('>', '')
    ans = ans.replace('=', '')
    ans = ans.replace('+', '')
    ans = ans.replace('*', '')
    ans = ans.replace('&', '')
    ans = ans.replace('^', '')
    ans = ans.replace('%', '')
    ans = ans.replace('$', '')
    # ans = ans.replace('#', '')
    ans = ans.replace('@', '')
    ans = ans.replace('~', '')
    ans = ans.replace('`', '')
    ans = ans.replace(' ', '')
    return ans

def get_answer_from_text(sentence):
    # sentence = sentence.replace(',', '')     # To remove the punctuation in number, e.g., $2,000
    pattern = re.compile(r'##(.*?)##')
    ans = re.findall(pattern, sentence)
    if len(ans):
        ans = ans[-1]
        ans = normalize_answer(ans)
        # try:
        #     ans = float(ans)
        # except:
        #     ans = float(10086100100)
    else:
        ans = ""
    return ans

def chat(messages, model_version):
    try:
        response = openai.ChatCompletion.create(
            model=model_version,
            messages=messages,
            request_timeout=50,
            temperature=0,
        )
        text = response.choices[0].message["content"]
    except:
        text = -1
    return text

def main(i, data, model):
    QAs = dict()
    QAs['index'] = i
    question = data['question']
    answer = data['answer']
    extractor = " Your final answer should be put between two ##, like ## yes ## (if your final answer is yes), at the end of your response."

    question = question + " Explain your reasoning step-by-step." + extractor
    QAs['Q1'] = {'role': 'user', 'content': question}
    messages=[{'role': 'user', 'content': question}]
    response_1 = chat(messages, model)
    if response_1==-1:
        return -1
    QAs['A1'] = {'role': 'assistant', 'content':response_1}
    messages.append({'role': 'assistant', 'content':response_1})
    
    question = "Review your previous answer. If you are confident about your answer, maintain your answer. Otherwise, update your answer." + extractor
    QAs['Q2'] = {'role': 'user', 'content': question}
    messages.append({'role': 'user', 'content': question})
    response_2 = chat(messages, model)
    if response_2==-1:
        return -1
    QAs['A2'] = {'role': 'assistant', 'content':response_2}
    messages.append({'role': 'assistant', 'content':response_2})

    QAs['answer'] = answer
    QAs['P1_ans'] = get_answer_from_text(response_1)
    QAs['P2_ans'] = get_answer_from_text(response_2)
    if QAs['P1_ans']!= QAs['P2_ans']:  
        question = "You give two different answers in previous responses. Check the problem and your answers again, and give the best answer." + extractor
        QAs['Q3'] = {'role': 'user', 'content': question}
        messages.append({'role': 'user', 'content': question})
        response_3 = chat(messages, model)
        if response_3==-1:
            return -1
        QAs['A3'] = {'role': 'assistant', 'content':response_3}
        messages.append({'role': 'assistant', 'content':response_3})
        QAs['P3_ans'] = get_answer_from_text(response_3)
        QAs['Final_ans'] = QAs['P3_ans']

    if 'Q3' in QAs.keys():
        QAs['Final_ans'] = QAs['P3_ans']
    else:
        QAs['Final_ans'] = QAs['P2_ans']
    return QAs 


if __name__=='__main__':
    """
    Flag: 
        1: Run the dataset to collect responses from LLM. 
        2: Evaluate the results.
        3: Run and Evaluate.
    
    Dataset:
        Sports, LLC.

    Model:
        'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', or 'gpt-4'.  
    """
    flag = 3

    dataset = 'Sports'
    model = "gpt-3.5-turbo-0613"
    input_dir = "dataset/"
    output_dir = f'output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path_input = f'{input_dir}/{dataset}.jsonl'
    path_output = f'{output_dir}/{dataset}_{model}_IoE.jsonl'

    skip_list, skip_list2 = [], []  # We use "skip_list2" to save these unsolved questions and solve them again in the next loop.
    if flag==1 or flag==3: 
        data = read_data(path_input)
        skip_list2 = range(len(data))   
        print(f"data size: {len(skip_list2)}, output: {path_output}, and key: {openai.api_key}")
        while len(skip_list2)!=0:
            skip_list = []
            for i in tqdm(skip_list2): 
                start = time.time()
                messages = main(i, data[i], model)
                if messages==-1:
                    print(f"I={i}, Skip this round. Next one!")
                    skip_list.append(i)
                    continue 
                save_result(messages, path_output)
                end = time.time()
            print(f"Skip list: {skip_list}.")
            skip_list2 = skip_list

    if flag==2 or flag==3:
        data_est = read_data(path_output)
        length = len(data_est)
        count_1 = 0 # The accuracy of standard prompt.
        count_2 = 0 # The accuracy of IoE prompt.
        count_3 = 0 # The accuracy of Refinement.
        for i in range(length): 
            if data_est[i]['answer']==data_est[i]['P1_ans']:
                count_1 += 1
            if data_est[i]['answer']==data_est[i]['P2_ans']:
                count_2 += 1
            if data_est[i]['answer']==data_est[i]['Final_ans']:
                count_3 += 1

        print(f"The accuracy of standard Prompt: {count_1/length*100}.")
        print(f"The accuracy of IoE prompt: {count_2/length*100}.") 
        print(f"The accuracy of Refinement: {count_3/length*100}.")

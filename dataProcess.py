import base64
import json
import re

from openai import OpenAI
from tqdm import tqdm
SystemPrompt_test = '''
Please tell me how many pictures you have seen?'''

SystemPrompt_step1 = '''You are a detailed image analyst. Please provide a thorough description of the image. If the image contains only text, present the text in Markdown format. 
If there are visual elements beyond text, describe the image comprehensively, paying special attention to intricate details. 
The goal is to enable someone who has never seen the image to visualize and recreate it based on your description.'''
SystemPrompt_step2 = '''You are a comprehensive issue analyst. The user will provide you with an issue that consists of multiple images and related text. Please connect the content of the text to provide a detailed description for each image, specifically relating it to the issue at hand. Additionally, analyze the role of each image within the context of the issue, explaining its significance and how it complements the overall narrative.

Please analyze each image and provide your analysis in the following structured JSON format:
{
  "image_analyses": [
    {
      "image_id": "<sequential number of the image>",
      "description": "<detailed description of the image in relation to the issue>",
      "analysis": "<analysis of the image's role in the issue>"
    }
  ]
}
'''
SystemPrompt_step3 = '''You are an issue organizer and analyzer. The user will provide you with an issue along with supplementary information that includes descriptions and analyses of images in the issue. Based on the issue and the supplementary information, please think through the details step by step and output the original issue in a structured JSON format. A suggested structure could include:

{
  "problemSummary": "<summary of the problem>",
  "stepsToReproduce": "<steps to reproduce the issue (if applicable)>",
  "expectedResults": "<expected results (if applicable)>",
  "actualResults": "<actual results (if applicable)>"
}

Please keep in mind the following:
1. The supplementary information may contain errors and is only for reference. You should prioritize the original issue.
2. Your structure does not need to match the suggested format exactly. Feel free to add any additional fields that you believe will help clarify the issue, ensuring that it remains clear and structured for better understanding.'''
SystemPrompt_step3_COT = '''You are an issue organizer and analyzer. User will provide you with an issue along with supplementary information that includes descriptions and analyses of images in the issue. Based on the issue and the supplementary information, please think through the details step by step and first output your rationale for the structured format, followed by the structured output itself.

The expected response format is as follows:
Rationale: <rationale>
Structured format: {
  "problemSummary": "<summary of the problem>",
  "stepsToReproduce": "<steps to reproduce the issue (if applicable)>",
  "expectedResults": "<expected results (if applicable)>",
  "actualResults": "<actual results (if applicable)>"
}

Please keep in mind the following:
1. The supplementary information may contain errors and is only for reference. You should prioritize the original issue.
2. The output structured format does not need to match the suggested format exactly. You are encouraged to add any additional fields or modify the structure as you see fit to clarify the issue, ensuring that it remains clear and structured for better understanding.

Please note that we not only need structured data, but more importantly, we need the rationale behind it to understand the reasoning process.'''


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def system_message(text):
    message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": text
            }
        ]
    }
    return message


def user_message_step1(image_path):
    cur_image = encode_image(image_path)
    message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{cur_image}"
                }
            }
        ]
    }
    return message


def user_message_step2(problem_list, image_list):
    contents = []
    for i in range(len(image_list)):
        if image_list[i] == 0:
            contents.append({
                "type": "text",
                "text": problem_list[i]
            })
        elif image_list[i] == 1:
            cur_image = encode_image(problem_list[i])
            contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{cur_image}"
                }
            })
        else:
            pass
    message = {
        "role": "user",
        "content": contents
    }
    return message


def user_message_step3(problem_list, image_list, description_list):
    contents = [{
        "type": "text",
        "text": "issue:\n"
    }]
    for i in range(len(image_list)):
        if image_list[i] == 0:
            contents.append({
                "type": "text",
                "text": problem_list[i]
            })
        elif image_list[i] == 1:
            cur_image = encode_image(problem_list[i])
            contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{cur_image}"
                }
            })
        else:
            pass
    supplementary_info = '''supplementary information:
    The following is a description and analysis of the images that appear in the issue. "raw description" refers to the direct description of the image, "description" refers to the description of the image in the context of the issue, and "analysis" refers to the analysis of the image.'''
    for description in description_list:
        supplementary_info += "\n"
        supplementary_info += description

    contents.append({
        "type": "text",
        "text": supplementary_info
    })

    message = {
        "role": "user",
        "content": contents
    }
    return message


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",  # 随便填写，只是为了通过接口参数校验
)

def filter_data(data_list,str_list):
    save_data = []
    for data in data_list:
        if data["instance_id"] in str_list:
            save_data.append(data)
    return save_data

def step1(data_file):
    with open(data_file, "r") as f:
        data_list = json.load(f)
    save_data_list = []
    for data in tqdm(data_list):
        raw_description_list = []
        instance_id = data["instance_id"]
        index = 0
        for problem in data["problem_statement"]:
            if problem.startswith('http'):
                message1 = system_message(SystemPrompt_step1)
                message2 = user_message_step1(f"images/{instance_id}/图片{index}.png")
                completion = client.chat.completions.create(
                    model="/gemini/platform/public/llm/huggingface/Qwen/Qwen2-VL-72B-Instruct",
                    messages=[message1, message2]
                )
                index += 1
                raw_description_list.append(completion.choices[0].message.content)
            else:
                continue
        save_data_list.append({
            "instance_id": instance_id,
            "raw_description_list": raw_description_list
        })
    with open("step1.json", 'w', encoding='utf-8') as outfile:
        json.dump(save_data_list, outfile, ensure_ascii=False, indent=4)


def step2(data_file):
    with open(data_file, "r") as f:
        data_list = json.load(f)
    save_data_list = []
    data_list = filter_data(data_list,["astropy__astropy-13838","matplotlib__matplotlib-22931", "matplotlib__matplotlib-24189","matplotlib__matplotlib-24768","mwaskom__seaborn-3276","sphinx-doc__sphinx-11502", "sphinx-doc__sphinx-8120", "sphinx-doc__sphinx-9698"])
    for data in tqdm(data_list):
        problem_list = []
        image_list = []
        instance_id = data["instance_id"]
        index = 0
        for problem in data["problem_statement"]:
            if problem.startswith('http'):
                problem_list.append(f"images/{instance_id}/图片{index}.png")
                image_list.append(1)
                index += 1
            else:
                problem_list.append(problem)
                image_list.append(0)

        message1 = system_message(SystemPrompt_step2)
        message2 = user_message_step2(problem_list, image_list)
        completion = client.chat.completions.create(
            model="/gemini/platform/public/llm/huggingface/Qwen/Qwen2-VL-72B-Instruct",
            messages=[message1, message2],
            temperature = 0.3,
            seed = 42
        )
        input_str = completion.choices[0].message.content
        print(f"success,input_str=" + input_str)
        try:
            # 使用正则表达式匹配 JSON 结构
            json_matches = re.findall(r'\{[^{}]*\}', input_str)
            # 将提取到的 JSON 字符串转换为 Python 字典，并存入列表
            description_list = [json.loads(json_str) for json_str in json_matches]
            save_data_list.append({
                "instance_id": instance_id,
                "description_list": description_list
            })
        except json.decoder.JSONDecodeError as e:
            # 如果解析失败，捕获JSONDecodeError异常并处理
            print(f"error,input_str=" + input_str)
            # 你可以选择在这里记录错误、跳过当前字符串或采取其他措施

    with open("step2_filter.json", 'w', encoding='utf-8') as outfile:
        json.dump(save_data_list, outfile, ensure_ascii=False, indent=4)


def step3(data_file, step1_file, step2_file):
    with open(data_file, "r") as f:
        data_list = json.load(f)

    with open(step1_file, "r") as f:
        step1_data_list = json.load(f)

    with open(step2_file, "r") as f:
        step2_data_list = json.load(f)

    description_list = []
    for i in range(len(data_list)):
        for j in range(len(step2_data_list[i]["description_list"])):
            description_list.append({
                "raw description": step1_data_list[i]["raw_description_list"][j],
                "description": step2_data_list[i]["description_list"][j]["description"],
                "analysis": step2_data_list[i]["description_list"][j]["analysis"]
            })

    save_data_list = []
    for data in tqdm(data_list):
        problem_list = []
        image_list = []
        instance_id = data["instance_id"]
        index = 0
        for problem in data["problem_statement"]:
            if problem.startswith('http'):
                problem_list.append(f"images/{instance_id}/图片{index}.png")
                image_list.append(1)
                index += 1
            else:
                problem_list.append(problem)
                image_list.append(0)

        message1 = system_message(SystemPrompt_step3)
        message2 = user_message_step3(problem_list, image_list, description_list)
        completion = client.chat.completions.create(
            model="/gemini/platform/public/llm/huggingface/Qwen/Qwen2-VL-72B-Instruct",
            messages=[message1, message2]
        )

        input_str = completion.choices[0].message.content
        # 使用正则表达式匹配 JSON 结构
        json_matches = re.findall(r'\{.*?\}', input_str)
        # 将提取到的 JSON 字符串转换为 Python 字典，并存入列表
        structure_problem = [json.loads(json_str) for json_str in json_matches][0]
        save_data_list.append({
            "instance_id": instance_id,
            "structure_problem": structure_problem
        })
    with open("step3.json", 'w', encoding='utf-8') as outfile:
        json.dump(save_data_list, outfile, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    step2('multi_data_onlyimage.json')
   #step2("test.json")

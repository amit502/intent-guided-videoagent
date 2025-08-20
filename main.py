import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI
import ollama
import google.generativeai as genai

from utils_clip import frame_retrieval_seg_ego, initial_frame_retrieval_seg_ego
from utils_general import get_from_cache, save_to_cache

from pydantic import BaseModel, StringConstraints, constr
from typing import Annotated, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("egoschema_subset.log")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# deepseek(local using ollama), chatgpt and gemini supported
llm_provider='deepseek'

if llm_provider=='chatgppt':
    client = OpenAI()
if llm_provider=='deepseek':
    client = ollama.Client(host='http://localhost:11434')
if llm_provider=='gemini':
    genai.configure(api_key='<API_KEY>')



def chat_with_schema_gemini(model: str, messages: list, format_model: BaseModel, num_frames:int,sample_idx):
    gen_model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
    full_prompt = (
        # f"While giving final answer, it should be a stringified number where 3 or greater is high"
        # f"The segment id is always a stringified number and the duration is a stringified version of 2 numbers conneted by a hyphen where each number represents a frame index of the video. The frame index should be less than or equal to 180 and segment id should be no more than the length of sample indices."
        f"While giving final answer, it should be a stringified number where 3 or greater value represents high confidence and lower values represent low confidence."
        f"The segment id is always a stringified number and the duration is a stringified version of 2 numbers conneted by a hyphen where each number represents a frame index of the video. The frame index should be less than or equal to {num_frames}."
        f"The duration should only use the adjacent values in {sample_idx} and should use all the adjacent values in {sample_idx}."
        f"You must respond with JSON in this schema:\n{json.dumps(format_model, indent=2)}\n\n" if format_model else ""
        f"{prompt}"
    )

    #gen_model = genai.GenerativeModel(model)
    response = gen_model.generate_content(full_prompt)
    return response.text.strip('`json\n').strip() #response.text



def parse_json(text):
    try:
        # First, try to directly parse the text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON
        json_pattern = r"\{.*?\}|\[.*?\]"  # Pattern for JSON objects and arrays

        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no JSON structure is found
        print("No valid JSON found in the text.")
        return None


def parse_text_find_number(text):
    item = parse_json(text)
    try:
        match = int(item["final_answer"])
        if match in range(-1, 5):
            return match
        else:
            return random.randint(0, 4)
    except Exception as e:
        logger.error(f"Answer Parsing Error: {e}")
        return -1


def parse_text_find_confidence(text):
    item = parse_json(text)
    try:
        match = int(item["confidence"])
        if match in range(1, 4):
            return match
        else:
            return random.randint(1, 3)
    except Exception as e:
        logger.error(f"Confidence Parsing Error: {e}")
        return 1


def get_llm_response(
    system_prompt, prompt, json_format=True, model="deepseek-r1:1.5b",     #"gpt-4-1106-preview",
    format=None, num_frames=180,sample_idx=[1,45,90,135,180]
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    # key = json.dumps([model, messages])
    # logger.info(messages)
    # cached_value = get_from_cache(key)
    # if cached_value is not None:
    #     logger.info("Cache Hit")
    #     logger.info(cached_value)
    #     return cached_value

    # print("Not hit cache", key)
    # input()

    for _ in range(3):
        try:
            if json_format and format:
                if llm_provider=='chatgpt':
                    completion = client.chat.completions.create(
                        model=model,
                        response_format={"type": "json_object"},
                        messages=messages,
                    )
                if llm_provider=="deepseek":
                    completion = client.chat(
                        model=model,
                        format=format,
                        messages=messages
                    )
                if llm_provider=="gemini":
                    response = chat_with_schema_gemini(
                        model="gemini-pro",
                        messages=messages,
                        format_model=format,
                        num_frames=num_frames,
                        sample_idx=sample_idx
                    )
            else:
                if llm_provider=='chatgpt':
                    completion = client.chat.completions.create(
                        model=model, messages=messages
                    )
                if llm_provider=="deepseek":
                    completion = client.chat(
                        model=model,
                        messages=messages
                    )
                if llm_provider=="gemini":
                    response = chat_with_schema_gemini(
                        model="gemini-pro",
                        messages=messages,
                        format_model=None,
                        num_frames=num_frames,
                        sample_idx=sample_idx
                    )

            if llm_provider=="chatgpt":
                response = completion.choices[0].message.content
            if llm_provider=="deepseek":
                response = completion.message.content
            logger.info(response)
            #save_to_cache(key, response)
            return response
        except Exception as e:
            logger.error(f"GPT Error: {e}")
            continue
    return "GPT Error"


def generate_final_answer(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}

    class AnswerFormat(BaseModel):
        final_answer: str
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think carefully and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question, and you must select one answer index from the candidates. The answer is a number indicating the answer index.
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True,format=AnswerFormat.model_json_schema())
    return response


def generate_description_step(question, caption, num_frames, segment_des,sample_idx):
    # Less powerful LLMs don't get the format right using the example only.
    # Hence, pydantic class provided instead of example
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "3", "duration": "xxx - xxx", "description": "frame of xxx"},
        ]
    }

    class FrameDescription(BaseModel):
        segment_id: str
        duration: str #Annotated[str, StringConstraints(pattern=r'^\d+\s*-\s*\d+$')]
        description: str

    class FormattedDescription(BaseModel):
        frame_descriptions: List[FrameDescription]

    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial frames is not suffient.
    Objective:
    Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
    To achieve this, we will:
    1. Divide the video into segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Select multiple frames from one segment if necessary to gather comprehensive insights. Use the same "segment_id" for the frames belonging to the same segment.
    The "segment_id" is always a stringified number and must be any number from the list {list(range(1,len(segment_des) + 1 ))}. The "duration" is a stringified version of 2 numbers conneted by a hyphen where each number represents a frame index of the video. The frame index should be less than or equal to {num_frames}.
    The total number of descriptions should strictly be less than {len(sample_idx)}.
    Return the descriptions and the segment id in JSON format, note "description" should be text that describes the visual information required to answer the question, "segment_id" must be a stringified number and should be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments like "1-45", where 1 and 45 are frame indices:
    ```
    {FormattedDescription.model_json_schema()}
    ```
    The above format is just an example for the format of the response. Use the initial frame descriptions, question and options for the context to generate the actual response.
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True, format=FormattedDescription.model_json_schema(),num_frames=num_frames, sample_idx=sample_idx)
    return response


def self_eval(previous_prompt, answer):
    confidence_format = {"confidence": "xxx"}

    class ConfidenceFormat(BaseModel):
        confidence: str

    prompt = f"""Please assess the confidence level in the decision-making process.
    The provided information is as as follows,
    {previous_prompt}
    The decision making process is as follows,
    {answer}
    Criteria for Evaluation:
    Insufficient Information (Confidence Level: 1): If information is too lacking for a reasonable conclusion.
    Partial Information (Confidence Level: 2): If information partially supports an informed guess.
    Sufficient Information (Confidence Level: 3): If information fully supports a well-informed decision.
    Assessment Focus:
    Evaluate based on the relevance, completeness, and clarity of the provided information in relation to the decision-making context.
    Confidence string is one of "1", "2" or "3".
    Please generate the confidence with JSON format {confidence_format}
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True, format=ConfidenceFormat.model_json_schema())
    return response


def ask_gpt_caption(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}

    class AnswerFormat(BaseModel):
        final_answer: str

    #this of five uniformly
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of a few sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question, and you must select one answer index from the candidates. The answer is a number indicating the answer index.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response


def ask_gpt_caption_step(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}

    class AnswerFormat(BaseModel):
        final_answer: str

    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question, and you must select one answer index from the candidates. The answer is a number indicating the answer index.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response


def read_caption(captions, sample_idx):
    video_caption = {}
    for idx in sample_idx:
        video_caption[f"frame {idx}"] = captions[idx - 1]
    return video_caption

def divide_range(value, parts):
    step = value // parts
    ranges = []
    for i in range(parts):
        start = i * step + 1
        end = (i + 1) * step
        ranges.append(f"{start}-{end}")
    return ranges


def run_one_question(video_id, ann, caps, logs):
    question = ann["question"]
    answers = [ann[f"option {i}"] for i in range(5)]
    formatted_question = (
        f"Here is the question: {question}\n"
        + "Here are the choices: "
        + " ".join([f"{i}. {ans}" for i, ans in enumerate(answers)])
    )
    num_frames = len(caps)

    ### Step 1 ###
    sample_idx = np.linspace(1, num_frames, num=4, dtype=int).tolist()
    ### Semantic intent-guided sampling ###
    initial_question_description = []
    for i, ans in enumerate(answers):
        initial_question_description.append({"segment_id": f'{i+1}', "description": ans})
    try:
        sample_idx = initial_frame_retrieval_seg_ego(
                    initial_question_description, video_id, sample_idx
                )
    except Exception as e:
        print(e)
    ### Semantic intent-guided sampling ###
    sampled_caps = read_caption(caps, sample_idx)
    previous_prompt, answer_str = ask_gpt_caption(
        formatted_question, sampled_caps, num_frames
    )
    answer = parse_text_find_number(answer_str)
    confidence_str = self_eval(previous_prompt, answer_str)
    confidence = parse_text_find_confidence(confidence_str)

    ### Step 2 ###
    if confidence < 3:
        try:
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                for i in range(len(sample_idx) - 1)
            }
            candiate_descriptions = generate_description_step(
                formatted_question,
                sampled_caps,
                num_frames,
                segment_des,
                sample_idx
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_idx = frame_retrieval_seg_ego(
                parsed_candiate_descriptions["frame_descriptions"], video_id, sample_idx
            )
            logger.info(f"Step 2: {frame_idx}")
            sample_idx += frame_idx
            sample_idx = sorted(list(set(sample_idx)))

            sampled_caps = read_caption(caps, sample_idx)
            previous_prompt, answer_str = ask_gpt_caption_step(
                formatted_question, sampled_caps, num_frames
            )
            answer = parse_text_find_number(answer_str)
            confidence_str = self_eval(previous_prompt, answer_str)
            confidence = parse_text_find_confidence(confidence_str)
        except Exception as e:
            logger.error(f"Step 2 Error: {e}")
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames
            )
            answer = parse_text_find_number(answer_str)

    ### Step 3 ###
    if confidence < 3:
        try:
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                for i in range(len(sample_idx) - 1)
            }
            candiate_descriptions = generate_description_step(
                formatted_question,
                sampled_caps,
                num_frames,
                segment_des,
                sample_idx
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_idx = frame_retrieval_seg_ego(
                parsed_candiate_descriptions["frame_descriptions"], video_id, sample_idx
            )
            logger.info(f"Step 3: {frame_idx}")
            sample_idx += frame_idx
            sample_idx = sorted(list(set(sample_idx)))
            sampled_caps = read_caption(caps, sample_idx)
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames
            )
            answer = parse_text_find_number(answer_str)
        except Exception as e:
            logger.error(f"Step 3 Error: {e}")
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames
            )
            answer = parse_text_find_number(answer_str)
    if answer == -1:
        logger.info("Answer Index Not Found!")
        answer = random.randint(0, 4)

    logger.info(f"Finished video: {video_id}/{answer}/{ann['truth']}")

    label = int(ann["truth"])
    corr = int(label == answer)
    count_frame = len(sample_idx)

    logs[video_id] = {
        "answer": answer,
        "label": label,
        "corr": corr,
        "count_frame": count_frame,
    }


def main():
    # if running full set, change subset to fullset
    input_ann_file = "subset_anno.json"
    all_cap_file = "lavila_subset.json"
    json_file_name = "egoschema_subset.json"

    anns = json.load(open(input_ann_file, "r"))
    all_caps = json.load(open(all_cap_file, "r"))
    logs = {}

    tasks = [
        (video_id, anns[video_id], all_caps[video_id], logs)
        for video_id in list(anns.keys())
    ]
    ### Running only 20 questions ###
    tasks = tasks[:20]
    ### Running only 20 questions ###
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(lambda p: run_one_question(*p), tasks)

    json.dump(logs, open(json_file_name, "w"))


if __name__ == "__main__":
    main()

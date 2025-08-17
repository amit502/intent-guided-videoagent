import pickle
from typing import List

import numpy as np

from transformers import CLIPProcessor, CLIPModel
import torch
# Load model directly
from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("BAAI/EVA-CLIP-8B", trust_remote_code=True)
model = AutoModel.from_pretrained("BAAI/EVA-CLIP-8B", trust_remote_code=True)


def get_embeddings(inputs: List[str]) -> np.ndarray:
    global model
    global processor


    model.eval()

    # Tokenize the text inputs
    texts = processor(text=inputs, return_tensors="pt", padding=True, truncation=True)

    # Get final text embeddings
    with torch.no_grad():
        text_outputs = model.text_model(**texts)
        final_embeddings = text_outputs.pooler_output

    print("Final text embeddings shape:", final_embeddings.shape)
    print("Final text embeddings:", final_embeddings)
    return final_embeddings.detach().cpu().numpy()

###
# Frame retrieval from original paper: VideoAgent. 
# Commenting out since we modified it slightly to use duration instead of segment ids
###
# def frame_retrieval_seg_ego(descriptions, video_id, sample_idx):
#     frame_embeddings = np.load(f"../ego_features_448/{video_id}.npy")
#     text_embedding = get_embeddings(
#         [description["description"] for description in descriptions]
#     )

#     frame_idx = []
#     try:
#         for idx, description in enumerate(descriptions):

#             seg = int(description["segment_id"]) - 1
#             seg_frame_embeddings = frame_embeddings[sample_idx[seg] : sample_idx[seg + 1]]
#             if seg_frame_embeddings.shape[0] < 2:
#                 frame_idx.append(sample_idx[seg] + 1)
#                 continue
#             seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
#             seg_frame_idx = sample_idx[seg] + seg_similarity.argmax() + 1
#             if seg_frame_idx <=180: frame_idx.append(seg_frame_idx)
#     except Exception as e:
#         print(e)

#     return frame_idx


# Our modified frame retrieval using duration instead of segment ids
def frame_retrieval_seg_ego(descriptions, video_id, sample_idx):
    frame_embeddings = np.load(f"../ego_features_448/{video_id}.npy")
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions]
    )
    frame_idx = []
    try:
        for idx, description in enumerate(descriptions):
            duration = description["duration"].split('-')
            seg = int(duration[0])
            end =  int(duration[1])
            seg_frame_embeddings = frame_embeddings[seg : end]
            if seg_frame_embeddings.shape[0] < 2:
                frame_idx.append(seg + 1)
                continue
            seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
            seg_frame_idx = seg + seg_similarity.argmax() + 1
            if seg_frame_idx <=180: frame_idx.append(seg_frame_idx)
    except Exception as e:
        print(e)

    return frame_idx

# Initial intent-guided frame sampling
def initial_frame_retrieval_seg_ego(descriptions, video_id,sample_idx):
    frame_embeddings = np.load(f"../ego_features_448/{video_id}.npy")
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions]
    )
    frame_idx = set()
    try:
        for idx in range(len(sample_idx)-1):
            seg_frame_embeddings = frame_embeddings[sample_idx[idx] : sample_idx[idx + 1]]
            for embedding in text_embedding:
                similarity = embedding @ seg_frame_embeddings.T
                new_frame_idx = sample_idx[idx] + similarity.argmax() + 1
                if new_frame_idx <=180: frame_idx.add(new_frame_idx)
    except Exception as e:
        print(e)

    return sorted(list(frame_idx))


if __name__ == "__main__":
    pass

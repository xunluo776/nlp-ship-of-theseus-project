def run_embedding_pipeline(DATASET_NAME="yelp"):
    
    import torch
    from transformers import AutoTokenizer, AutoModel
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    import pickle
    
    # ---------------------------------------------------------
    # CONFIG
    # ---------------------------------------------------------
    
    EMB_PICKLE = Path(f"bert_embeddings_{DATASET_NAME}_fp16.pkl")
    
    MODEL_NAME = "roberta-base"
    
    TEXT_COLUMNS = [
        "text_T0",
        "text_chatgpt",
        "text_palm",
        "text_dipper_low",
        "text_dipper",
        "text_dipper_high",
        "text_pegasus_slight",
        "text_pegasus_full",
    ]
    
    # Chunking parameters
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 256
    STRIDE = CHUNK_SIZE - CHUNK_OVERLAP
    
    # ---------------------------------------------------------
    # FILTER TO ONE DATASET
    # ---------------------------------------------------------
    
    paired_ds = paired_t1[paired_t1["dataset"] == DATASET_NAME].copy()
    print(f"Loaded {len(paired_ds)} rows from dataset '{DATASET_NAME}'")
    
    # Extract unique texts for this dataset only
    def extract_unique_texts(df, cols):
        texts = set()
        for col in cols:
            texts.update(df[col].dropna().astype(str).unique())
        return list(texts)
    
    unique_texts = extract_unique_texts(paired_ds, TEXT_COLUMNS)
    print(f"Found {len(unique_texts)} unique texts in '{DATASET_NAME}'")
    
    text_to_id = {txt: i for i, txt in enumerate(unique_texts)}

    # ---------------------------------------------------------
    # EARLY EXIT IF CACHE COMPLETE (NO WARNINGS)
    # ---------------------------------------------------------

    if EMB_PICKLE.exists():
        with open(EMB_PICKLE, "rb") as f:
            cache_obj = pickle.load(f)
            bert_cache = cache_obj["embeddings"]

        if len(bert_cache) == len(unique_texts):
            print("All embeddings already computed — nothing to do.")
            return
    else:
        bert_cache = {}

    # ---------------------------------------------------------
    # DEVICE + MODEL LOAD
    # ---------------------------------------------------------
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    # ---------------------------------------------------------
    # CHUNKED EMBEDDING FUNCTION (FP16)
    # ---------------------------------------------------------
    
    @torch.no_grad()
    def compute_chunked_embeddings_fp16(text):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )
    
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        seq_len = input_ids.size(0)
    
        chunk_embs = []
    
        # Short text → single chunk
        if seq_len <= CHUNK_SIZE:
            batch = {
                "input_ids": input_ids.unsqueeze(0).to(device),
                "attention_mask": attention_mask.unsqueeze(0).to(device),
            }
            outputs = model(**batch)
            token_embs = outputs.last_hidden_state.squeeze(0)
            chunk_embs.append(token_embs.to("cpu").half().numpy())
            return chunk_embs
    
        # Long text → sliding window
        start = 0
        while start < seq_len:
            end = min(start + CHUNK_SIZE, seq_len)
    
            ids_chunk = input_ids[start:end]
            mask_chunk = attention_mask[start:end]
    
            batch = {
                "input_ids": ids_chunk.unsqueeze(0).to(device),
                "attention_mask": mask_chunk.unsqueeze(0).to(device),
            }
            outputs = model(**batch)
            token_embs = outputs.last_hidden_state.squeeze(0)
    
            chunk_embs.append(token_embs.to("cpu").half().numpy())
    
            if end == seq_len:
                break
    
            start += STRIDE
    
        return chunk_embs
    
    # ---------------------------------------------------------
    # MAIN LOOP — RESUMABLE PER-DATASET CACHE
    # ---------------------------------------------------------
    
    print(f"Starting embedding computation for dataset '{DATASET_NAME}'...")
    
    for text in tqdm(unique_texts):
        tid = text_to_id[text]
    
        # Skip if already computed
        if tid in bert_cache:
            continue
    
        emb_chunks = compute_chunked_embeddings_fp16(text)
        bert_cache[tid] = emb_chunks
    
        # Save every 500 items for safety
        if len(bert_cache) % 500 == 0:
            with open(EMB_PICKLE, "wb") as f:
                pickle.dump(
                    {
                        "embeddings": bert_cache,
                        "text_to_id": text_to_id,
                        "model_name": MODEL_NAME,
                        "dtype": "float16",
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"Checkpoint saved ({len(bert_cache)} embeddings).")
    
    # Final save
    with open(EMB_PICKLE, "wb") as f:
        pickle.dump(
            {
                "embeddings": bert_cache,
                "text_to_id": text_to_id,
                "model_name": MODEL_NAME,
                "dtype": "float16",
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    
    print(f"Finished dataset '{DATASET_NAME}'. Saved to {EMB_PICKLE}")


# Default call
run_embedding_pipeline("yelp")

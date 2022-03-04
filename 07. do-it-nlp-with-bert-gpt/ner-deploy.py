# %% set hyperparameter
from ratsnlp.nlpbook.ner import NERDeployArguments

args = NERDeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_dir="nlpbook/checkpoint-ner",
    max_seq_length=64,
)
# %% load model
import torch
from transformers import BertConfig, BertForTokenClassification

fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_fpath,
    map_location=torch.device("cuda")
)

pt_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels = fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)

model = BertForTokenClassification(pt_model_config)
model.load_state_dict({k.replace("model.", ""): v for k,v in fine_tuned_model_ckpt['state_dict'].items()})
model.eval()

# %% 

for k,v in fine_tuned_model_ckpt['state_dict'].items():
    print(k, v)

# %% 
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)
# %% create label map
labels = [label.strip() for label in open(args.downstream_model_labelmap_fpath, mode="r").readlines()]
id_to_label = {}
for idx, label in enumerate(labels):
    if "PER" in label:
        label = "인명",
    elif "LOC" in label:
        label = "지명",
    elif "ORG" in label:
        label = "기관명",
    elif "DAT" in label:
        label = "시간",
    elif "DUR" in label:
        label = "기간",
    elif "MNY" in label:
        label = "통화",
    elif "PNT" in label:
        label = "비율",
    elif "NOH" in label:
        label = "기타 수량표현"
    elif "POH" in label:
        label = "기타"
    else:
        label = label
    
    id_to_label[idx] = label
    

# %% define inference function
def inference_fn(sentence):
    inputs = tokenizer(
        [sentence],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k,v in inputs.items()})
        probs = outputs.logits[0].softmax(dim=1)
        top_probs, preds = torch.topk(probs, dim=1, k=1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_tags = [id_to_label[pred.item()] for pred in preds]
        result = []
        for token, predicted_tag, top_prob in zip(tokens, predicted_tags, top_probs):
            if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                token_result = {
                    "token": token,
                    "predicted_tag": predicted_tag,
                    "top_prob": str(round(top_prob[0].item(), 4)),
                }
                result.append(token_result)
    
    return {
        "sentence": sentence,
        "result": result,
    }
        

# %%
from ratsnlp.nlpbook.ner import get_web_service_app
app = get_web_service_app(inference_fn)
app.run()
# %%

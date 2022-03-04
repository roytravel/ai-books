# %% load library
import torch
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer
from ratsnlp.nlpbook.classification import ClassificationDeployArguments

# %% define inference func
def inference_fn(sentence):
    inputs = tokenizer(
        [sentence],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k,v in inputs.items()})
        prob = outputs.logits.softmax(dim=1)
        pos_prob = round(prob[0][1].item(), 4)
        neg_prob = round(prob[0][0].item(), 4)
        pred = "긍정 (pos)" if torch.argmax(prob) == 1 else "부정 (neg)"
    
    return {
        'sentence': sentence,
        'prediction': pred,
        'pos_data' : f"긍정 {pos_prob}",
        'neg_data' : f"부정 {neg_prob}",
        'pos_width' : f"{pos_prob * 100}%",
        'neg_width' : f"{neg_prob * 100}%",
    }


# %% set hyperparameter
args = ClassificationDeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_dir="nlpbook/checkpoint-doccls",
    max_seq_length=128,
)

# %% load model
fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_fpath,
    map_location=torch.device("cuda")
)

pt_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_lables=fine_tuned_model_ckpt['state_dict']['model_classifier.bias'].shape.numel(),
)

model = BertForSequenceClassification(pt_model_config)
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
model.eval()

tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name, do_lower_case=False)
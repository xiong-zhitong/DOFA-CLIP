import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on timm>=0.9.8

hf_repo = "hf-hub:earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO"
model, preprocess = create_model_from_pretrained(hf_repo)
tokenizer = get_tokenizer(hf_repo)


def encode_image(model, image, wvs, normalize: bool = False):
    features = model.visual.trunk(image, wvs)
    return F.normalize(features, dim=-1) if normalize else features


image = Image.open("airplane.png")
image = preprocess(image).unsqueeze(0).cuda()

labels_list = ["A busy airport with many aeroplanes.", "Satellite view of Hohai university.", "Satellite view of sydney", "Many people in a stadium"]

text = tokenizer(labels_list, context_length=model.context_length)
text = text.cuda()
model = model.cuda()

with torch.no_grad(), torch.cuda.amp.autocast():
    wvs = torch.tensor([0.665, 0.560, 0.490]).cuda()
    image_features,_ = encode_image(model, image, wvs)
    text_features = model.encode_text(text)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    text_probs = torch.sigmoid(image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias)

zipped_list = list(zip(labels_list, [round(p.item(), 3) for p in text_probs[0]]))
print("Label probabilities: ", zipped_list)

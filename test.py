from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, os

model_path = "qol_classifier_fine_tuned"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load only if model.safetensors exists
model_file = os.path.join(model_path, "model.safetensors")
if not os.path.exists(model_file):
    raise FileNotFoundError("Trained model not found.")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)
model.eval()

inputs_1 = tokenizer("yesihavebeencarefulaboutwalkingoutsideasigetalotofpainihavetostopwalkingandhavetostopwalkingthenthepaingoesthensittingdownmassagingmytummythenstartwalkingagainiavoidgoingaroundsmallerspacesasmytummyst", return_tensors="pt", truncation=True, padding="max_length", max_length=256)
inputs_2 = tokenizer("okayheresapatientnarrativefocusingonthespecificdetailsprovidedandaimingforrealisticandvariedlimitationsitsjustfrustratingyouknowbeingunemployedishouldbeabletousethistimetogetthingsdonearoundthehousema", return_tensors="pt", truncation=True, padding="max_length", max_length=256)

with torch.no_grad():
    logits_1 = model(**inputs_1).logits
    logits_2 = model(**inputs_2).logits

print("Logits 1:", logits_1)
print("Logits 2:", logits_2)

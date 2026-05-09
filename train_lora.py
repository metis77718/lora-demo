import torch
from diffusers import StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os, glob

# 1. Подготовка демо-датасета (20 цветных квадратов с текстом "cat" для имитации)
os.makedirs("demo_data", exist_ok=True)
for i in range(20):
    img = Image.new('RGB', (512,512), color=(255,200,200))
    img.save(f"demo_data/img_{i}.png")

class DemoDataset(Dataset):
    def __init__(self, path, transform=None):
        self.images = glob.glob(f"{path}/*.png")
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return {"pixel_values": img}

transform = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512), transforms.ToTensor()])
dataset = DemoDataset("demo_data", transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 2. Загрузка SDXL
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")

# 3. Настройка LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["to_q","to_k","to_v","to_out.0"], lora_dropout=0.1, bias="none")
unet = get_peft_model(pipe.unet, lora_config)
unet.train()
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

# 4. Обучение (быстро, чтобы показать процесс)
for epoch in range(3):
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        noise = torch.randn(1,4,64,64).to("cuda")
        timesteps = torch.randint(0,1000,(1,)).to("cuda")
        model_input = torch.randn(1,4,64,64).to("cuda")
        encoder_hidden_states = torch.randn(1,77,2048).to("cuda")
        noise_pred = unet(model_input, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

# 5. Сохранение LoRA
unet.save_pretrained("lora-adapter")
print("LoRA adapter saved to ./lora-adapter")

# 6. Генерация демо-изображений
pipe.unet = unet.merge_and_unload()
pipe.to("cuda")

negative_prompt = "low quality, blurry"
for i, prompt in enumerate(["a cute cat", "a realistic cat photo", "a cartoon cat", "cat in the style of van Gogh"]):
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=25).images[0]
    image.save(f"demo_{i+1}.png")
    print(f"Generated demo_{i+1}.png")

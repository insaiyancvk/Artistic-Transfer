import streamlit as st
from streamlit_image_select import image_select
from diffusers import StableDiffusionPipeline
from dependencies import \
                  TransformerNetworkNN, \
                  VGG16, \
                  device, \
                  load_image, \
                  itot, \
                  STYLE_IMAGE_PATH, \
                  imagenet_neg_mean, \
                  BATCH_SIZE, \
                  gram, \
                  optim, \
                  ADAM_LR, \
                  train_loader, \
                  CONTENT_WEIGHT, \
                  STYLE_WEIGHT, \
                  SAVE_MODEL_EVERY, \
                  SAVE_MODEL_PATH, \
                  NUM_EPOCHS
import torch.nn as nn
import torch, os, glob, time
from PIL import Image 


num_cols = 3
num_rows = 3

st.title("Text Based Style Transfer On Segmented Images")

with st.expander("Generate a Style Image"):
  PROMPT = st.text_input("",placeholder="Enter a prompt to generate images")
  st.write('Example: "Japanese abstract art illustration"')

  gen_imgs = []

  if not os.path.exists('Generated_images'):
    os.mkdir('Generated_images')

  if len(PROMPT) != 0:
    if not os.path.exists(f"Generated_images/{PROMPT}"):
      with st.info('Setting up Stable Diffusion Pipeline', icon="ℹ️"):
        pipe = StableDiffusionPipeline.from_pretrained(
          "runwayml/stable-diffusion-v1-5",
          revision="fp16",
          torch_dtype=torch.float16,
          low_cpu_mem_usage = True,
        ).to("cuda")

      prompt = [PROMPT] * num_cols # input from user

      with st.spinner('Generating images...'):
          for _ in range(num_rows):
            images = pipe(prompt).images
            gen_imgs.extend(images)

      os.mkdir(f'Generated_images/{PROMPT}')

      for x, i in enumerate(gen_imgs):
        i.save(f'Generated_images/{PROMPT}/{x}.jpg')

    if len(gen_imgs) == 0:
      for f in glob.iglob(f"Generated_images/{PROMPT}/*"):
          gen_imgs.append(Image.open(f))
    all_images = gen_imgs

    SELECTED_IMAGE = image_select(
        label="Select an image",
        images=all_images,
        captions=[str(i) for i in range(1,10)],
        use_container_width = False
    )

    if SELECTED_IMAGE:
      st.write("Selected Image:")
      st.image(SELECTED_IMAGE)
      SELECTED_IMAGE.save('style.jpg')

with st.expander("Train Transformer network"):
  
  TRAIN_TIME = st.slider('Select a training time', 30, 120, 5)
  st.write("Set training time (min). Higher the training time, better the style transfer by the network.")

  try:
    del pipe
  except:
    pass
  torch.cuda.empty_cache()
  st.write('clearing cache data')

  # Implement a button click for the following - "Initialize Network and dataset"
  # Automate the below blocks of code
  # -----------------------------------------------------------------------------------------------------------------------
                                      # Stuff required to train a transformer network from scratch
  TransformerNetwork = TransformerNetworkNN().to(device)
  if os.path.exists(f"/content/drive/MyDrive/gen_imgs/{PROMPT}/style.pth"):
    TransformerNetwork.load_state_dict(torch.load(f"/content/drive/MyDrive/gen_imgs/{PROMPT}/style.pth", map_location=device))
  VGG = VGG16('/content/vgg16.pth').to(device)
  style_image = load_image(STYLE_IMAGE_PATH)
  style_tensor = itot(style_image).to(device)
  style_tensor = style_tensor.add(imagenet_neg_mean)
  B, C, H, W = style_tensor.shape
  style_features = VGG(style_tensor.expand([BATCH_SIZE, C, H, W]))
  style_gram = {}
  for key, value in style_features.items():
      style_gram[key] = gram(value)

  # Optimizer settings
  optimizer = optim.Adam(TransformerNetwork.parameters(), lr=ADAM_LR)
  # -----------------------------------------------------------------------------------------------------------------------
                                      # If there's pretrained weights with the user, import that
  # TRAIN_IMAGE_SIZE = 512
  # DATASET_PATH = "/content/train"
  # SEED = 68

  # device = ("cuda" if torch.cuda.is_available() else "cpu")

  # PROMPT = "Kazimir Malevich art in Retrowave style" #@param {type:"string"}
  # # Load networks
  # TransformerNetwork = TransformerNetworkNN().to(device)
  # if os.path.exists(f"/content/drive/MyDrive/gen_imgs/{PROMPT}/style.pth"):
  #   print("Loading the pretrained weights")
  #   TransformerNetwork.load_state_dict(torch.load(f"/content/drive/MyDrive/gen_imgs/{PROMPT}/style.pth", map_location=device))
  # -----------------------------------------------------------------------------------------------------------------------

  # TODO: automate the above two steps

  MODEL_NAME = "style"
  DRIVE_PATH = f"/content/drive/MyDrive/gen_imgs/{PROMPT}"

  # -----------------------------------------------------------------------------------------------------------------------
                                                  # Training loop

  batch_count = 1
  TRAIN_TIME *= 60
  start_time = time.time()
  progress_bar = st.progress(0)

  for content_batch, _ in train_loader:
    curr_batch_size = content_batch.shape[0]
    optimizer.zero_grad()
    
    content_batch = content_batch[:,[2,1,0]].to(device)
    generated_batch = TransformerNetwork(content_batch)
    content_features = VGG(content_batch.add(imagenet_neg_mean))
    generated_features = VGG(generated_batch.add(imagenet_neg_mean))

    # Content Loss
    MSELoss = nn.MSELoss().to(device)
    content_loss = CONTENT_WEIGHT * MSELoss(content_features['relu2_2'], generated_features['relu2_2'])

    # Style Loss
    style_loss = 0
    for key, value in generated_features.items():
        s_loss = MSELoss(gram(value), style_gram[key][:curr_batch_size])
        style_loss += s_loss
    style_loss *= STYLE_WEIGHT

    # Total Loss
    total_loss = content_loss + style_loss

    # Backprop and Weight Update
    total_loss.backward()
    optimizer.step()

    if (((batch_count-1)%SAVE_MODEL_EVERY == 0) or (batch_count==NUM_EPOCHS*len(train_loader))):
        torch.save(TransformerNetwork.state_dict(), f"{DRIVE_PATH}/{MODEL_NAME}.pth")
    batch_count+=1
    progress_bar.progress((time.time()-start_time)/TRAIN_TIME)
    
    if (time.time()-start_time) > TRAIN_TIME:
      break

  # -----------------------------------------------------------------------------------------------------------------------

  """ TODO: 
            - Initialize transformer network
            - Move all classes and helper functions to a different file
            - Training code
  
  """


with st.expander("Segment an Image"):
  image_file = st.file_uploader("Upload an image to segment", type=["png","jpg","jpeg"])

  if image_file is not None:

      file_details = {"filename":image_file.name, "filetype":image_file.type,
                            "filesize":image_file.size}
      st.write(file_details)
      with open(os.path.join("content.jpg",image_file.name),"wb") as f: 
        f.write(image_file.getbuffer())

with st.expander("Style transfer on the segment"):
  st.write("To be implemented")

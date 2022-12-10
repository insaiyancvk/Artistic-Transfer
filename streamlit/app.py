import streamlit as st
from streamlit_image_select import image_select
from diffusers import StableDiffusionPipeline
import torch, os, glob
from PIL import Image 

num_cols = 3
num_rows = 3

st.title("Text Based Style Transfer On Segmented Images")

PROMPT = st.text_input("",placeholder="Enter a prompt to generate images")
st.write('Example: "Japanese abstract art illustration"')

gen_imgs = []

if not os.path.exists('Generated_images'):
  os.mkdir('Generated_images')

if len(PROMPT) != 0:
  
  if not os.path.exists(f"Generated_images/{PROMPT}"):
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
state = st.button("Clear cache")
st.write(state)

if state:
  try:
    del pipe
  except:
    pass
  torch.cuda.empty_cache()
  st.write('Cache data cleared')

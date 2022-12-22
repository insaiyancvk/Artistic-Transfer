import streamlit as st
from streamlit_image_select import image_select
from diffusers import StableDiffusionPipeline
from time import sleep
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
                  NUM_EPOCHS, \
                  segments, \
                  ttoi, \
                  saveimg
                  
import torch.nn as nn
import torch, os, glob, time, cv2, shutil
from PIL import Image 

num_cols = 3
num_rows = 3

st.title("Text Based Style Transfer On Segmented Images")

if 'is_expanded_first' not in st.session_state:
    st.session_state['is_expanded_first'] = True
first_container = st.expander("Generate a Style Image", expanded=st.session_state['is_expanded_first'])
with first_container:
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
      
      if not os.path.exists(f"/content/drive/MyDrive/gen_imgs/{PROMPT}"):
        os.mkdir(f"/content/drive/MyDrive/gen_imgs/{PROMPT}")
      
      for j,i in enumerate(gen_imgs):
        i.save(f"/content/drive/MyDrive/gen_imgs/{PROMPT}/{j}.jpg")

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
      st.header("Selected Image:")
      st.image(SELECTED_IMAGE)
    if st.button("Proceed further"):
      with st.spinner('Saving the style image'):
        SELECTED_IMAGE.save('style.jpg')
  sleep(2)
  st.session_state['is_expanded_first'] = False


if os.path.exists("/content/train"):
  TransformerNetwork = TransformerNetworkNN().to(device)
  st.session_state['is_expanded_sec'] = False
  if 'is_expanded_sec' not in st.session_state:
    st.session_state['is_expanded_sec'] = True
  sec_container = st.expander("Transformer network", expanded=st.session_state['is_expanded_sec'])
  with sec_container:
    if os.path.exists(f'Generated_images/{PROMPT}'):
      try:
        del pipe
        torch.cuda.empty_cache()
      except:
        pass
    
    option = st.selectbox(
      'Select an option to proceed further',
      ('Load a pretrained network', 'Train a transformer network'))

    if option == "Load a pretrained network":
      pretrained_prompt = st.text_input("",placeholder="Enter the prompt was previously used for training the network")
      if os.path.exists(f"/content/drive/MyDrive/gen_imgs/{pretrained_prompt}/style.pth"):
        TRAIN_IMAGE_SIZE = 512
        DATASET_PATH = "/content/train"
        SEED = 68
        device = ("cuda" if torch.cuda.is_available() else "cpu")

        st.write("Loading the pretrained weights")
        TransformerNetwork.load_state_dict(torch.load(f"/content/drive/MyDrive/gen_imgs/{pretrained_prompt}/style.pth", map_location=device))
    
    elif option == "Train a transformer network":
      if os.path.exists("style.jpg"):
        
        TRAIN_TIME = st.slider('Select a training time', 30, 120, 5)
        st.write("Set training time (min). Higher the training time, better the style transfer by the network.")
        
        if st.button("Start training"):
          st.session_state['is_expanded_first'] = False
          if os.path.exists(f"/content/drive/MyDrive/gen_imgs/{PROMPT}/style.pth"):
            with st.spinner("Pretrained weights found. Loading it"):
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
          MODEL_NAME = "style"
          DRIVE_PATH = f"/content/drive/MyDrive/gen_imgs/{PROMPT}"
          batch_count = 1
          TRAIN_TIME *= 60
          start_time = time.time()
          progress_bar = st.progress(0)
          latest_iteration = st.empty()

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
            time_passed = time.time()-start_time
            progress_time = time_passed/TRAIN_TIME
            if progress_time < 1:
              latest_iteration.text(f"{(time_passed/60):.1f}/{TRAIN_TIME//60} mins")
              progress_bar.progress(progress_time)
            else:
              progress_bar.progress(1)

            if (time.time()-start_time) > TRAIN_TIME:
              latest_iteration.success('Training completed!', icon="✅")
              break

      else:
        st.write("Generate and choose a style image")

INDEX = None
solutions = None

with st.expander("Segment an Image"):

  image_file = st.file_uploader("Upload an image to segment", type=["png","jpg","jpeg"])
  
  if image_file is not None:
    with open("content.jpg","wb") as f: 
      f.write(image_file.getbuffer())

  elif image_file is None:
    if os.path.exists('content.jpg'):
      os.remove('content.jpg')
    if os.path.exists('utils/segment.jpg'):
      os.remove("utils/segment.jpg")
      solutions = []
    if os.path.exists('segments'):
      shutil.rmtree('segments')

  if os.path.exists("content.jpg") and (not os.path.exists("segments")):
    if st.button("Generate segments"):
      with st.spinner('Generating segments...'):
        solutions = segments()
        
      st.write(type(solutions))

    segimgs = []
    for f in glob.iglob(f"segments/*"):
      segimgs.append(Image.open(f))
    
  if len(solutions)>0:
    INDEX = image_select(
        label="Select a segment",
        images=segimgs,
        return_value="index",
        captions=[str(i) for i in range(1,len(segimgs)+1)],
        use_container_width = False
    )
    
    if segimgs[INDEX]:
      st.header("Selected Segment:")
      segimgs[INDEX].save('utils/segment.jpg')
      st.image("utils/segment.jpg")

with st.expander("Style transfer on the segment"):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  st.write(solutions)
  if os.path.exists("utils/segment.jpg"):
    if st.button("Apply style transfer"):
      with torch.no_grad():
        torch.cuda.empty_cache()
        content_image = load_image("utils/segment.jpg")
        content_tensor = itot(content_image).to(device)
        TransformerNetwork = TransformerNetwork.to(device)
        generated_tensor = TransformerNetwork(content_tensor)
        generated_image = ttoi(generated_tensor.detach())
        saveimg(generated_image, "utils/segment_style.jpg")

      with st.spinner("Applying style transfer"):
        nimg = cv2.imread('content.jpg')[:, :, ::-1]
        styleimg = cv2.imread('utils/segment_style.jpg')[:,:,::-1]
        for i in solutions[INDEX]:
          nimg[i[0]][i[1]] = styleimg[i[0]][i[1]]

        img = Image.fromarray(nimg.astype('uint8'))
        img.save('Final Transformation.png')
  else:
    st.write("Upload a content image to generate segments")

if os.path.exists('Final Transformation.png'):

  if 'final_trans' not in st.session_state:
    st.session_state.final_trans = 'noimg'
  st.image("Final Transformation.png")
  st.session_state.final_trans = 'yeimg'

  if st.session_state.final_trans == 'yeimg':
    try:
      if os.path.exists(f"/content/drive/MyDrive/gen_imgs/{PROMPT}"):
        if st.button("Save content, style, transformed images to drive"):
          all_imgs =  [
            'Final Transformation.png',
            'content.jpg',
            'style.jpg'
        ]
        for i in all_imgs:
          os.system(f"!cp {i} /content/drive/MyDrive/gen_imgs/{PROMPT}")
    except NameError:
      pass

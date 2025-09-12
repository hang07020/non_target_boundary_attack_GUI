from __future__ import print_function
import numpy as np
from PIL import Image
import streamlit as st
import io 

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

def orthogonal_perturbation(delta, prev_sample, target_sample):
    prev_sample = prev_sample.reshape(224, 224, 3)
    perturb = np.random.randn(224, 224, 3)
    perturb /= get_diff(perturb, np.zeros_like(perturb))
    perturb *= delta * np.mean(get_diff(target_sample, prev_sample))
    diff = (target_sample - prev_sample).astype(np.float32)
    diff /= get_diff(target_sample, prev_sample)
    diff = diff.reshape(3, 224, 224)
    perturb = perturb.reshape(3, 224, 224)
    for i, channel in enumerate(diff):
        perturb[i] -= np.dot(perturb[i], channel) * channel
    mean = [103.939, 116.779, 123.68]
    perturb = perturb.reshape(224, 224, 3)
    overflow = (prev_sample + perturb) - np.concatenate((np.ones((224, 224, 1)) * (255. - mean[0]), np.ones((224, 224, 1)) * (255. - mean[1]), np.ones((224, 224, 1)) * (255. - mean[2])), axis=2)
    perturb -= overflow * (overflow > 0)
    underflow = np.concatenate((np.ones((224, 224, 1)) * (0. - mean[0]), np.ones((224, 224, 1)) * (0. - mean[1]), np.ones((224, 224, 1)) * (0. - mean[2])), axis=2) - (prev_sample + perturb)
    perturb += underflow * (underflow > 0)
    return perturb

def forward_perturbation(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample).astype(np.float32)
    perturb /= get_diff(target_sample, prev_sample)
    perturb *= epsilon
    return perturb

def postprocess_for_display(sample):
    sample = sample.reshape(224, 224, 3)
    mean = [103.939, 116.779, 123.68]
    sample = sample.copy()
    sample[..., 0] += mean[0]
    sample[..., 1] += mean[1]
    sample[..., 2] += mean[2]
    sample = sample[..., ::-1]
    sample = np.clip(sample, 0, 255).astype(np.uint8)
    return Image.fromarray(sample)

def preprocess(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_diff(sample_1, sample_2):
    sample_1 = sample_1.reshape(3, 224, 224)
    sample_2 = sample_2.reshape(3, 224, 224)
    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
    return np.array(diff)

def get_sc_img(option, original_image=None):
    """選択されたオプションに基づいて画像を生成または読み込み"""
    height, width = 224, 224

    if option == "Noise":
        try:
            img = Image.open('noise.png').resize((height, width))
            return img
        except FileNotFoundError:
            st.error("'noise.png' not found. Please add it to the same folder. Using blue as default.")
            option = "Blue"
    
    if option == "Mean":
        if original_image:
            image_array = np.array(original_image)
            mean_r = int(np.mean(image_array[:, :, 0]))
            mean_g = int(np.mean(image_array[:, :, 1]))
            mean_b = int(np.mean(image_array[:, :, 2]))
            return Image.new('RGB', (width, height), (mean_r, mean_g, mean_b))
        else:
            st.error("Original image is needed to calculate mean. Using blue as default.")
            option = "Blue"

    if option == "Red":
        red, green, blue = 255, 0, 0
    elif option == "Green":
        red, green, blue = 0, 255, 0
    else:  # "Blue" or default
        red, green, blue = 0, 0, 255
    
    return Image.new('RGB', (width, height), (red, green, blue))


def run_attack(original_image_pil, max_steps, color, progress_bar, adversarial_placeholder, download_placeholder, caption_placeholder):
    classifier = load_model()
    
    initial_sample = preprocess(original_image_pil)
    target_sample = preprocess(get_sc_img(color, original_image_pil))
    
    attack_class = np.argmax(classifier.predict(initial_sample))
    adversarial_sample = initial_sample
    
    epsilon = 0.02
    delta = 0.1

    # Move first step to the boundary
    while True:
        trial_sample = adversarial_sample + forward_perturbation(epsilon * np.mean(get_diff(adversarial_sample, target_sample)), adversarial_sample, target_sample)
        prediction = classifier.predict(trial_sample.reshape(1, 224, 224, 3))
        if np.argmax(prediction) == attack_class:
            epsilon /= 0.9
        else:
            adversarial_sample = trial_sample
            break
        if epsilon > 100:
             st.error("Could not find the decision boundary with this option.")
             return postprocess_for_display(initial_sample)

    target_class = np.argmax(classifier.predict(adversarial_sample))

    # Main attack loop
    for n_steps in range(max_steps):

        # ... (attack logic remains the same) ...
        for _ in range(10):
            trial_samples = [adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, initial_sample) for _ in range(10)]
            predictions = np.argmax(classifier.predict(np.array(trial_samples).reshape(-1, 224, 224, 3)), axis=1)
            d_score = np.mean(predictions == target_class)
            
            if d_score > 0.0:
                if d_score < 0.3: delta *= 0.9
                elif d_score > 0.7: delta /= 0.9
                adversarial_sample = np.array(trial_samples)[np.where(predictions == target_class)[0][0]]
                break 
            else:
                delta *= 0.9
        for _ in range(10):
            trial_sample = adversarial_sample + forward_perturbation(epsilon * np.mean(get_diff(adversarial_sample, initial_sample)), adversarial_sample, initial_sample)
            prediction = classifier.predict(trial_sample.reshape(1, 224, 224, 3))
            if np.argmax(prediction) == target_class:
                adversarial_sample = trial_sample
                epsilon /= 0.5
                break
            else:
                epsilon *= 0.5

        # Update GUI
        current_step = n_steps + 1
        progress_bar.progress(current_step / max_steps, text=f"Step {current_step}/{max_steps}")
        if current_step == 1 or current_step % 10 == 0 or current_step == max_steps:
            current_adv_pil = postprocess_for_display(adversarial_sample)
            mse = np.mean(get_diff(initial_sample, adversarial_sample))
            adversarial_placeholder.image(current_adv_pil, caption=f"Adversarial (Step {current_step}) | MSE: {mse:.4f}", use_container_width=True)

            # ステップ1以降でダウンロードボタンと注釈を表示
            buf = io.BytesIO()
            current_adv_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            download_placeholder.download_button(
                label=f"Download (Step {current_step})",
                data=byte_im,
                file_name=f"adversarial_step_{current_step}.png",
                mime="image/png",
                key=f"download_step_{current_step}"
            )
            caption_placeholder.caption("Note: Downloading will stop the attack process.")
    
    progress_bar.empty()
    final_image = postprocess_for_display(adversarial_sample)
    final_mse = np.mean(get_diff(initial_sample, adversarial_sample))
    adversarial_placeholder.image(final_image, caption=f"Final Adversarial | MSE: {final_mse:.4f}", use_container_width=True)
    
    buf = io.BytesIO()
    final_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    download_placeholder.download_button(
        label="Download Final Image",
        data=byte_im,
        file_name="adversarial_final.png",
        mime="image/png",
        key="download_final"
    )
    caption_placeholder.empty()

    return final_image